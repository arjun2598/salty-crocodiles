"""One key in, LLM out. Standard library only -- no SDK to install.

Both APIs are a single HTTP POST, so this uses urllib. numpy is the harness's
only real dependency.

Config comes from lab/.env (gitignored), or from real environment variables,
which win over the file:

    LLM_API_KEY=sk-ant-...        # Anthropic
    LLM_API_KEY=sk-or-...         # OpenRouter
    LLM_API_KEY=sk-proj-...       # OpenAI
    LLM_API_KEY=<anything>
    LLM_BASE_URL=https://your-endpoint/v1

Set LLM_MODEL to pick the model. An unrecognised key prefix without an
explicit LLM_BASE_URL is refused rather than guessed -- guessing would mail a
credential to a third party.

The `anthropic` package is used only if it is already installed AND no key is
set, so that an `ant auth login` OAuth profile still works. It is never
required.
"""
from __future__ import annotations

import json
import os
import ssl
import urllib.error
import urllib.request

TIMEOUT_S = 300


def _ssl_context():
    """TLS trust.

    LLM_CA_BUNDLE=/path/to/ca.pem  -- the right fix when your network uses a
      TLS-intercepting proxy, or the endpoint's CA is not in Python's bundle.
    LLM_INSECURE_TLS=1             -- last resort. Turns OFF verification,
      which means anything on the path can read your API key. Use only on a
      network you trust, and rotate the key afterwards if you are unsure.
    """
    bundle = os.environ.get("LLM_CA_BUNDLE")
    if bundle:
        return ssl.create_default_context(cafile=bundle)
    if os.environ.get("LLM_INSECURE_TLS") == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    # Prefer certifi when present -- this is what requests/httpx (and so the
    # openai SDK) use. urllib otherwise falls back to whatever OpenSSL was
    # built against, e.g. Homebrew's bundle, which is often missing CAs that
    # certifi has. Same network, different trust store, different outcome.
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def load_env(*paths: str) -> None:
    """Minimal .env loader -- no dependency, and a real exported variable
    always wins over the file. KEY=value per line, # comments, optional quotes.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    for path in paths or (os.path.join(here, ".env"),
                          os.path.join(os.path.dirname(here), ".env")):
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key, val = key.strip(), val.strip().strip("'\"")
                if key and key not in os.environ:      # exports win
                    os.environ[key] = val


load_env()

# Key prefixes we can route with confidence. Anything else REQUIRES an
# explicit LLM_BASE_URL.
_KNOWN = [("sk-ant-",  None),                                  # Anthropic
          ("sk-or-",   "https://openrouter.ai/api/v1"),
          ("sk-proj-", "https://api.openai.com/v1"),
          ("gsk_",     "https://api.groq.com/openai/v1")]

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5"
DEFAULT_OPENAI_MODEL = "gpt-5"


def _post(url: str, headers: dict, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), method="POST",
        headers={"content-type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_S,
                                    context=_ssl_context()) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:800]
        raise RuntimeError(f"HTTP {e.code} from {url}\n{body}") from None
    except urllib.error.URLError as e:
        if "CERTIFICATE_VERIFY_FAILED" not in str(e.reason):
            raise
        raise RuntimeError(
            f"TLS verification failed for {url}\n  {e.reason}\n"
            "  Either your network intercepts TLS, or this CA is not in "
            "Python's bundle.\n"
            "  Fix properly:  LLM_CA_BUNDLE=/path/to/ca.pem  in lab/.env\n"
            "  Last resort :  LLM_INSECURE_TLS=1  (disables verification -- "
            "your key is then\n                 readable by anything on the "
            "path; rotate it afterwards)") from None


class AnthropicProvider:
    """POST /v1/messages. Uses the anthropic SDK only when it is installed and
    no key is set, so an `ant auth login` profile still resolves."""

    def __init__(self, model, api_key=None):
        self.model = model or DEFAULT_ANTHROPIC_MODEL
        self.name = f"anthropic:{self.model}"
        self._key = api_key
        self._sdk = None
        if not api_key:
            try:
                import anthropic
                self._sdk = anthropic.Anthropic()
            except ImportError:
                raise SystemExit(
                    "no LLM_API_KEY set, and the `anthropic` package is not "
                    "installed to resolve an\n`ant auth login` profile. Put a "
                    "key in lab/.env, or pip install anthropic.") from None

    def complete(self, system, messages, max_tokens):
        if self._sdk is not None:
            r = self._sdk.messages.create(model=self.model, max_tokens=max_tokens,
                                          system=system, messages=messages)
            return ("".join(b.text for b in r.content if b.type == "text"),
                    r.usage.input_tokens, r.usage.output_tokens)
        d = _post("https://api.anthropic.com/v1/messages",
                  {"x-api-key": self._key, "anthropic-version": "2023-06-01"},
                  {"model": self.model, "max_tokens": max_tokens,
                   "system": system, "messages": messages})
        text = "".join(b.get("text", "") for b in d.get("content", [])
                       if b.get("type") == "text")
        u = d.get("usage", {})
        return text, u.get("input_tokens", 0), u.get("output_tokens", 0)


class OpenAICompatProvider:
    """POST {base_url}/chat/completions -- OpenAI, OpenRouter, Gemini's compat
    endpoint, Groq, DeepSeek, Together, vLLM, a university research endpoint,
    LM Studio. They all speak this shape."""

    def __init__(self, model, api_key, base_url):
        self.model = model or DEFAULT_OPENAI_MODEL
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key
        self.name = f"{base_url}:{self.model}"

    def complete(self, system, messages, max_tokens):
        body = {"model": self.model,
                "messages": [{"role": "system", "content": system}] + list(messages),
                "max_tokens": max_tokens}
        headers = {"authorization": f"Bearer {self._key}"}
        try:
            d = _post(self._url, headers, body)
        except RuntimeError as exc:
            if "max_completion_tokens" not in str(exc):
                raise                      # reasoning models reject max_tokens
            body.pop("max_tokens")
            body["max_completion_tokens"] = max_tokens
            d = _post(self._url, headers, body)

        text = (d["choices"][0].get("message") or {}).get("content") or ""
        u = d.get("usage") or {}
        return text, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def build(api_key: str | None = None, base_url: str | None = None,
          model: str | None = None):
    key = api_key or os.environ.get("LLM_API_KEY") \
        or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base = base_url or os.environ.get("LLM_BASE_URL")
    model = model or os.environ.get("LLM_MODEL")

    # an explicit endpoint always wins, unless it IS Anthropic
    if base and "api.anthropic.com" not in base:
        if not key:
            raise SystemExit("LLM_BASE_URL is set but LLM_API_KEY is not")
        return OpenAICompatProvider(model, key, base)

    if not key:
        return AnthropicProvider(model, None)

    for prefix, url in _KNOWN:
        if key.startswith(prefix):
            return (AnthropicProvider(model, key) if url is None
                    else OpenAICompatProvider(model, key, url))

    raise SystemExit(
        f"LLM_API_KEY starts with {key[:6]!r}, which is not a prefix this can "
        "route.\nSet LLM_BASE_URL so the key goes to the right place -- "
        "without it your\nkey would be sent to whatever endpoint was guessed. "
        "For example:\n\n"
        "    LLM_BASE_URL=https://api.swissai.cscs.ch/v1\n"
        "    LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct\n")
