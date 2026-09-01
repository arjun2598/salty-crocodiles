#!/usr/bin/env bash
# One-time setup. Idempotent -- safe to re-run.
#   bash lab/setup.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
echo "repo: $ROOT"

# ---- 1. dependencies (assumed present; we only check) ----------------------
echo "==> checking dependencies"
python3 - <<'PY'
import importlib.util, sys
if not importlib.util.find_spec("numpy"):
    sys.exit("    MISSING: numpy  ->  pip install numpy")
print("    numpy ok -- that is the only dependency")
PY

# ---- 2. data ---------------------------------------------------------------
if [ ! -d KuaiRand-Pure/data ]; then
  echo "==> downloading KuaiRand-Pure (~250 MB)"
  curl -L -O https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz
  tar -xzf KuaiRand-Pure.tar.gz && rm -f KuaiRand-Pure.tar.gz
else
  echo "==> KuaiRand-Pure/data already present"
fi

# ---- 3. the test wall ------------------------------------------------------
if [ -f lab/data/trainval/log_standard_4_22_to_5_08_pure.csv ]; then
  echo "==> lab/data/trainval already built"
else
  echo "==> building lab/data/trainval (the test wall)"
  mkdir -p lab/data/trainval
  cp KuaiRand-Pure/data/user_features_pure.csv \
     KuaiRand-Pure/data/video_features_basic_pure.csv \
     KuaiRand-Pure/data/video_features_statistic_pure.csv \
     KuaiRand-Pure/data/log_standard_4_08_to_4_21_pure.csv \
     lab/data/trainval/
  python3 - <<'PY'
import csv, sys
jobs = [('log_standard_4_22_to_5_08_pure.csv', 124909),
        ('log_random_4_22_to_5_08_pure.csv',   288338)]
bad = False
for name, expect in jobs:
    kept = 0
    with open(f'KuaiRand-Pure/data/{name}') as fi, \
         open(f'lab/data/trainval/{name}', 'w', newline='') as fo:
        r = csv.DictReader(fi)
        w = csv.DictWriter(fo, fieldnames=r.fieldnames); w.writeheader()
        for row in r:
            if int(row['date']) <= 20220428:   # the valid window ends here
                w.writerow(row); kept += 1
    ok = kept == expect
    bad |= not ok
    print(f'    {name}: {kept} rows (expected {expect}) [{"OK" if ok else "MISMATCH"}]')
if bad:
    sys.exit('trainval built wrong -- every validation number would be meaningless')
PY
fi

# ---- 4. .env ---------------------------------------------------------------
if [ -f lab/.env ]; then
  echo "==> lab/.env already exists, leaving it alone"
else
  echo "==> writing lab/.env template -- put your key in it"
  cat > lab/.env <<'ENV'
# lab/.env -- gitignored. NEVER commit this file.
# Routed by key prefix: sk-ant- Anthropic / sk-or- OpenRouter /
# sk-proj- OpenAI / gsk_ Groq. Any other prefix REQUIRES LLM_BASE_URL below.
LLM_API_KEY=

# Custom endpoint (research server, vLLM, LM Studio). Uncomment BOTH.
# LLM_BASE_URL=https://api.swissai.cscs.ch/v1
# LLM_MODEL=meta-llama/Llama-3.3-70B-Instruct
ENV
fi

# ---- 5. verify -------------------------------------------------------------
echo "==> self-check"
python3 lab/selfcheck.py --full || true

cat <<'MSG'

Next: put your key in lab/.env, then
  python3 lab/selfcheck.py      # should be all [ok]
Then read SETUP.md.
MSG
