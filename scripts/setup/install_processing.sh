#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQ_FILE="${ROOT_DIR}/scripts/process/requirements.txt"

if [[ ! -f "${REQ_FILE}" ]]; then
    echo "Could not find ${REQ_FILE}" >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No active virtual environment detected. Activate your venv first." >&2
    exit 1
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${REQ_FILE}"

cat <<'EOF'
Installed Python processing requirements.

You still need these external tools for full processing workflows:
  - mmseqs
  - redis

Reference:
  - https://github.com/soedinglab/mmseqs2#installation
  - https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/
EOF
