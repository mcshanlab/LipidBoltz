#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ ! -f "${ROOT_DIR}/pyproject.toml" ]]; then
    echo "Could not find pyproject.toml from ${ROOT_DIR}" >&2
    exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    echo "No active virtual environment detected. Activate your venv first." >&2
    exit 1
fi

extras=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cuda)
            extras+=("cuda")
            ;;
        --test)
            extras+=("test")
            ;;
        -h|--help)
            cat <<'EOF'
Usage: scripts/setup/install_boltz.sh [--cuda] [--test]

Installs the local project in editable mode using dependencies from pyproject.toml.

Options:
  --cuda    Install CUDA extras declared by the project
  --test    Install test extras declared by the project
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
    shift
done

extras_csv=""
if [[ ${#extras[@]} -gt 0 ]]; then
    extras_csv="["
    for extra in "${extras[@]}"; do
        if [[ "${extras_csv}" != "[" ]]; then
            extras_csv+=","
        fi
        extras_csv+="${extra}"
    done
    extras_csv+="]"
fi

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e "${ROOT_DIR}${extras_csv}"

echo "Installed boltz from ${ROOT_DIR}${extras_csv}"
