#!/usr/bin/env bash
set -Eeuo pipefail

commit="${1:?commit SHA is required}"
app_dir="$(cd "$(dirname "$0")/.." && pwd)"
venv_dir="${VENV_DIR:-$app_dir/.venv}"
backup_dir="${BACKUP_DIR:-$app_dir/.deploy-backups}"

cd "$app_dir"
mkdir -p "$backup_dir"

# Keep runtime data and server-side analysis artifacts outside Git changes.
if [[ -f data/state.json ]]; then
  cp data/state.json "$backup_dir/state-$(date -u +%Y%m%dT%H%M%SZ).json"
fi

git fetch --prune origin main
git cat-file -e "$commit^{commit}"
git checkout --force "$commit"

python3 -m venv "$venv_dir" 2>/dev/null || true
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r requirements.txt

if command -v systemctl >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; then
  systemctl --user daemon-reload
  systemctl --user restart gnss-results.service
  systemctl --user restart gnss-worker.service
else
  printf 'User systemd is unavailable; leaving services unchanged.\n'
fi

curl --fail --silent --show-error http://127.0.0.1:${RESULTS_PORT:-8001}/ >/dev/null

printf 'Deployed %s\n' "$commit"
