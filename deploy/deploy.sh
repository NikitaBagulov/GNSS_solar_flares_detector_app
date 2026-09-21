#!/usr/bin/env bash
set -Eeuo pipefail

commit="${1:?commit SHA is required}"
app_dir="$(cd "$(dirname "$0")/.." && pwd)"
venv_dir="${VENV_DIR:-$app_dir/.venv}"
backup_dir="${BACKUP_DIR:-$app_dir/.deploy-backups}"
service_prefix="${GNSS_SERVICE_PREFIX:-gnss}"
previous_commit=""

systemctl_cmd() {
  if [[ "${EUID}" -eq 0 ]]; then
    systemctl "$@"
  else
    sudo -n systemctl "$@"
  fi
}

service_restart() {
  systemctl_cmd restart "${service_prefix}-results.service"
  systemctl_cmd restart "${service_prefix}-worker.service"
}

health_check() {
  local port="${RESULTS_PORT:-8001}"
  systemctl_cmd is-active "${service_prefix}-results.service" >/dev/null
  systemctl_cmd is-active "${service_prefix}-worker.service" >/dev/null
  curl --fail --silent --show-error --max-time 15 "http://127.0.0.1:${port}/" >/dev/null
  [[ "$(git symbolic-ref --short HEAD)" == "main" ]]
}

rollback() {
  local status=$?
  if [[ -n "$previous_commit" ]]; then
    printf 'Deployment failed; rolling back to %s\n' "$previous_commit" >&2
    git checkout --force -B main "$previous_commit" || true
    service_restart || true
    health_check || true
  fi
  exit "$status"
}

trap rollback ERR

cd "$app_dir"
mkdir -p "$backup_dir"
previous_commit="$(git rev-parse main 2>/dev/null || git rev-parse HEAD)"

# Keep runtime data and server-side analysis artifacts outside Git changes.
if [[ -f data/state.json ]]; then
  cp data/state.json "$backup_dir/state-$(date -u +%Y%m%dT%H%M%SZ).json"
fi

git fetch --prune origin main
git cat-file -e "$commit^{commit}"
git checkout --force -B main "$commit"

python3 -m venv "$venv_dir" 2>/dev/null || true
"$venv_dir/bin/python" -m pip install --upgrade pip
"$venv_dir/bin/python" -m pip install -r requirements.txt

systemctl_cmd daemon-reload
service_restart
health_check

trap - ERR

printf 'Deployed %s\n' "$commit"
