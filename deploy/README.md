# Deployment

Copy the user units to `~/.config/systemd/user/`, then enable them:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gnss-results.service gnss-worker.service
loginctl enable-linger "$USER"
```

Create a GitHub `production` environment. Add these environment variables:

- `DEPLOY_HOST` = `10.0.6.78`
- `DEPLOY_PORT` = `23422`
- `DEPLOY_USER` = `user`

Add only these environment secrets:

- `DEPLOY_SSH_KEY`
- `DEPLOY_FINGERPRINT`

The deploy workflow uses the fixed server path
`/home/user/app/GNSS_solar_flares_detector_app`.

The deployment script updates tracked files only and preserves `data/`, `results/`
and server-side untracked artifacts. It restarts services only when user systemd
is available.
