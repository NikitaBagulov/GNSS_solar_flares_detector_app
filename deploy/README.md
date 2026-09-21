# Deployment

Copy the user units to `~/.config/systemd/user/`, then enable them:

```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now gnss-results.service gnss-worker.service
loginctl enable-linger "$USER"
```

Install a GitHub Actions self-hosted runner on the server and add the labels
`self-hosted`, `linux`, and `x64`. The deploy workflow runs locally on that
machine, so no SSH secrets are required and the private address is reachable.

The deploy workflow uses the fixed server path
`/home/user/app/GNSS_solar_flares_detector_app`.

The deployment script updates tracked files only and preserves `data/`, `results/`
and server-side untracked artifacts. It restarts services only when user systemd
is available.
