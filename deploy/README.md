# Deployment

The deploy runner is a system service, so production services must also be
system units. Install them once on the server as root:

```bash
sudo install -m 0644 deploy/systemd/gnss-results.system.service /etc/systemd/system/gnss-results.service
sudo install -m 0644 deploy/systemd/gnss-worker.system.service /etc/systemd/system/gnss-worker.service
sudo systemctl daemon-reload
sudo systemctl enable --now gnss-results.service gnss-worker.service
```

The deployment runner must be able to reload and restart these units. If the
runner is not installed as root, add a narrowly scoped sudoers rule once:

```text
user ALL=(root) NOPASSWD: /usr/bin/systemctl daemon-reload, /usr/bin/systemctl restart gnss-results.service, /usr/bin/systemctl restart gnss-worker.service, /usr/bin/systemctl is-active gnss-results.service, /usr/bin/systemctl is-active gnss-worker.service
```

Save it as `/etc/sudoers.d/gnss-deploy` and validate it with:

```bash
sudo visudo -cf /etc/sudoers.d/gnss-deploy
```

The service units run as `user` and use the fixed production path. The runner
must be installed as a systemd service with labels `self-hosted`, `linux`, and
`x64`; systemd will restart it if it exits.

Install a GitHub Actions self-hosted runner on the server and add the labels
`self-hosted`, `linux`, and `x64`. The deploy workflow runs locally on that
machine, so no SSH secrets are required and the private address is reachable.

The deploy workflow uses the fixed server path
`/home/user/app/GNSS_solar_flares_detector_app`.

The deployment script updates tracked files only and preserves `data/`, `results/`
and server-side untracked artifacts. It records the previous `main` commit and
automatically restores it, restarts both services, and reruns the health check
if deployment fails. It never runs `git clean`.
