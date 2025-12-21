## RegionIQ Backend: Production (systemd) Setup

This repo assumes the **working directory is the backend repo root** so relative paths like `data/lake/warehouse.duckdb` resolve correctly.

### Required environment variables (prod)

- **`SUPABASE_URI`**: Postgres connection string used for:
  - export/sync stage (`scripts/export/Broad_export.py`)
  - email recipient resolution (`scripts/notify/weekly_email.py`) via `public.notification_recipients`
- **`RESEND_API_KEY`**: API key used by `scripts/notify/weekly_email.py` to send HTML emails via Resend
- **`PIPELINE_MODE`**: typically `prod` (optional; defaults to `prod` if unset)

### Email recipients

Email recipients are sourced from Supabase Postgres table **`public.notification_recipients`** (not directly from `auth.users`).

Create and maintain this table using:
- `scripts/supabase/notification_recipients.sql` (run once in Supabase SQL Editor)

### systemd unit (example)

Create `/etc/regioniq/backend.env` (owned by root, mode 600):

```bash
SUPABASE_URI=postgres://...
RESEND_API_KEY=re_...
PIPELINE_MODE=prod
```

Example `/etc/systemd/system/regioniq-pipeline.service`:

```ini
[Unit]
Description=RegionIQ Forecast Pipeline
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
User=regioniq
Group=regioniq

WorkingDirectory=/home/regioniq/apps/regioniq-uk-backend
EnvironmentFile=/etc/regioniq/backend.env

ExecStart=/home/regioniq/apps/regioniq-uk-backend/venv/bin/python3 /home/regioniq/apps/regioniq-uk-backend/run_full_forecast_pipeline.py

StandardOutput=journal
StandardError=journal
```

Example timer `/etc/systemd/system/regioniq-pipeline.timer`:

```ini
[Unit]
Description=Run RegionIQ pipeline daily

[Timer]
OnCalendar=*-*-* 06:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
```

Enable and test:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now regioniq-pipeline.timer
sudo systemctl start regioniq-pipeline.service
sudo journalctl -u regioniq-pipeline.service -n 200 --no-pager
```


