# Article computation stages

Run from the repository root on the Ubuntu server:

```bash
cd /home/user/app/GNSS_solar_flares_detector_app
source .venv/bin/activate
mkdir -p analysis/article_outputs
```

## 1. Inventory existing results

```bash
python3 -m analysis.event_readiness \
  --results-dir results \
  --output analysis/article_outputs/event_readiness.csv \
  | tee analysis/article_outputs/event_readiness.log
```

## 2. Select the latest five X-class events

```bash
events=$(find results/X -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
  | sort | tail -5 | tr '\n' ' ')
printf '%s\n' "$events"
```

## 3. Recompute indices from existing maps

The corrected Day/Night implementation uses a linear distance weight equal to
1 at the subsolar point and 0 at the terminator. Day values are divided by
`cos(chi)` with a numerical lower bound.

```bash
python3 -m analysis.recompute_existing_indices \
  --results-dir results \
  --events $events \
  --products roti dtec_2_10 dtec_10_20 dtec_20_60 \
  --policy overwrite \
  --backup-existing \
  --report analysis/article_outputs/recompute_last5.csv \
  2>&1 | tee analysis/article_outputs/recompute_last5.log
```

Existing index CSV files are copied to `indices_legacy_<UTC timestamp>` before
replacement.

## 4. Build dashboards

```bash
python3 -m analysis.plotting_scripts.flare_dashboard \
  --results-dir results \
  --output-dir analysis/article_outputs/dashboards_last5 \
  --products roti dtec_2_10 dtec_10_20 dtec_20_60 \
  --events $events \
  2>&1 | tee analysis/article_outputs/dashboard_last5.log
```

## 5. Solar zenith response

```bash
python3 -m analysis.plotting_scripts.flare_solar_zenith_response \
  --results-dir results \
  --output-dir analysis/article_outputs/zenith_last5 \
  --products roti dtec_2_10 dtec_10_20 dtec_20_60 \
  --events $events \
  --min-count 1 \
  --save-points \
  --verbose \
  2>&1 | tee analysis/article_outputs/zenith_last5.log
```

## 6. Solar-driver/index peak lags

```bash
python3 -m analysis.solar_driver_index_lags.solar_driver_index_lags \
  --results-dir results \
  --output-dir analysis/article_outputs/lags \
  2>&1 | tee analysis/article_outputs/lags.log
```

Before a long run, confirm the available arguments:

```bash
python3 -m analysis.solar_driver_index_lags.solar_driver_index_lags --help
```

Send back `event_readiness.csv`, `recompute_last5.csv`, the zenith output
directory, the lag output directory, and any failing log. Validate five events
before recomputing the complete catalog.
