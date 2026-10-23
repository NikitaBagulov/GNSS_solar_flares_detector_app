"""Explicit sequential recovery of the three requested incomplete M events."""
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    if Path.cwd().resolve() != root:
        raise SystemExit(f'Run from {root}')
    state = subprocess.run(['systemctl', 'show', 'gnss-worker.service',
                            '--property=ActiveState', '--value'],
                           text=True, capture_output=True)
    if state.returncode or state.stdout.strip() not in {'inactive', 'failed'}:
        raise SystemExit('Stop gnss-worker.service before targeted processing.')
    import pandas as pd
    import main as application
    import pipeline.runner as runner
    from download_lock import download_directory_lock

    sys.argv = ['retry', '--start_date', '2008-01-01', '--min_flare_class', 'C1.0',
                '--existing-data-policy', 'validate']
    config = application.build_config(application.parse_args())
    catalog = pd.read_csv('data/all_flares.csv')
    dates = pd.to_datetime(catalog['start_time'], utc=True).dt.strftime('%Y-%m-%d')
    targets = [('2024-03-23', 'M2.5'), ('2024-05-05', 'M2.3'), ('2024-10-28', 'M1.3')]
    failed = []
    # Explicit exception applies only to this process and the requested events.
    runner.should_plot_flare = lambda *args, **kwargs: True
    with download_directory_lock('data'):
        for day, flare_class in targets:
            matches = catalog[(dates == day) & (catalog['class'] == flare_class)]
            if len(matches) != 1:
                raise RuntimeError(f'{day} {flare_class}: expected one catalog row, got {len(matches)}')
            key = matches.iloc[0]['flare_key']
            try:
                print('TARGET', day, flare_class, key, flush=True)
                if not runner.run_download_for_flare(config, key):
                    raise RuntimeError('download returned False')
                runner.run_preprocessing_for_flare(config, key)
                runner.run_index_calculation_for_flare(config, key)
                result = runner.run_plotting_for_flare(config, key)
                if key not in result.plotted_flare_keys:
                    raise RuntimeError('plotting produced no completed event')
                print('SUCCESS', key, flush=True)
            except Exception:
                import traceback
                traceback.print_exc()
                failed.append(key)
    print('TARGETED_RUN_FINISHED', 'failed=', failed, flush=True)
    return bool(failed)


if __name__ == '__main__':
    raise SystemExit(main())
