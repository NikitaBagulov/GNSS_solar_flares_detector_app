from pipeline_testkit import service_loop


def test_service_loop_runs_exact_cycles_with_mocked_sleep():
    calls = {"run": 0, "sleep": []}

    def _run_once():
        calls["run"] += 1

    def _sleep(seconds):
        calls["sleep"].append(seconds)

    cycles = service_loop(_run_once, interval_seconds=5, sleep_fn=_sleep, max_cycles=3)

    assert cycles == 3
    assert calls["run"] == 3
    assert calls["sleep"] == [5, 5]


def test_service_loop_survives_exceptions():
    calls = {"run": 0, "sleep": 0}

    def _run_once():
        calls["run"] += 1
        if calls["run"] == 1:
            raise RuntimeError("transient")

    def _sleep(_):
        calls["sleep"] += 1

    cycles = service_loop(_run_once, interval_seconds=1, sleep_fn=_sleep, max_cycles=2)

    assert cycles == 2
    assert calls["run"] == 2
    assert calls["sleep"] == 1
