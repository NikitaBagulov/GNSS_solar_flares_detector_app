from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Any, Optional, Iterable
import time


MODULE_ORDER = ["download", "preprocess", "index", "plot"]


@dataclass
class RunConfig:
    existing_data_policy: str = "skip"
    skip_modules: set[str] = field(default_factory=set)
    overwrite_modules: set[str] = field(default_factory=set)
    validate_modules: set[str] = field(default_factory=set)


def should_run_module(module_name: str, config: RunConfig) -> bool:
    return module_name not in config.skip_modules


def effective_policy(module_name: str, config: RunConfig) -> str:
    if module_name in config.overwrite_modules:
        return "overwrite"
    if module_name in config.validate_modules:
        return "validate"
    return config.existing_data_policy


def apply_data_policy(
    target_path: Path,
    policy: str,
    validator: Callable[[Path], bool],
    producer: Callable[[Path], Any],
) -> Dict[str, Any]:
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if policy == "overwrite":
        if target_path.exists():
            target_path.unlink()
        value = producer(target_path)
        return {"status": "recreated", "path": str(target_path), "value": value}

    if policy == "skip":
        if target_path.exists():
            return {"status": "skipped", "path": str(target_path)}
        value = producer(target_path)
        return {"status": "created", "path": str(target_path), "value": value}

    if policy == "validate":
        if target_path.exists() and validator(target_path):
            return {"status": "valid", "path": str(target_path)}
        if target_path.exists():
            target_path.unlink()
        value = producer(target_path)
        return {"status": "repaired", "path": str(target_path), "value": value}

    raise ValueError(f"Unknown policy: {policy}")


def run_pipeline_once(
    config: RunConfig,
    modules: Dict[str, Callable[[Dict[str, Any], str], Dict[str, Any]]],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pipeline_context = dict(context or {})
    outputs: Dict[str, Dict[str, Any]] = {}

    for module_name in MODULE_ORDER:
        if module_name not in modules:
            continue
        if not should_run_module(module_name, config):
            continue

        policy = effective_policy(module_name, config)
        result = modules[module_name](pipeline_context, policy)
        if not isinstance(result, dict):
            raise TypeError(f"Module '{module_name}' must return dict, got {type(result)!r}")

        outputs[module_name] = result
        pipeline_context[module_name] = result

    return {"context": pipeline_context, "outputs": outputs}


def service_loop(
    run_once: Callable[[], Any],
    interval_seconds: int = 3600,
    sleep_fn: Callable[[float], None] = time.sleep,
    max_cycles: Optional[int] = None,
) -> int:
    cycles = 0
    while True:
        try:
            run_once()
        except Exception:
            # в сервисном режиме продолжаем работу на следующем цикле
            pass

        cycles += 1
        if max_cycles is not None and cycles >= max_cycles:
            return cycles
        sleep_fn(interval_seconds)


def make_fixture_modules(base_dir: Path) -> Dict[str, Callable[[Dict[str, Any], str], Dict[str, Any]]]:
    def _download(context: Dict[str, Any], policy: str):
        path = base_dir / "raw" / "download.txt"
        result = apply_data_policy(path, policy, lambda p: p.read_text() == "ok", lambda p: p.write_text("ok"))
        return {"raw_path": result["path"], "status": result["status"]}

    def _preprocess(context: Dict[str, Any], policy: str):
        assert "download" in context
        path = base_dir / "preprocessed" / "prep.txt"
        result = apply_data_policy(path, policy, lambda p: p.read_text() == "prep", lambda p: p.write_text("prep"))
        return {"prep_path": result["path"], "status": result["status"]}

    def _index(context: Dict[str, Any], policy: str):
        assert "preprocess" in context
        path = base_dir / "indices" / "index.txt"
        result = apply_data_policy(path, policy, lambda p: p.read_text() == "index", lambda p: p.write_text("index"))
        return {"index_path": result["path"], "status": result["status"]}

    def _plot(context: Dict[str, Any], policy: str):
        assert "index" in context
        path = base_dir / "plots" / "plot.txt"
        result = apply_data_policy(path, policy, lambda p: p.read_text() == "plot", lambda p: p.write_text("plot"))
        return {"plot_path": result["path"], "status": result["status"]}

    return {
        "download": _download,
        "preprocess": _preprocess,
        "index": _index,
        "plot": _plot,
    }
