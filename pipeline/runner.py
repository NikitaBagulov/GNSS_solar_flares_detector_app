from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
import shutil
from typing import Dict, List, Optional

from DataManager import DataManager
from DataPreprocessor import DataPreprocessor
from FlareTracker import FlareTracker
from IndexCalculator import IndexCalculator
from PlotDataLoader import PlotDataLoader
from Plotter import Plotter, CombinedPlotter
from download_functions.euv import download_soho_sem
from download_functions.simurg_hdf import download_simurg_hdf
from download_functions.xray import download_goes_xray
from pipeline.run_config import RunConfig
from results_layout import event_results_dir, publish_file, source_file_name


@dataclass(frozen=True)
class PipelineConfig:
    start_date: date
    end_date: date
    min_flare_class: str
    state_json_path: Path
    data_download_path: Path
    run_config: RunConfig


@dataclass(frozen=True)
class DiscoveryDownloadResult:
    state_json_path: Path
    all_flares_csv_path: Path
    flare_keys: List[str]


@dataclass(frozen=True)
class PreprocessingResult:
    flare_keys: List[str]
    maps_by_flare: Dict[str, Dict[str, str]]


@dataclass(frozen=True)
class IndexCalculationResult:
    flare_keys: List[str]
    indices_by_flare: Dict[str, Dict[str, str]]


@dataclass(frozen=True)
class PlottingResult:
    plotted_flare_keys: List[str]


def _build_data_manager(config: PipelineConfig) -> DataManager:
    data_manager = DataManager(
        base_download_dir=str(config.data_download_path),
        existing_data_policy=config.run_config.policy_for("download"),
    )
    data_manager.register_download_function("soho_sem", download_func=download_soho_sem)
    data_manager.register_download_function("goes_xray", download_func=download_goes_xray)
    data_manager.register_download_function(
        "simurg_hdf",
        download_func=download_simurg_hdf,
        default_extension=".h5",
    )
    return data_manager


def _load_tracker(config: PipelineConfig) -> FlareTracker:
    return FlareTracker(
        data_manager=_build_data_manager(config),
        start_date=config.start_date,
        end_date=config.end_date,
        min_flare_class=config.min_flare_class,
        state_file_path=str(config.state_json_path),
    )


def _flare_classes_by_key(tracker: FlareTracker) -> Dict[str, str]:
    flares = tracker._load_all_flares()
    if flares.empty or "flare_key" not in flares.columns or "class" not in flares.columns:
        return {}
    return {
        str(row["flare_key"]): str(row["class"])
        for _, row in flares.iterrows()
        if row.get("flare_key") and row.get("class")
    }


def _event_dir(flare_key: str, flare_classes: Optional[Dict[str, str]] = None) -> Path:
    flare_class = (flare_classes or {}).get(flare_key)
    return event_results_dir(flare_key, flare_class=flare_class)


def _publish_source_files(tracker: FlareTracker, overwrite: bool = False) -> None:
    updated = False
    flare_classes = _flare_classes_by_key(tracker)
    for flare_key, files in list(tracker.state.get("files_by_flare", {}).items()):
        for source in ("goes_xray", "soho_sem"):
            source_path = files.get(source)
            if not source_path or not Path(source_path).exists():
                continue

            original = Path(source_path)
            flare_class = flare_classes.get(flare_key)
            target = (
                _event_dir(flare_key, flare_classes)
                / source
                / source_file_name(source, flare_key, original.suffix or ".csv", flare_class=flare_class)
            )
            published = publish_file(original, target, overwrite=overwrite)
            if published.exists() and str(published) != str(source_path):
                tracker.state["files_by_flare"][flare_key][source] = str(published)
                updated = True

    if updated:
        tracker._save_state(message="Исходные GOES/SOHO опубликованы в results")


def run_discovery_and_download(config: PipelineConfig) -> DiscoveryDownloadResult:
    tracker = _load_tracker(config)

    if hasattr(tracker, "download_missed_data"):
        tracker.download_missed_data()
    _publish_source_files(
        tracker,
        overwrite=config.run_config.policy_for("download") == "overwrite",
    )

    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())
    return DiscoveryDownloadResult(
        state_json_path=tracker.state_file,
        all_flares_csv_path=tracker.all_flares_file,
        flare_keys=flare_keys,
    )


def run_preprocessing(config: PipelineConfig) -> PreprocessingResult:
    tracker = _load_tracker(config)
    _publish_source_files(
        tracker,
        overwrite=config.run_config.policy_for("download") == "overwrite",
    )
    preprocessor = DataPreprocessor(
        input_root=str(config.data_download_path),
        existing_data_policy=config.run_config.policy_for("preprocess"),
    )
    preprocessor.process_all(tracker)
    tracker.sync_state_with_files()

    maps_by_flare: Dict[str, Dict[str, str]] = {}
    for flare_key, files in tracker.state.get("files_by_flare", {}).items():
        maps = files.get("maps") or {}
        if maps:
            maps_by_flare[flare_key] = dict(maps)

    return PreprocessingResult(
        flare_keys=sorted(maps_by_flare.keys()),
        maps_by_flare=maps_by_flare,
    )


def run_index_calculation(config: PipelineConfig) -> IndexCalculationResult:
    tracker = _load_tracker(config)
    _publish_source_files(
        tracker,
        overwrite=config.run_config.policy_for("download") == "overwrite",
    )
    calculator = IndexCalculator(
        existing_data_policy=config.run_config.policy_for("index"),
    )

    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())
    for flare_key in flare_keys:
        calculator.process_single_flare(flare_key, tracker=tracker)

    tracker.sync_state_with_files()

    indices_by_flare: Dict[str, Dict[str, str]] = {}
    for flare_key, files in tracker.state.get("files_by_flare", {}).items():
        indices = files.get("indices") or {}
        if indices:
            indices_by_flare[flare_key] = dict(indices)

    return IndexCalculationResult(
        flare_keys=sorted(indices_by_flare.keys()),
        indices_by_flare=indices_by_flare,
    )


def run_plotting(config: PipelineConfig) -> PlottingResult:
    tracker = _load_tracker(config)
    _publish_source_files(
        tracker,
        overwrite=config.run_config.policy_for("download") == "overwrite",
    )
    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())

    loader = PlotDataLoader(tracker.all_flares_file, tracker.state_file)
    plotted_flare_keys: List[str] = []
    plot_policy = config.run_config.policy_for("plot")
    flare_classes = _flare_classes_by_key(tracker)

    for flare_key in flare_keys:
        flare_plot_dir = _event_dir(flare_key, flare_classes) / "graphs"
        if plot_policy == "skip" and flare_plot_dir.exists():
            plotted_flare_keys.append(flare_key)
            tracker.set_files_for_flare_section(flare_key, "plots", {"root": flare_plot_dir})
            continue
        if plot_policy == "overwrite" and flare_plot_dir.exists():
            shutil.rmtree(flare_plot_dir)
        if plot_policy == "validate" and flare_plot_dir.exists():
            has_valid_plots = any(
                path.is_file() and path.suffix.lower() == ".png" and path.stat().st_size > 0
                for path in flare_plot_dir.rglob("*.png")
            )
            if has_valid_plots:
                plotted_flare_keys.append(flare_key)
                tracker.set_files_for_flare_section(flare_key, "plots", {"root": flare_plot_dir})
                continue

        plot_data = loader.load_flare(flare_key)
        if not plot_data:
            continue
        plotter = Plotter(
            plot_data,
            products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"],
            output_dir=flare_plot_dir,
        )
        plotter.plot_all()    
        
        combined_plotter = CombinedPlotter(
            plot_data,
            products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"],
            output_dir=flare_plot_dir,
        )
        combined_plotter.plot_all()
        tracker.set_files_for_flare_section(flare_key, "plots", {"root": flare_plot_dir})
        plotted_flare_keys.append(flare_key)

    return PlottingResult(plotted_flare_keys=plotted_flare_keys)
