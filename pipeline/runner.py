from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List

from DataManager import DataManager
from DataPreprocessor import DataPreprocessor
from FlareTracker import FlareTracker
from IndexCalculator import IndexCalculator
from PlotDataLoader import PlotDataLoader
from Plotter import Plotter, CombinedPlotter
from download_functions.euv import download_soho_sem
from download_functions.simurg_hdf import download_simurg_hdf
from download_functions.xray import download_goes_xray


@dataclass(frozen=True)
class PipelineConfig:
    start_date: date
    end_date: date
    min_flare_class: str
    state_json_path: Path
    data_download_path: Path


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
    data_manager = DataManager(base_download_dir=str(config.data_download_path))
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


def run_discovery_and_download(config: PipelineConfig) -> DiscoveryDownloadResult:
    tracker = _load_tracker(config)

    if hasattr(tracker, "download_missed_data"):
        tracker.download_missed_data()

    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())
    return DiscoveryDownloadResult(
        state_json_path=tracker.state_file,
        all_flares_csv_path=tracker.all_flares_file,
        flare_keys=flare_keys,
    )


def run_preprocessing(config: PipelineConfig) -> PreprocessingResult:
    tracker = _load_tracker(config)
    preprocessor = DataPreprocessor(input_root=str(config.data_download_path))
    preprocessor.process_all(tracker)

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
    calculator = IndexCalculator()

    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())
    for flare_key in flare_keys:
        calculator.process_single_flare(flare_key, tracker=tracker)

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
    flare_keys = list(tracker.state.get("files_by_flare", {}).keys())

    loader = PlotDataLoader(tracker.all_flares_file, tracker.state_file)
    plotted_flare_keys: List[str] = []

    for flare_key in flare_keys:
        plot_data = loader.load_flare(flare_key)
        if not plot_data:
            continue
        plotter = Plotter(plot_data, products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"])
        plotter.plot_all()    
        combined_plotter = CombinedPlotter(
            plot_data,
            products_to_plot=["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"],
        )
        combined_plotter.plot_all()
        plotted_flare_keys.append(flare_key)

    return PlottingResult(plotted_flare_keys=plotted_flare_keys)
