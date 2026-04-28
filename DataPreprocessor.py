import re
from pathlib import Path
from datetime import datetime, timedelta
import h5py

import time
from simurg_core.storage.hdf_query import get_map_chunked
from simurg_core.storage.hdf_storage import get_sites_attrs
from simurg_core.storage.hdf_maps import store_maps_time_based
from flare_utils import build_flare_key, get_flare_window
from results_layout import event_results_dir, product_file_name

from dateutil import tz

class DataPreprocessor:
    DATE_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})")
    _UTC = tz.gettz('UTC')

    def __init__(self, input_root="./data", output_dir="./results", data_products=None, window_minutes: int = 15, existing_data_policy: str = "validate"):
        self.input_root = Path(input_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_minutes = window_minutes
        self.existing_data_policy = existing_data_policy

        if data_products is None:
            self.data_products = ["roti", "dtec_2_10", "dtec_10_20", "dtec_20_60"]
            # self.data_products = ["roti"]
        else:
            self.data_products = data_products

        self.maps_files = {prod: self.output_dir / f"map_{prod}.h5" for prod in self.data_products}

    def get_h5_files(self):
        return list(self.input_root.rglob("*.h5"))

    def extract_date_from_filename(self, file_path):
        match = self.DATE_PATTERN.search(file_path.name)
        if match:
            year, month, day = map(int, match.groups())
            return datetime(year, month, day).replace(tzinfo=self._UTC)
        else:
            raise ValueError(f"Не удалось извлечь дату из имени файла: {file_path.name}")

    def safe_store_maps(self, query, data, filename):
        try:
            store_maps_time_based(query, data, filename)
        except UnboundLocalError as e:
            if "lock_file" in str(e):
                print(e)
            else:
                raise e

    def get_output_dir_for_flare(self, flare_key: str, flare_class: str | None = None):
        flare_dir = event_results_dir(flare_key, flare_class=flare_class, root=self.output_dir) / "maps"
        flare_dir.mkdir(parents=True, exist_ok=True)
        return flare_dir

    def get_map_path(self, flare_dir: Path, flare_key: str, product: str, flare_class: str | None = None) -> Path:
        return flare_dir / product_file_name("map", product, flare_key, ".h5", flare_class=flare_class)

    def _is_map_file_valid(self, path: Path) -> bool:
        if not path.exists() or path.stat().st_size == 0:
            return False
        try:
            with h5py.File(path, "r") as h5:
                return bool(h5.keys())
        except Exception:
            return False

    def _select_products_to_generate(self, flare_dir: Path, flare_key: str, flare_class: str | None = None):
        selected = []
        for prod in self.data_products:
            maps_file = self.get_map_path(flare_dir, flare_key, prod, flare_class=flare_class)
            if self.existing_data_policy == "overwrite":
                maps_file.unlink(missing_ok=True)
                selected.append(prod)
                continue

            if not maps_file.exists():
                selected.append(prod)
                continue

            if self.existing_data_policy == "skip":
                continue

            if self._is_map_file_valid(maps_file):
                continue

            maps_file.unlink(missing_ok=True)
            selected.append(prod)

        return selected

    def process_file(self, file_path, tracker):
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"Could not find {file_path}")

        study_date = self.extract_date_from_filename(file_path)

        sites_description = get_sites_attrs(file_path)
        n_sites = len(sites_description)
        print(f"Processing file {file_path.name} for {n_sites} sites on {study_date.date()}")

        if n_sites == 0:
            print("No sites found in file, skipping.")
            return

        flares_for_date = []
        if tracker is not None:
            flares_for_date = tracker.get_flares_for_date(study_date.date())

        if not flares_for_date:
            print(f"No flares found for {study_date.date()}, skipping preprocessing.")
            return

        print(f"Products to process: {self.data_products}")
        print(file_path)
        for flare in flares_for_date:
            flare_key = flare.get("flare_key") or build_flare_key(
                flare["start_time"],
                flare["peak_time"],
                flare["end_time"],
                flare.get("class"),
            )
            flare_class = flare.get("class")
            flare_dir = self.get_output_dir_for_flare(flare_key, flare_class=flare_class)

            products_to_generate = self._select_products_to_generate(flare_dir, flare_key, flare_class=flare_class)
            if not products_to_generate:
                print(f"Data for flare {flare_key} already processed for policy={self.existing_data_policy}, skipping.")
                if tracker is not None:
                    tracker.set_files_for_flare_section(
                        flare_key,
                        "maps",
                        {
                            prod: self.get_map_path(flare_dir, flare_key, prod, flare_class=flare_class)
                            for prod in self.data_products
                            if self.get_map_path(flare_dir, flare_key, prod, flare_class=flare_class).exists()
                        },
                    )
                continue

            start_interval, end_interval = get_flare_window(
                flare["start_time"],
                flare["end_time"],
                window_minutes=self.window_minutes,
            )
            times = []
            current = start_interval
            while current <= end_interval:
                times.append(current)
                current = current + timedelta(seconds=30)
            print(times[0], type(times[0]), times[0].timestamp())
            
            print(times[-1], type(times[-1]), times[-1].timestamp())
            if not times:
                print(f"Empty time window for flare {flare_key}, skipping.")
                continue

            print(f"Flare {flare_key}: {times[0]} ... {times[-1]}")
            start_time = time.time()
            generator = get_map_chunked(
                sites_description,
                times,
                file_path=file_path,
                product_types=products_to_generate,
                roti_type='mapping_function',
                chunk=120
            )

            maps_files = []
            for prod in products_to_generate:
                maps_file = self.get_map_path(flare_dir, flare_key, prod, flare_class=flare_class).resolve()
                maps_file.parent.mkdir(parents=True, exist_ok=True)
                maps_files.append(maps_file)

            iprod = 0
            for chunk_idx, data in enumerate(generator, 1):
                took = time.time() - start_time
                print(f"Chunk {chunk_idx}. Time {took:.2f} seconds.")

                if not data:
                    print(f"Chunk {chunk_idx} is empty. Skipping.\n")
                    continue

                # ✅ текущий продукт определяется номером чанка
                out_file = maps_files[iprod % len(maps_files)]
                current_prod = products_to_generate[iprod % len(products_to_generate)]

                try:
                    store_maps_time_based({'sites': 'sites'}, data, str(out_file), lock=False)
                except Exception as e:
                    print(f"Failed to save {out_file.name} (prod={current_prod}): {e}")
                else:
                    print(f"Saved {out_file.name} successfully (prod={current_prod})")

                iprod += 1
            if tracker is not None:
                tracker.set_files_for_flare_section(
                    flare_key,
                    "maps",
                    {
                        prod: self.get_map_path(flare_dir, flare_key, prod, flare_class=flare_class)
                        for prod in self.data_products
                        if self.get_map_path(flare_dir, flare_key, prod, flare_class=flare_class).exists()
                    }
                )
            

       


    def process_all(self, tracker=None):
        h5_files = self.get_h5_files()
        print(f"Found {len(h5_files)} HDF5 files to process.\n")

        for file_idx, file_path in enumerate(h5_files, 1):
            print(f"[{file_idx}/{len(h5_files)}] {file_path.name}")
            self.process_file(file_path, tracker)

        print("All files processed successfully!")
