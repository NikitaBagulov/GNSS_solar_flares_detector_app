import re
from pathlib import Path
from datetime import datetime, timedelta
from dateutil import tz
import time
from simurg_core.storage.hdf_query import get_map_chunked
from simurg_core.storage.hdf_storage import get_sites_attrs
from simurg_core.storage.hdf_maps import store_maps_time_based

class DataPreprocessor:
    DATE_PATTERN = re.compile(r"(\d{4})(\d{2})(\d{2})")
    _UTC = tz.gettz('UTC')

    def __init__(self, input_root="./data", output_dir="./preprocessed_maps", data_products=None):
        self.input_root = Path(input_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

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

    def get_output_dir_for_date(self, study_date):
        date_str = study_date.strftime("%Y-%m-%d")
        date_dir = self.output_dir / date_str
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir

    def is_date_already_processed(self, study_date):
        date_output_dir = self.get_output_dir_for_date(study_date)

        for prod in self.data_products:
            maps_file = date_output_dir / f"map_{prod}.h5"
            if not maps_file.exists():
                return False
        return True

    def process_file(self, file_path, tracker):
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"Could not find {file_path}")

        study_date = self.extract_date_from_filename(file_path)
        date_output_dir = self.get_output_dir_for_date(study_date)

        # Регистрируем существующие файлы
        existing_files = {}
        for prod in self.data_products:
            maps_file = date_output_dir / f"map_{prod}.h5"
            if maps_file.exists():
                existing_files[prod] = maps_file

        if existing_files and tracker is not None:
            tracker.register_files_for_date(
                study_date.date(),
                {"maps": existing_files}
            )

        if self.is_date_already_processed(study_date):
            print(f"Data for {study_date.date()} already processed, skipping.")
            return
        
        date_output_dir = self.get_output_dir_for_date(study_date)

        sites_description = get_sites_attrs(file_path)
        n_sites = len(sites_description)
        print(f"Processing file {file_path.name} for {n_sites} sites on {study_date.date()}")

        if n_sites == 0:
            print("No sites found in file, skipping.")
            return

        times = [study_date + timedelta(seconds=30*i) for i in range(2880)]
        print(f"First 5 times: {times[:5]} ... Last 5 times: {times[-5:]}")

        print(f"Products to process: {self.data_products}")
        print(file_path)
        start_time = time.time()
        generator = get_map_chunked(
            sites_description,
            times,
            file_path=file_path,
            product_types=self.data_products,
            roti_type='simple',
            chunk=120
        )
        for chunk_idx, data in enumerate(generator, 1):
            
            if not data:
                print(f"Chunk {chunk_idx} is empty: {data}. Time {time.time() - start_time:.2f} seconds\n") 
                continue

            print(f"Chunk {chunk_idx}. Time {time.time() - start_time:.2f} seconds.")
            for prod in self.data_products:
                maps_file = (date_output_dir / f"map_{prod}.h5").resolve()
                maps_file.parent.mkdir(parents=True, exist_ok=True)
                try:
                    store_maps_time_based({'sites': 'sites'}, data, str(maps_file), lock=False)
                except Exception as e:
                    print(f"Failed to save {maps_file.name}: {e}")
                else:
                    print(f"Saved {maps_file.name} successfully")
        if tracker is not None:
            tracker.register_files_for_date(
            study_date.date(),
            {
                "maps": {
                    prod: date_output_dir / f"map_{prod}.h5"
                    for prod in self.data_products
                    }
                }
            )

       


    def process_all(self, tracker=None):
        h5_files = self.get_h5_files()
        print(f"Found {len(h5_files)} HDF5 files to process.\n")

        for file_idx, file_path in enumerate(h5_files, 1):
            print(f"[{file_idx}/{len(h5_files)}] {file_path.name}")
            self.process_file(file_path, tracker)

        print("All files processed successfully!")
