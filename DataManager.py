from pathlib import Path
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime, date

class DataManager:
    def __init__(self, base_download_dir: str = "./data"):

        self.base_download_dir = Path(base_download_dir)
        self.base_download_dir.mkdir(parents=True, exist_ok=True)

        self.download_functions: Dict[str, Callable] = {}
        self.sources_config: Dict[str, Dict] = {}

    def register_download_function(self, 
        source_name: str, 
        download_func: Callable,
        config: Optional[Dict] = None
        ):

        self.download_functions[source_name] = download_func
        if config:
            self.sources_config[source_name] = config
    
    def download_by_date(
        self, 
        target_date: date,
        sources: Optional[List[str]] = None,
        force_redownload: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        results = {}

        sources_to_download = sources or list(self.download_functions.keys())
        
        for source_name in sources_to_download:
            if source_name not in self.download_functions:
                #f"Источник {source_name} не зарегистрирован"
                continue
            
            try:
                #f"Скачивание из {source_name} за {target_date}"
                download_func = self.download_functions[source_name]

                result = download_func(
                    date=target_date,
                    data_manager=self,
                    force_redownload=force_redownload,
                    **kwargs
                )
                
                results[source_name] = {
                    'status': 'success',
                    'result': result,
                    'date': target_date
                }
                
                #f"Успешно скачано из {source_name}"
                
            except Exception as e:
                #f"Ошибка при скачивании из {source_name}: {e}"
                results[source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'date': target_date
                }
        
        return results
    
    def get_download_path(
        self, 
        source_name: str, 
        target_date: date,
        filename: Optional[str] = None,
        create_dir: bool = True
    ) -> Path:

        date_dir = target_date.strftime('%Y-%m-%d')
        date_path = self.base_download_dir / date_dir / source_name
        
        if create_dir:
            date_path.mkdir(parents=True, exist_ok=True)
        
        if filename:
            return date_path / filename
        else:
            return date_path / f"{source_name}.csv"
    
    def check_file_exists(
        self, 
        source_name: str, 
        target_date: date,
        filename: Optional[str] = None
    ) -> bool:
        
        file_path = self.get_download_path(source_name, target_date, filename, create_dir=False)
        return file_path.exists()
    