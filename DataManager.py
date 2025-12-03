from pathlib import Path
from typing import Dict, List, Callable, Optional, Any, Set
from datetime import datetime, date
import atexit
import signal
import sys
import json
import pandas as pd

class DataManager:
    def __init__(self, base_download_dir: str = "./data"):
        self.base_download_dir = Path(base_download_dir)
        self.base_download_dir.mkdir(parents=True, exist_ok=True)

        self.download_functions: Dict[str, Callable] = {}
        self.sources_config: Dict[str, Dict] = {}

        self._active_downloads: Dict[str, List[Path]] = {}
        self._transaction_file = self.base_download_dir / ".download_transactions.json"
        
        self._recover_interrupted_downloads()

        self._register_cleanup_handlers()
    
    def _register_cleanup_handlers(self):
        atexit.register(self._cleanup_all_downloads)

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\n⚠️ Получен сигнал прерывания {signum}, очищаем временные файлы...")
        self._cleanup_all_downloads()
        sys.exit(1)
    
    def _save_transaction_state(self):
        try:
            state = {}
            for source, temp_files in self._active_downloads.items():
                state[source] = [str(f) for f in temp_files]
            
            with open(self._transaction_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения состояния транзакций: {e}")
    
    def _recover_interrupted_downloads(self):
        if not self._transaction_file.exists():
            return
        
        try:
            with open(self._transaction_file, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            files_removed = 0
            for source, temp_files in state.items():
                for temp_file_str in temp_files:
                    temp_file = Path(temp_file_str)
                    if temp_file.exists():
                        try:
                            temp_file.unlink()
                            files_removed += 1
                            print(f"🧹 Удален недокаченный файл: {temp_file}")
                        except Exception as e:
                            print(f"⚠️ Не удалось удалить {temp_file}: {e}")

            self._transaction_file.unlink(missing_ok=True)
            
            if files_removed > 0:
                print(f"✅ Восстановление: удалено {files_removed} недокаченных файлов")
                
        except Exception as e:
            print(f"⚠️ Ошибка восстановления после прерывания: {e}")
            self._transaction_file.unlink(missing_ok=True)
    
    def _cleanup_all_downloads(self):
        files_removed = 0
        
        for source, temp_files in list(self._active_downloads.items()):
            for temp_file in temp_files[:]:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        files_removed += 1
                        print(f"🧹 Удален временный файл: {temp_file}")
                    except Exception as e:
                        print(f"⚠️ Не удалось удалить {temp_file}: {e}")
                self._active_downloads[source].remove(temp_file)

        self._transaction_file.unlink(missing_ok=True)
        
        if files_removed > 0:
            print(f"🧹 Очистка: удалено {files_removed} временных файлов")
    
    def _start_download_transaction(self, source_name: str, temp_file: Path):
        if source_name not in self._active_downloads:
            self._active_downloads[source_name] = []
        
        self._active_downloads[source_name].append(temp_file)
        self._save_transaction_state()
    
    def _complete_download_transaction(self, source_name: str, temp_file: Path):
        if source_name in self._active_downloads and temp_file in self._active_downloads[source_name]:
            self._active_downloads[source_name].remove(temp_file)
            if not self._active_downloads[source_name]:
                del self._active_downloads[source_name]
            self._save_transaction_state()
    
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
                continue
            
            try:
                download_func = self.download_functions[source_name]

                filename = f"{source_name}_{target_date.strftime('%Y%m%d')}.csv"
                final_path = self.get_download_path(source_name, target_date, filename)

                if not force_redownload and final_path.exists():
                    try:
                        if self._is_file_valid(final_path, source_name):
                            results[source_name] = {
                                'status': 'skipped',
                                'result': str(final_path),
                                'date': target_date,
                                'message': 'Файл уже существует'
                            }
                            continue
                        else:
                            print(f"⚠️ Файл {final_path} поврежден, перезагружаем...")
                    except:
                        print(f"⚠️ Не удалось прочитать файл {final_path}, перезагружаем...")

                temp_path = final_path.with_suffix('.tmp')

                self._start_download_transaction(source_name, temp_path)

                print(f"📥 Скачивание {source_name} за {target_date}...")
                result = download_func(
                    date=target_date,
                    data_manager=self,
                    force_redownload=force_redownload,
                    temp_path=temp_path,
                    **kwargs
                )

                if result and temp_path.exists():
                    temp_path.rename(final_path)

                    self._complete_download_transaction(source_name, temp_path)
                    
                    results[source_name] = {
                        'status': 'success',
                        'result': str(final_path),
                        'date': target_date,
                        'size': final_path.stat().st_size if final_path.exists() else 0
                    }
                    print(f"✅ Успешно скачано: {final_path}")
                else:
                    if temp_path.exists():
                        temp_path.unlink()

                    self._complete_download_transaction(source_name, temp_path)
                    
                    results[source_name] = {
                        'status': 'error',
                        'error': 'Не удалось скачать файл',
                        'date': target_date
                    }
                    print(f"❌ Ошибка скачивания {source_name}")
                    
            except Exception as e:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                    self._complete_download_transaction(source_name, temp_path)
                
                results[source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'date': target_date
                }
                print(f"❌ Ошибка при скачивании из {source_name}: {e}")
        
        return results
    
    def _is_file_valid(self, file_path: Path, source_name: str) -> bool:
        if not file_path.exists():
            return False
        
        try:

            if file_path.stat().st_size == 0:
                return False

            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path, nrows=1)
                if df.empty:
                    return False
            
            return True
            
        except:
            return False
    
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
        return file_path.exists() and self._is_file_valid(file_path, source_name)