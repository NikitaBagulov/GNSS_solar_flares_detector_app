from pathlib import Path
from typing import Dict, List, Callable, Optional, Any, Set
from datetime import datetime, date
import atexit
import signal
import sys
import json
import pandas as pd
import os


class DataManager:
    def __init__(self, base_download_dir: str = "./data", existing_data_policy: str = "validate"):
        self.base_download_dir = Path(base_download_dir)
        self.base_download_dir.mkdir(parents=True, exist_ok=True)

        self.download_functions: Dict[str, Callable] = {}

        self.sources_config: Dict[str, Dict] = {}

        self.source_extensions: Dict[str, str] = {}

        self._active_downloads: Dict[str, List[Path]] = {}
        self._transaction_file = self.base_download_dir / ".download_transactions.json"

        self._cleanup_orphaned_temp_files()
        self._recover_interrupted_downloads()

        self._register_cleanup_handlers()
        self.existing_data_policy = existing_data_policy
        
    
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
    
    def _cleanup_orphaned_temp_files(self):
        tmp_files = list(self.base_download_dir.rglob("*.tmp"))
        
        files_removed = 0
        for tmp_file in tmp_files:
            try:
                file_age = datetime.now().timestamp() - tmp_file.stat().st_mtime
                if file_age > 3600:
                    tmp_file.unlink()
                    files_removed += 1
            except Exception as e:
                print(f"   ⚠️ Не удалось удалить {tmp_file}: {e}")
        
        if files_removed > 0:
            print(f"✅ Удалено {files_removed} старых временных файлов")
    
    def _save_transaction_state(self):
        try:
            state = {}
            for source, temp_files in self._active_downloads.items():
                state[source] = [str(f) for f in temp_files]
            
            with open(self._transaction_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения состояния транзакций: {e}")
    
    def _recover_interrupted_downloads(self):
        if not self._transaction_file.exists():
            return
        
        try:
            print("🔄 Восстановление после прерванных загрузок...")
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
                        except Exception as e:
                            print(f"   ⚠️ Не удалось удалить {temp_file}: {e}")

            self._transaction_file.unlink(missing_ok=True)
            
            if files_removed > 0:
                print(f"✅ Восстановление завершено: удалено {files_removed} недокаченных файлов")
                
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
                    except Exception as e:
                        print(f"   ⚠️ Не удалось удалить {temp_file}: {e}")
                self._active_downloads[source].remove(temp_file)

        self._transaction_file.unlink(missing_ok=True)
        
        if files_removed > 0:
            print(f"✅ Очистка завершена: удалено {files_removed} временных файлов")
    
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
        config: Optional[Dict] = None,
        default_extension: str = '.csv'
    ):
        self.download_functions[source_name] = download_func
        self.source_extensions[source_name] = default_extension
        
        if config:
            self.sources_config[source_name] = config
        
    
    def download_by_date(
        self, 
        target_date: date,
        sources: Optional[List[str]] = None,
        force_redownload: bool = False,
        tracker=None,
        **kwargs
    ) -> Dict[str, Any]:
        print(f"\n📥 ЗАГРУЗКА ДАННЫХ ЗА {target_date}")
        results = {}

        sources_to_download = sources or list(self.download_functions.keys())
        for source_name in sources_to_download:
            if source_name not in self.download_functions:
                print(f"⚠️ Источник '{source_name}' не зарегистрирован, пропускаем")
                continue
            
            try:
                download_func = self.download_functions[source_name]

                extension = self.source_extensions.get(source_name, '.csv')

                filename = f"{source_name}_{target_date.strftime('%Y%m%d')}{extension}"
                final_path = self.get_download_path(source_name, target_date, filename, create_dir=False)
                policy = "overwrite" if force_redownload else self.existing_data_policy

                if (
                    policy != "overwrite"
                    and tracker is not None
                    and hasattr(tracker, "_is_source_consumed")
                    and tracker._is_source_consumed(target_date, source_name)
                    and not final_path.exists()
                ):
                    results[source_name] = {
                        'status': 'skipped',
                        'result': str(final_path),
                        'date': target_date,
                        'message': 'Source was already consumed and removed after preprocessing',
                        'size': 0,
                    }
                    print(f"   ⏭️ {source_name}: уже переработан и удален, повторное скачивание не требуется")
                    continue

                if final_path.exists():
                    if policy == "skip":
                        if tracker is not None:
                            tracker.register_files_for_date(target_date, {source_name: str(final_path)})
                        results[source_name] = {
                            'status': 'skipped',
                            'result': str(final_path),
                            'date': target_date,
                            'message': 'Файл уже существует (skip policy)',
                            'size': final_path.stat().st_size
                        }
                        print(f"   ⏭️ {source_name}: файл уже существует (skip)")
                        continue

                    if policy == "overwrite":
                        final_path.unlink(missing_ok=True)
                        print(f"   ♻️ {source_name}: существующий файл удален (overwrite)")

                    if policy == "validate":
                        try:
                            if self._is_file_valid(final_path, source_name):
                                if tracker is not None:
                                    tracker.register_files_for_date(target_date, {source_name: str(final_path)})
                                results[source_name] = {
                                    'status': 'skipped',
                                    'result': str(final_path),
                                    'date': target_date,
                                    'message': 'Файл уже существует и валиден',
                                    'size': final_path.stat().st_size
                                }
                                print(f"   ⏭️ {source_name}: файл валиден (validate)")
                                continue
                            print(f"   ⚠️ {source_name}: файл поврежден, перезагружаем (validate)")
                            final_path.unlink(missing_ok=True)
                        except Exception as e:
                            print(f"   ⚠️ {source_name}: не удалось проверить файл ({e}), перезагружаем")
                            final_path.unlink(missing_ok=True)

                temp_path = final_path.with_suffix(extension + '.tmp')
                temp_path.parent.mkdir(parents=True, exist_ok=True)

                self._start_download_transaction(source_name, temp_path)

                print(f"   📥 {source_name}: скачивание...")
                result = download_func(
                    date=target_date,
                    data_manager=self,
                    force_redownload=force_redownload,
                    temp_path=temp_path,
                    **kwargs
                )

                if result and isinstance(result, Path) and temp_path.exists():
                    if temp_path.stat().st_size == 0:
                        raise Exception("Временный файл пустой")

                    temp_path.rename(final_path)

                    self._complete_download_transaction(source_name, temp_path)
                    if tracker is not None:
                        tracker.register_files_for_date(target_date, {source_name: str(final_path)})

                    results[source_name] = {
                        'status': 'success',
                        'result': str(final_path),
                        'date': target_date,
                        'size': final_path.stat().st_size
                    }
                    print(f"   ✅ {source_name}: успешно скачан ({final_path.stat().st_size / 1024:.1f} KB)")
                else:
                    if temp_path.exists():
                        temp_path.unlink()

                    self._complete_download_transaction(source_name, temp_path)

                    results[source_name] = {
                        'status': 'error',
                        'error': 'Не удалось скачать файл',
                        'date': target_date
                    }
                    print(f"   ❌ {source_name}: ошибка скачивания")
                    
            except Exception as e:
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                    self._complete_download_transaction(source_name, temp_path)
                
                results[source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'date': target_date
                }
                print(f"   ❌ {source_name}: ошибка - {e}")
        
        return results
    
    def _is_file_valid(self, file_path: Path, source_name: str) -> bool:
        if not file_path.exists():
            return False
        
        try:
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False

            extension = file_path.suffix.lower()
            
            if extension == '.csv':
                df = pd.read_csv(file_path, nrows=1)
                return not df.empty
                
            elif extension in ['.h5', '.hdf5', '.hdf']:
                return file_size > 100
                
            elif extension in ['.nc', '.cdf']:
                return file_size > 100
                
            else:
                return file_size > 0
                
        except Exception as e:
            print(f"⚠️ Ошибка проверки файла {file_path}: {e}")
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
            extension = self.source_extensions.get(source_name, '.csv')
            return date_path / f"{source_name}{extension}"
    
    def check_file_exists(
        self, 
        source_name: str, 
        target_date: date,
        filename: Optional[str] = None
    ) -> bool:
        file_path = self.get_download_path(source_name, target_date, filename, create_dir=False)
        return file_path.exists() and self._is_file_valid(file_path, source_name)
    
    def list_downloaded_dates(self, source_name: str) -> List[date]:
        dates = []
        source_dir = self.base_download_dir / source_name
        
        if not source_dir.exists():
            return dates

        for item in source_dir.iterdir():
            if item.is_dir():
                try:
                    date_obj = datetime.strptime(item.name, '%Y-%m-%d').date()
                    dates.append(date_obj)
                except ValueError:
                    continue
        
        return sorted(dates)
    
    def get_available_sources(self) -> List[str]:
        return list(self.download_functions.keys())
    
    def get_source_info(self, source_name: str) -> Dict:
        if source_name not in self.download_functions:
            raise ValueError(f"Источник '{source_name}' не зарегистрирован")
        
        info = {
            'name': source_name,
            'extension': self.source_extensions.get(source_name, '.csv'),
            'has_config': source_name in self.sources_config
        }
        
        if source_name in self.sources_config:
            info['config'] = self.sources_config[source_name]
        
        return info
