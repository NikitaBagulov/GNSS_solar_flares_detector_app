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
    def __init__(self, base_download_dir: str = "./data"):
        """
        Инициализация менеджера данных
        
        Args:
            base_download_dir: Базовая директория для сохранения файлов
        """
        self.base_download_dir = Path(base_download_dir)
        self.base_download_dir.mkdir(parents=True, exist_ok=True)

        # Регистрация функций загрузки
        self.download_functions: Dict[str, Callable] = {}
        
        # Конфигурация источников
        self.sources_config: Dict[str, Dict] = {}
        
        # Расширения файлов по умолчанию для каждого источника
        self.source_extensions: Dict[str, str] = {}
        
        # Активные загрузки (для восстановления при прерывании)
        self._active_downloads: Dict[str, List[Path]] = {}
        self._transaction_file = self.base_download_dir / ".download_transactions.json"
        
        # Очистка старых временных файлов и восстановление
        self._cleanup_orphaned_temp_files()
        self._recover_interrupted_downloads()
        
        # Регистрация обработчиков завершения
        self._register_cleanup_handlers()
        
        print(f"📁 DataManager инициализирован в {self.base_download_dir}")
    
    def _register_cleanup_handlers(self):
        """Регистрация обработчиков для корректного завершения"""
        atexit.register(self._cleanup_all_downloads)
        
        # Обработчики сигналов прерывания
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        if hasattr(signal, 'SIGBREAK'):  # Windows
            signal.signal(signal.SIGBREAK, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов прерывания"""
        print(f"\n⚠️ Получен сигнал прерывания {signum}, очищаем временные файлы...")
        self._cleanup_all_downloads()
        sys.exit(1)
    
    def _cleanup_orphaned_temp_files(self):
        """Удаление старых временных файлов, оставшихся от предыдущих запусков"""
        print("🧹 Поиск старых временных файлов...")
        
        # Ищем все файлы с расширением .tmp в дереве директорий
        tmp_files = list(self.base_download_dir.rglob("*.tmp"))
        
        files_removed = 0
        for tmp_file in tmp_files:
            try:
                # Проверяем возраст файла (больше 1 часа)
                file_age = datetime.now().timestamp() - tmp_file.stat().st_mtime
                if file_age > 3600:  # 1 час в секундах
                    tmp_file.unlink()
                    files_removed += 1
                    print(f"   🗑️ Удален старый временный файл: {tmp_file}")
            except Exception as e:
                print(f"   ⚠️ Не удалось удалить {tmp_file}: {e}")
        
        if files_removed > 0:
            print(f"✅ Удалено {files_removed} старых временных файлов")
    
    def _save_transaction_state(self):
        """Сохранение состояния активных загрузок в файл"""
        try:
            state = {}
            for source, temp_files in self._active_downloads.items():
                state[source] = [str(f) for f in temp_files]
            
            with open(self._transaction_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ Ошибка сохранения состояния транзакций: {e}")
    
    def _recover_interrupted_downloads(self):
        """Восстановление после прерванных загрузок"""
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
                            print(f"   🗑️ Удален недокаченный файл: {temp_file}")
                        except Exception as e:
                            print(f"   ⚠️ Не удалось удалить {temp_file}: {e}")

            # Удаляем файл состояния транзакций
            self._transaction_file.unlink(missing_ok=True)
            
            if files_removed > 0:
                print(f"✅ Восстановление завершено: удалено {files_removed} недокаченных файлов")
                
        except Exception as e:
            print(f"⚠️ Ошибка восстановления после прерывания: {e}")
            self._transaction_file.unlink(missing_ok=True)
    
    def _cleanup_all_downloads(self):
        """Очистка всех активных загрузок"""
        print("🧹 Очистка активных загрузок...")
        files_removed = 0
        
        for source, temp_files in list(self._active_downloads.items()):
            for temp_file in temp_files[:]:  # Копируем список для безопасной итерации
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                        files_removed += 1
                        print(f"   🗑️ Удален временный файл: {temp_file}")
                    except Exception as e:
                        print(f"   ⚠️ Не удалось удалить {temp_file}: {e}")
                self._active_downloads[source].remove(temp_file)

        # Очищаем файл состояния
        self._transaction_file.unlink(missing_ok=True)
        
        if files_removed > 0:
            print(f"✅ Очистка завершена: удалено {files_removed} временных файлов")
    
    def _start_download_transaction(self, source_name: str, temp_file: Path):
        """Начало транзакции загрузки"""
        if source_name not in self._active_downloads:
            self._active_downloads[source_name] = []
        
        self._active_downloads[source_name].append(temp_file)
        self._save_transaction_state()
    
    def _complete_download_transaction(self, source_name: str, temp_file: Path):
        """Завершение транзакции загрузки"""
        if source_name in self._active_downloads and temp_file in self._active_downloads[source_name]:
            self._active_downloads[source_name].remove(temp_file)
            if not self._active_downloads[source_name]:
                del self._active_downloads[source_name]
            self._save_transaction_state()
    
    def register_download_function(self, 
        source_name: str, 
        download_func: Callable,
        config: Optional[Dict] = None,
        default_extension: str = '.csv'  # Расширение файла по умолчанию
    ):
        """
        Регистрация функции загрузки для источника
        
        Args:
            source_name: Имя источника данных
            download_func: Функция загрузки данных
            config: Конфигурация источника
            default_extension: Расширение файла по умолчанию (.csv, .h5, и т.д.)
        """
        self.download_functions[source_name] = download_func
        self.source_extensions[source_name] = default_extension
        
        if config:
            self.sources_config[source_name] = config
        
        print(f"📝 Зарегистрирован источник '{source_name}' с расширением '{default_extension}'")
    
    def download_by_date(
        self, 
        target_date: date,
        sources: Optional[List[str]] = None,
        force_redownload: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Скачать данные за указанную дату
        
        Args:
            target_date: Дата для скачивания
            sources: Список источников для скачивания (если None - все источники)
            force_redownload: Принудительное перескачивание
            **kwargs: Дополнительные параметры для функций загрузки
        
        Returns:
            Словарь с результатами скачивания для каждого источника
        """
        print(f"\n📥 ЗАГРУЗКА ДАННЫХ ЗА {target_date}")
        results = {}
        
        # Определяем источники для скачивания
        sources_to_download = sources or list(self.download_functions.keys())
        
        for source_name in sources_to_download:
            if source_name not in self.download_functions:
                print(f"⚠️ Источник '{source_name}' не зарегистрирован, пропускаем")
                continue
            
            try:
                download_func = self.download_functions[source_name]
                
                # Получаем расширение файла для этого источника
                extension = self.source_extensions.get(source_name, '.csv')
                
                # Формируем имя файла
                filename = f"{source_name}_{target_date.strftime('%Y%m%d')}{extension}"
                final_path = self.get_download_path(source_name, target_date, filename)
                
                # Проверяем существующий файл
                if not force_redownload and final_path.exists():
                    try:
                        if self._is_file_valid(final_path, source_name):
                            results[source_name] = {
                                'status': 'skipped',
                                'result': str(final_path),
                                'date': target_date,
                                'message': 'Файл уже существует',
                                'size': final_path.stat().st_size
                            }
                            print(f"   ⏭️ {source_name}: файл уже существует")
                            continue
                        else:
                            print(f"   ⚠️ {source_name}: файл поврежден, перезагружаем...")
                    except Exception as e:
                        print(f"   ⚠️ {source_name}: не удалось проверить файл ({e}), перезагружаем...")

                # Создаем путь для временного файла
                temp_path = final_path.with_suffix(extension + '.tmp')
                
                # Начинаем транзакцию
                self._start_download_transaction(source_name, temp_path)
                
                # Скачиваем данные
                print(f"   📥 {source_name}: скачивание...")
                result = download_func(
                    date=target_date,
                    data_manager=self,
                    force_redownload=force_redownload,
                    temp_path=temp_path,
                    **kwargs
                )
                
                # Проверяем результат
                if result and isinstance(result, Path) and temp_path.exists():
                    # Проверяем размер временного файла
                    if temp_path.stat().st_size == 0:
                        raise Exception("Временный файл пустой")
                    
                    # Атомарное переименование временного файла в постоянный
                    temp_path.rename(final_path)
                    
                    # Завершаем транзакцию
                    self._complete_download_transaction(source_name, temp_path)
                    
                    # Формируем результат успешной загрузки
                    results[source_name] = {
                        'status': 'success',
                        'result': str(final_path),
                        'date': target_date,
                        'size': final_path.stat().st_size
                    }
                    print(f"   ✅ {source_name}: успешно скачан ({final_path.stat().st_size / 1024:.1f} KB)")
                    
                else:
                    # Очищаем временный файл при неудаче
                    if temp_path.exists():
                        temp_path.unlink()
                    
                    # Завершаем транзакцию
                    self._complete_download_transaction(source_name, temp_path)
                    
                    results[source_name] = {
                        'status': 'error',
                        'error': 'Не удалось скачать файл',
                        'date': target_date
                    }
                    print(f"   ❌ {source_name}: ошибка скачивания")
                    
            except Exception as e:
                # Очищаем временный файл при исключении
                if 'temp_path' in locals() and temp_path.exists():
                    temp_path.unlink(missing_ok=True)
                    self._complete_download_transaction(source_name, temp_path)
                
                results[source_name] = {
                    'status': 'error',
                    'error': str(e),
                    'date': target_date
                }
                print(f"   ❌ {source_name}: ошибка - {e}")
        
        print(f"📊 ИТОГИ ЗАГРУЗКИ ЗА {target_date}:")
        for source, result in results.items():
            status = result.get('status', 'unknown')
            if status == 'success':
                print(f"   ✅ {source}: успешно")
            elif status == 'skipped':
                print(f"   ⏭️ {source}: пропущен")
            elif status == 'error':
                print(f"   ❌ {source}: ошибка")
        
        return results
    
    def _is_file_valid(self, file_path: Path, source_name: str) -> bool:
        """
        Проверка валидности файла
        
        Args:
            file_path: Путь к файлу
            source_name: Имя источника
        
        Returns:
            True если файл валиден, False в противном случае
        """
        if not file_path.exists():
            return False
        
        try:
            # Проверка размера файла
            file_size = file_path.stat().st_size
            if file_size == 0:
                return False
            
            # Проверка в зависимости от расширения файла
            extension = file_path.suffix.lower()
            
            if extension == '.csv':
                # Для CSV файлов проверяем, что их можно прочитать
                df = pd.read_csv(file_path, nrows=1)
                return not df.empty
                
            elif extension in ['.h5', '.hdf5', '.hdf']:
                # Для HDF5 файлов проверяем минимальный размер
                # (полная проверка требует библиотеки h5py)
                return file_size > 100  # Минимум 100 байт для HDF5
                
            elif extension in ['.nc', '.cdf']:
                # Для NetCDF файлов
                return file_size > 100
                
            else:
                # Для других форматов просто проверяем размер
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
        """
        Получить путь для сохранения файла
        
        Args:
            source_name: Имя источника данных
            target_date: Дата данных
            filename: Имя файла (если None - сгенерируется автоматически)
            create_dir: Создавать директорию, если она не существует
        
        Returns:
            Полный путь к файлу
        """
        # Формируем путь по шаблону: base/YYYY-MM-DD/source_name/
        date_dir = target_date.strftime('%Y-%m-%d')
        date_path = self.base_download_dir / date_dir / source_name
        
        if create_dir:
            date_path.mkdir(parents=True, exist_ok=True)
        
        if filename:
            return date_path / filename
        else:
            # Если имя файла не указано, генерируем его
            extension = self.source_extensions.get(source_name, '.csv')
            return date_path / f"{source_name}{extension}"
    
    def check_file_exists(
        self, 
        source_name: str, 
        target_date: date,
        filename: Optional[str] = None
    ) -> bool:
        """
        Проверка существования и валидности файла
        
        Args:
            source_name: Имя источника данных
            target_date: Дата данных
            filename: Имя файла
        
        Returns:
            True если файл существует и валиден
        """
        file_path = self.get_download_path(source_name, target_date, filename, create_dir=False)
        return file_path.exists() and self._is_file_valid(file_path, source_name)
    
    def list_downloaded_dates(self, source_name: str) -> List[date]:
        """
        Получить список дат, для которых есть данные
        
        Args:
            source_name: Имя источника данных
        
        Returns:
            Список дат с данными
        """
        dates = []
        source_dir = self.base_download_dir / source_name
        
        if not source_dir.exists():
            return dates
        
        # Ищем все директории с датами
        for item in source_dir.iterdir():
            if item.is_dir():
                try:
                    date_obj = datetime.strptime(item.name, '%Y-%m-%d').date()
                    dates.append(date_obj)
                except ValueError:
                    continue
        
        return sorted(dates)
    
    def get_available_sources(self) -> List[str]:
        """
        Получить список доступных источников
        
        Returns:
            Список зарегистрированных источников
        """
        return list(self.download_functions.keys())
    
    def get_source_info(self, source_name: str) -> Dict:
        """
        Получить информацию об источнике
        
        Args:
            source_name: Имя источника
        
        Returns:
            Словарь с информацией об источнике
        """
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