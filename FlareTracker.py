import json
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from sunpy.net import Fido, attrs as a
from astropy.time import Time
from DataManager import DataManager
import os


class FlareTracker:
    def __init__(self, 
                 data_manager: DataManager,
                 start_date: Optional[date] = None,
                 end_date: Optional[date] = None,
                 min_flare_class: str = "X1.0",
                 state_file_path: Optional[Path] = None):
        print(f"🚀 Инициализация FlareTracker")
        print(f"📁 Директория данных: {data_manager.base_download_dir}")

        current_date = datetime.now().date()
        
        if start_date is None:
            start_date = date(2010, 1, 1)
        
        if end_date is None:
            end_date = current_date
        
        print(f"📅 Начальная дата: {start_date}")
        print(f"📅 Конечная дата: {end_date}")
        
        
        self.data_manager = data_manager
        self.start_date = start_date
        self.end_date = end_date
        self.min_flare_class = min_flare_class
        print(f"⭐ Минимальный класс вспышек: {self.min_flare_class}")

        if state_file_path is None:
            self.state_file = self.data_manager.base_download_dir / "flare_tracker_state.json"
        else:
            self.state_file = Path(state_file_path)
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"📄 Файл состояния: {self.state_file}")

        self.state = self._load_state()

        self.all_flares_file = self.state_file.parent / "all_flares.csv"
        print(f"📊 Файл всех вспышек: {self.all_flares_file}")
        
        # Автоматически синхронизируем состояние с файлами при инициализации
        self._sync_state_with_files()
        
        print(f"✅ FlareTracker инициализирован\n")
    
    def _load_state(self) -> Dict:
        print(f"📖 Загрузка состояния из {self.state_file}")
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    print(f"   ✓ Состояние загружено")
                    if state.get("last_update_date"):
                        print(f"   📅 Последнее обновление: {state['last_update_date']}")
                    print(f"   📊 Всего вспышек в истории: {state.get('total_flares', 0)}")
                    print(f"   💾 В state записано дней со скачанными данными: {len(state.get('data_downloaded', []))}")
                    return state
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки состояния: {e}")
                print(f"   📝 Создание нового состояния")
        else:
            print(f"   📝 Файл состояния не найден, создание нового")

        return {
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "min_flare_class": self.min_flare_class,
            "last_update_date": None,
            "flare_dates": [],
            "total_flares": 0,
            "data_downloaded": [],
            "files_by_date": {},
            "files_by_flare": {}
        }

    def register_files_for_date(self, date: date, files: dict):
        """
        Регистрирует файлы, связанные с конкретной датой вспышек.
        Добавляет данные в state и сразу сохраняет.
        """
        date_str = date.strftime("%Y-%m-%d")

        if "files_by_date" not in self.state:
            self.state["files_by_date"] = {}

        if date_str not in self.state["files_by_date"]:
            self.state["files_by_date"][date_str] = {}

        # Преобразуем все пути в строки
        for key, value in files.items():
            if isinstance(value, Path):
                self.state["files_by_date"][date_str][key] = str(value)
            elif isinstance(value, list):
                self.state["files_by_date"][date_str][key] = [str(p) if isinstance(p, Path) else p for p in value]
            elif isinstance(value, dict):
                self.state["files_by_date"][date_str][key] = {
                    k: str(v) if isinstance(v, Path) else v for k, v in value.items()
                }
            else:
                # Если передали строку или другой тип
                self.state["files_by_date"][date_str][key] = str(value)

        # Отладка: выводим, что зарегистрировано
        print(f"\n🛠️ [DEBUG] Файлы зарегистрированы для {date_str}:")
        for k, v in self.state["files_by_date"][date_str].items():
            print(f"   {k}: {v}")

        # Сохраняем состояние сразу после регистрации
        self._save_state(message=f"Файлы зарегистрированы для {date_str} (DEBUG)")

    def register_files_for_flare(self, flare_key: str, files: dict):
        if "files_by_flare" not in self.state:
            self.state["files_by_flare"] = {}

        if flare_key not in self.state["files_by_flare"]:
            self.state["files_by_flare"][flare_key] = {}

        for key, value in files.items():
            if isinstance(value, Path):
                self.state["files_by_flare"][flare_key][key] = str(value)
            elif isinstance(value, list):
                self.state["files_by_flare"][flare_key][key] = [str(p) if isinstance(p, Path) else p for p in value]
            elif isinstance(value, dict):
                self.state["files_by_flare"][flare_key][key] = {
                    k: str(v) if isinstance(v, Path) else v for k, v in value.items()
                }
            else:
                self.state["files_by_flare"][flare_key][key] = str(value)

        print(f"\n🛠️ [DEBUG] Файлы зарегистрированы для вспышки {flare_key}:")
        for k, v in self.state["files_by_flare"][flare_key].items():
            print(f"   {k}: {v}")

        self._save_state(message=f"Файлы зарегистрированы для вспышки {flare_key} (DEBUG)")


    def get_files_for_flare_date(self, flare_date: date) -> dict:
        date_str = flare_date.strftime("%Y-%m-%d")
        return self.state.get("files_by_date", {}).get(date_str, {})

    def get_files_for_flare(self, flare_key: str) -> dict:
        return self.state.get("files_by_flare", {}).get(flare_key, {})

    def get_flares_for_date(self, flare_date: date) -> List[Dict[str, Any]]:
        all_flares = self._load_all_flares()
        if all_flares.empty:
            return []

        day_flares = all_flares[all_flares["date"] == flare_date]
        flares = []
        for _, row in day_flares.iterrows():
            flares.append(
                {
                    "class": row.get("class"),
                    "start_time": row.get("start_time"),
                    "peak_time": row.get("peak_time"),
                    "end_time": row.get("end_time"),
                }
            )
        return flares

    def _sync_state_with_files(self):
        """Синхронизирует состояние с реально существующими файлами"""
        print(f"\n🔍 СИНХРОНИЗАЦИЯ СОСТОЯНИЯ С ФАЙЛАМИ...")
        
        # Загружаем вспышки
        all_flares = self._load_all_flares()
        if all_flares.empty:
            print(f"   📭 Нет вспышек в файле")
            self.state["data_downloaded"] = []
            self._save_state(message="Состояние очищено - нет вспышек")
            return
        
        # Получаем уникальные даты со вспышками
        flare_dates_set = set()
        for date_str in all_flares['date'].unique():
            try:
                if isinstance(date_str, str):
                    flare_date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
                elif isinstance(date_str, pd.Timestamp):
                    flare_date = date_str.date()
                elif isinstance(date_str, date):
                    flare_date = date_str
                else:
                    continue
                flare_dates_set.add(flare_date)
            except Exception as e:
                print(f"   ⚠️ Ошибка преобразования даты {date_str}: {e}")
                continue
        
        flare_dates = sorted(flare_dates_set)
        
        print(f"   📊 Всего дней со вспышками: {len(flare_dates)}")
        
        # Получаем список источников
        available_sources = list(self.data_manager.download_functions.keys())
        print(f"   📁 Проверяемые источники: {available_sources}")
        
        # Проверяем какие даты действительно скачаны
        actually_downloaded_dates = []
        dates_with_missing_files = []
        
        for flare_date in flare_dates:
            date_str = flare_date.strftime("%Y-%m-%d")
            all_sources_have_data = True
            
            for source in available_sources:
                if not self._check_source_has_data(source, flare_date):
                    all_sources_have_data = False
                    dates_with_missing_files.append(date_str)
                    break
            
            if all_sources_have_data:
                actually_downloaded_dates.append(date_str)
        
        # Обновляем состояние
        self.state["data_downloaded"] = sorted(actually_downloaded_dates)
        
        # Обновляем список дат со вспышками
        flare_date_strings = [d.strftime("%Y-%m-%d") for d in flare_dates]
        self.state["flare_dates"] = flare_date_strings
        
        # Обновляем общее количество вспышек
        self.state["total_flares"] = len(all_flares)
        
        print(f"\n📊 РЕЗУЛЬТАТ СИНХРОНИЗАЦИИ:")
        print(f"   ✅ Файлы существуют для: {len(actually_downloaded_dates)} дней")
        print(f"   ⚠️ Отсутствуют файлы для: {len(dates_with_missing_files)} дней")
        
        if actually_downloaded_dates:
            print(f"\n   📅 Дни с полными данными:")
            for date_str in actually_downloaded_dates[:5]:
                print(f"      - {date_str}")
            if len(actually_downloaded_dates) > 5:
                print(f"      ... и еще {len(actually_downloaded_dates) - 5} дней")
        
        if dates_with_missing_files:
            print(f"\n   ⚠️ Дни с неполными данными:")
            for date_str in dates_with_missing_files[:5]:
                print(f"      - {date_str}")
            if len(dates_with_missing_files) > 5:
                print(f"      ... и еще {len(dates_with_missing_files) - 5} дней")
        
        self._save_state(message="Состояние синхронизировано с файлами")
    
    def _save_state(self, force: bool = False, message: str = ""):
        """Сохраняет состояние с опциональным принудительным сохранением"""
        try:
            # Обновляем время последнего обновления
            self.state["last_update_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            
            if message:
                print(f"💾 {message}")
            elif force:
                print(f"💾 Состояние принудительно сохранено в {self.state_file}")
            else:
                print(f"💾 Состояние сохранено в {self.state_file}")
                
            # Выводим информацию о сохраненном состоянии
            print(f"   📊 Дней со скачанными данными в state: {len(self.state.get('data_downloaded', []))}")
            print(f"   📅 Всего дней со вспышками: {len(self.state.get('flare_dates', []))}")
            print(f"   ⭐ Всего вспышек: {self.state.get('total_flares', 0)}")
            
        except Exception as e:
            print(f"❌ Ошибка сохранения состояния: {e}")
    
    def _save_state_if_needed(self, counter: int, every_n: int = 2) -> bool:
        """Сохраняет состояние каждые every_n операций"""
        if counter % every_n == 0:
            self._save_state(force=True, message=f"Промежуточное состояние сохранено (шаг {counter})")
            return True
        return False
    
    def _save_all_flares(self, df: pd.DataFrame):
        try:
            df.to_csv(self.all_flares_file, index=False)
            print(f"💾 Все вспышки сохранены в {self.all_flares_file}")
            print(f"   📊 Всего записей: {len(df)}")
            
            # После сохранения вспышек синхронизируем состояние
            self._sync_state_with_files()
            
        except Exception as e:
            print(f"❌ Ошибка сохранения всех вспышек: {e}")
    
    def _load_all_flares(self) -> pd.DataFrame:
        if self.all_flares_file.exists():
            try:
                df = pd.read_csv(self.all_flares_file)

                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date

                time_columns = ['start_time', 'peak_time', 'end_time']
                for col in time_columns:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col])
                
                return df
            except Exception as e:
                print(f"⚠️ Ошибка загрузки всех вспышек: {e}")

        return pd.DataFrame(columns=[
            'class', 'class_value', 'start_time', 'peak_time', 'end_time',
            'duration_min', 'hpc_x', 'hpc_y', 'peak_flux', 'date'
        ])
    
    def get_all_flares_in_range(self) -> pd.DataFrame:
        print(f"\n{'='*70}")
        print(f"📅 ПОЛУЧЕНИЕ ВСЕХ ВСПЫШЕК")
        print(f"📅 Диапазон: {self.start_date} - {self.end_date}")
        print(f"⭐ Минимальный класс: {self.min_flare_class}")
        print(f"{'='*70}")

        tstart = self.start_date.strftime('%Y/%m/%d 00:00')
        tend = self.end_date.strftime('%Y/%m/%d 23:59')
        
        print(f"⏰ Временной диапазон: {tstart} - {tend}")
        print(f"🔍 Выполнение запроса к HEK...")
        
        try:
            result = Fido.search(
                a.Time(tstart, tend),
                a.hek.EventType("FL"),
                a.hek.FL.GOESCls > self.min_flare_class
            )
            
            if len(result) == 0:
                print(f"📭 Вспышек в указанном диапазоне не найдено")
                return pd.DataFrame()
            
            hek_results = result['hek']
            print(f"✅ Найдено {len(hek_results)} вспышек")
            
            flares_data = []
            for flare in hek_results:
                try:
                    flare_class = flare.get('fl_goescls', '').strip()
                    
                    if not flare_class:
                        continue
                    
                    start_dt = Time(flare['event_starttime']).to_datetime()
                    peak_dt = Time(flare['event_peaktime']).to_datetime()
                    end_dt = Time(flare['event_endtime']).to_datetime()
                    flare_date = start_dt.date()
                    
                    flares_data.append({
                        'class': flare_class,
                        'class_value': self._flare_class_to_numeric(flare_class),
                        'start_time': start_dt,
                        'peak_time': peak_dt,
                        'end_time': end_dt,
                        'duration_min': (end_dt - start_dt).total_seconds() / 60,
                        'hpc_x': flare.get('hpc_x'),
                        'hpc_y': flare.get('hpc_y'),
                        'peak_flux': flare.get('fl_peakflux'),
                        'date': flare_date
                    })
                except Exception as e:
                    print(f"   ⚠️ Ошибка обработки вспышки: {e}")
                    continue
            
            df = pd.DataFrame(flares_data)
            
            if not df.empty:
                df = df.sort_values(['date', 'class_value'], ascending=[True, False])
            
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения вспышек: {e}")
            return pd.DataFrame()
    
    def _update_flares_from_api(self) -> Dict[str, Any]:
        """Проверить API и дополнить файл новыми вспышками"""
        print(f"\n{'='*70}")
        print(f"🔍 ПРОВЕРКА API НА НОВЫЕ ВСПЫШКИ")
        print(f"📅 Диапазон: {self.start_date} - {self.end_date}")
        print(f"{'='*70}")

        existing_flares = self._load_all_flares()

        api_flares = self.get_all_flares_in_range()
        
        if api_flares.empty:
            print(f"📭 На API нет вспышек в указанном диапазоне")
            return {'new_flares': 0, 'total_flares': len(existing_flares)}
        
        if existing_flares.empty:
            self._save_all_flares(api_flares)
            print(f"✅ Файл был пуст, сохранены {len(api_flares)} вспышек из API")
            
            return {'new_flares': len(api_flares), 'total_flares': len(api_flares)}

        existing_dates = set()
        for date_val in existing_flares['date']:
            try:
                if isinstance(date_val, str):
                    d = datetime.strptime(str(date_val), "%Y-%m-%d").date()
                elif isinstance(date_val, pd.Timestamp):
                    d = date_val.date()
                elif isinstance(date_val, date):
                    d = date_val
                else:
                    continue
                existing_dates.add(d)
            except:
                continue
        
        api_dates = set()
        for date_val in api_flares['date']:
            try:
                if isinstance(date_val, str):
                    d = datetime.strptime(str(date_val), "%Y-%m-%d").date()
                elif isinstance(date_val, pd.Timestamp):
                    d = date_val.date()
                elif isinstance(date_val, date):
                    d = date_val
                else:
                    continue
                api_dates.add(d)
            except:
                continue

        missing_dates = api_dates - existing_dates
        
        if not missing_dates:
            print(f"📭 Все вспышки из API уже есть в файле")
            return {'new_flares': 0, 'total_flares': len(existing_flares)}
        
        print(f"📊 Найдено {len(missing_dates)} новых дней со вспышками")

        missing_flares = []
        for _, flare in api_flares.iterrows():
            flare_date = flare['date']
            if isinstance(flare_date, str):
                d = datetime.strptime(str(flare_date), "%Y-%m-%d").date()
            elif isinstance(flare_date, pd.Timestamp):
                d = flare_date.date()
            elif isinstance(flare_date, date):
                d = flare_date
            else:
                continue
            
            if d in missing_dates:
                missing_flares.append(flare)

        if missing_flares:
            new_flares_df = pd.DataFrame(missing_flares)
            updated_flares = pd.concat([existing_flares, new_flares_df], ignore_index=True)
            updated_flares = updated_flares.sort_values(['date', 'class_value'], ascending=[True, False])
            
            self._save_all_flares(updated_flares)
            
            print(f"✅ Добавлено {len(missing_flares)} новых вспышек")
            print(f"📊 Всего вспышек в файле: {len(updated_flares)}")
            
            dates_with_flares = sorted(updated_flares['date'].unique())
            date_strings = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates_with_flares]
            
            self.state["flare_dates"] = date_strings
            self.state["total_flares"] = len(updated_flares)
            
            self._save_state(message="Состояние сохранено после обновления из API")
            
            return {
                'new_flares': len(missing_flares),
                'total_flares': len(updated_flares),
                'new_dates': [d.strftime("%Y-%m-%d") for d in sorted(missing_dates)]
            }
        
        return {'new_flares': 0, 'total_flares': len(existing_flares)}
    
    def download_missed_data(self):
        print(f"\n{'='*70}")
        print(f"📥 ОБРАБОТКА ДАННЫХ")
        print(f"📅 Диапазон: {self.start_date} - {self.end_date}")
        print(f"{'='*70}")

        # 1. Всегда проверяем API на новые вспышки
        print(f"\n🔍 ВСЕГДА ПРОВЕРЯЕМ API НА НОВЫЕ ВСПЫШКИ")
        api_result = self._update_flares_from_api()
        
        if api_result['new_flares'] > 0:
            print(f"✅ Найдено {api_result['new_flares']} новых вспышек")
            print(f"📊 Всего вспышек: {api_result['total_flares']}")
        else:
            print(f"📭 Новых вспышек не найдено")
            print(f"📊 Всего вспышек: {api_result['total_flares']}")

        # 2. Загружаем данные через DataManager для дней со вспышками
        print(f"\n{'='*70}")
        print(f"📥 СКАЧИВАНИЕ ДАННЫХ ЧЕРЕЗ DataManager")
        print(f"{'='*70}")

        # Синхронизируем состояние перед началом скачивания
        self._sync_state_with_files()
        
        all_flares = self._load_all_flares()
        
        if all_flares.empty:
            print(f"📭 Нет вспышек для скачивания данных")
            print(f"{'='*70}\n")
            return None

        # Получаем список дат для скачивания (тех, что в state как не скачанные)
        flare_dates = self.get_flare_dates()
        downloaded_dates = set(self.state.get("data_downloaded", []))
        
        dates_to_download = []
        for flare_date in flare_dates:
            date_str = flare_date.strftime("%Y-%m-%d")
            if date_str not in downloaded_dates:
                dates_to_download.append(flare_date)
        
        total_dates = len(dates_to_download)
        
        if total_dates == 0:
            print(f"📭 Нет дат для скачивания (все уже скачаны)")
            print(f"{'='*70}\n")
            return {
                'status': 'all_downloaded',
                'downloaded_dates': len(downloaded_dates)
            }
        
        print(f"📊 Найдено дней для скачивания: {total_dates}")
        print(f"📅 С {dates_to_download[0]} по {dates_to_download[-1]}")

        # Получаем список источников
        available_sources = list(self.data_manager.download_functions.keys())
        print(f"\n📁 Доступные источники: {available_sources}")

        # 3. Скачиваем данные для каждого дня
        print(f"\n📥 НАЧИНАЕМ СКАЧИВАНИЕ...")
        
        success_count = 0
        failed_dates = []
        
        for i, flare_date in enumerate(dates_to_download, 1):
            date_str = flare_date.strftime("%Y-%m-%d")
            print(f"\n[{i}/{len(dates_to_download)}] 📅 {date_str}")
            
            try:
                # Скачиваем все типы данных через DataManager
                download_result = self.data_manager.download_by_date(target_date=flare_date, tracker=self)
                
                if download_result:
                    # Проверяем результаты для каждого источника
                    all_sources_success = True
                    for source, result in download_result.items():
                        status = result.get('status', 'unknown')
                        
                        if status == 'success':
                            print(f"   ✅ {source}: скачан")
                        elif status == 'skipped':
                            print(f"   ⏭️ {source}: уже существует")
                        elif status == 'error':
                            print(f"   ❌ {source}: ошибка - {result.get('error', 'неизвестно')}")
                            all_sources_success = False
                        else:
                            print(f"   ❓ {source}: неизвестный статус - {status}")
                            all_sources_success = False
                    
                    if all_sources_success:
                        files_exist = True
                        missing_sources = []
                        
                        for source in available_sources:
                            if not self._check_source_has_data(source, flare_date):
                                print(f"   ⚠️ Отсутствует файл для источника: {source}")
                                files_exist = False
                                missing_sources.append(source)
                        
                        if files_exist:
                            date_str = flare_date.strftime("%Y-%m-%d")
                            if date_str not in self.state["data_downloaded"]:
                                self.state["data_downloaded"].append(date_str)
                                self.state["data_downloaded"] = sorted(self.state["data_downloaded"])
                                self._save_state(message=f"Дата {date_str} подтверждена как полностью скачанная")
                            
                            print(f"   ✅ Все файлы существуют → дата {date_str} подтверждена")
                            success_count += 1
                        else:
                            print(f"   ❌ Не хватает файлов от источников: {missing_sources}")
                            failed_dates.append(date_str)
                    else:
                        print(f"   ⚠️ Не все источники успешно скачаны")
                        failed_dates.append(date_str)
                else:
                    print(f"   ⚠️ Результат скачивания пустой")
                    failed_dates.append(date_str)
                    
            except Exception as e:
                print(f"   ❌ Ошибка скачивания: {e}")
                failed_dates.append(date_str)
            
            # Дополнительное сохранение каждые 2 дня для надежности
            self._save_state_if_needed(i, every_n=2)

        # 4. Финальная синхронизация и сохранение
        print(f"\n🔄 ФИНАЛЬНАЯ СИНХРОНИЗАЦИЯ СОСТОЯНИЯ...")
        self._sync_state_with_files()
        
        print(f"\n{'='*70}")
        print(f"📊 ИТОГИ СКАЧИВАНИЯ:")
        print(f"   ✅ Успешно скачано: {success_count} дней")
        print(f"   ❌ Не удалось скачать: {len(failed_dates)} дней")
        print(f"   💾 Всего дней со скачанными данными: {len(self.state.get('data_downloaded', []))}")
        
        if failed_dates:
            print(f"\n📝 Даты с ошибками:")
            for date_str in failed_dates[:10]:
                print(f"   - {date_str}")
            if len(failed_dates) > 10:
                print(f"   ... и еще {len(failed_dates) - 10} дней")
        
        print(f"{'='*70}\n")
        
        return {
            'status': 'completed',
            'success_count': success_count,
            'failed_count': len(failed_dates),
            'failed_dates': failed_dates,
            'downloaded_dates': len(self.state.get('data_downloaded', [])),
            'api_check': api_result
        }

    def _check_source_has_data(self, source_name, target_date, file_hint=None):
        try:
            if file_hint:
                p = Path(file_hint)
                if p.exists() and p.stat().st_size > 0:
                    return True

            patterns = [
                f"{source_name}_{target_date.strftime('%Y%m%d')}",
                f"{source_name}-{target_date.strftime('%Y%m%d')}",
                f"{target_date.strftime('%Y%m%d')}",
                f"{source_name}"
            ]

            exts = ['.csv', '.h5', '.hdf5', '.dat', '.nc']

            base_dir = self.data_manager.get_download_path(source_name, target_date, ".", create_dir=False)
            if base_dir is None:
                base_dir = self.data_manager.base_download_dir
            base_dir = Path(base_dir)
            if not base_dir.exists():
                base_dir = base_dir.parent if base_dir.parent.exists() else Path(".")

            for root, dirs, files in os.walk(base_dir):
                for f in files:
                    fp = Path(root) / f
                    name = fp.name
                    if any(p in name for p in patterns) and fp.suffix in exts and fp.stat().st_size > 0:
                        return True

            return False

        except:
            return False

    
    def _flare_class_to_numeric(self, flare_class: str) -> float:
        if not isinstance(flare_class, str):
            return 0.0
        
        flare_class = flare_class.strip().upper()
        
        multiplier = {
            'X': 10.0,
            'M': 1.0,
            'C': 0.1,
            'B': 0.01,
            'A': 0.001
        }
        
        if not flare_class:
            return 0.0
        
        letter = flare_class[0]
        number_part = flare_class[1:] if len(flare_class) > 1 else '1.0'
        
        try:
            number = float(number_part)
        except:
            number = 1.0
        
        return multiplier.get(letter, 0.0) * number

    def get_flare_dates(self) -> List[date]:
        all_flares = self._load_all_flares()
        if all_flares.empty:
            return []
        
        dates = []
        for date_str in all_flares['date'].unique():
            try:
                if isinstance(date_str, str):
                    flare_date = datetime.strptime(str(date_str), "%Y-%m-%d").date()
                elif isinstance(date_str, pd.Timestamp):
                    flare_date = date_str.date()
                elif isinstance(date_str, date):
                    flare_date = date_str
                else:
                    continue
                dates.append(flare_date)
            except:
                continue
        
        return sorted(dates)

    def get_dates_to_download(self) -> List[date]:
        flare_dates = self.get_flare_dates()
        downloaded_dates = set(self.state.get("data_downloaded", []))
        
        dates_to_download = []
        for flare_date in flare_dates:
            date_str = flare_date.strftime("%Y-%m-%d")
            if date_str not in downloaded_dates:
                dates_to_download.append(flare_date)
        
        return dates_to_download
    
    def force_update_from_api(self):
        print(f"\n{'='*70}")
        print(f"🔄 ПРИНУДИТЕЛЬНОЕ ОБНОВЛЕНИЕ ИЗ API")
        print(f"{'='*70}")
        api_flares = self.get_all_flares_in_range()
        
        if api_flares.empty:
            print(f"📭 На API нет вспышек в указанном диапазоне")
            return False
        self._save_all_flares(api_flares)

        dates_with_flares = sorted(api_flares['date'].unique())
        date_strings = [d.strftime("%Y-%m-%d") if hasattr(d, 'strftime') else str(d) for d in dates_with_flares]
        
        self.state["flare_dates"] = date_strings
        self.state["total_flares"] = len(api_flares)
        self.state["last_update_date"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.state["data_downloaded"] = []
        
        # Синхронизируем с файлами после обновления
        self._sync_state_with_files()
        
        print(f"✅ Данные полностью обновлены из API")
        print(f"📊 Сохранено {len(api_flares)} вспышек")
        print(f"📅 Дней со вспышками: {len(dates_with_flares)}")
        print(f"{'='*70}\n")
        
        return True
    
    def get_status(self):
        """Возвращает текущий статус системы"""
        return {
            'total_flare_days': len(self.state.get('flare_dates', [])),
            'downloaded_days': len(self.state.get('data_downloaded', [])),
            'total_flares': self.state.get('total_flares', 0),
            'last_update': self.state.get('last_update_date', 'никогда'),
            'min_flare_class': self.state.get('min_flare_class', 'X1.0'),
            'date_range': {
                'start': self.state.get('start_date'),
                'end': self.state.get('end_date')
            }
        }
