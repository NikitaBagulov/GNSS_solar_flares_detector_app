import json
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Any
import pandas as pd
from sunpy.net import Fido, attrs as a
from astropy.time import Time
from DataManager import DataManager


class FlareTracker:
    def __init__(self, 
                 data_manager: DataManager,
                 min_year: int = 2020,
                 min_flare_class: str = "X1.0"):
        print(f"🚀 Инициализация FlareTracker")
        print(f"📁 Директория данных: {data_manager.base_download_dir}")
        print(f"📅 Минимальный год: {min_year}")
        print(f"⭐ Минимальный класс вспышек: {min_flare_class}")
        
        self.data_manager = data_manager
        self.min_year = min_year
        self.min_flare_class = min_flare_class

        self.state_file = self.data_manager.base_download_dir / "flare_tracker_state.json"
        print(f"📄 Файл состояния: {self.state_file}")

        self.state = self._load_state()
        self._register_download_functions()
        print(f"✅ FlareTracker инициализирован\n")
    
    def _load_state(self) -> Dict:
        print(f"📖 Загрузка состояния из {self.state_file}")
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    print(f"   ✓ Состояние загружено")
                    if state.get("last_check_date"):
                        print(f"   📅 Последняя проверка: {state['last_check_date']}")
                    print(f"   📊 Всего вспышек в истории: {state.get('total_flares', 0)}")
                    return state
            except Exception as e:
                print(f"   ⚠️ Ошибка загрузки состояния: {e}")
                print(f"   📝 Создание нового состояния")
        else:
            print(f"   📝 Файл состояния не найден, создание нового")

        return {
            "last_check_date": None,
            "min_year": self.min_year,
            "min_flare_class": self.min_flare_class,
            "flare_dates": [],
            "downloaded_flares": [],
            "total_flares": 0,
            "data_downloaded": [],
            "yearly_cache": {}  # Кэш по годам: {"2024": ["2024-01-15", "2024-03-22"]}
        }
    
    def _save_state(self):
        try:
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            print(f"💾 Состояние сохранено в {self.state_file}")
        except Exception as e:
            print(f"❌ Ошибка сохранения состояния: {e}")
    
    def _register_download_functions(self):
        print(f"🔧 Регистрация функций в DataManager...")
        self.data_manager.register_download_function(
            'flare_data',
            self.download_flare_data,  
            config={
                'description': 'Данные GOES X-ray за день вспышки',
                'min_class': self.min_flare_class
            }
        )
        print(f"   ✅ Функция 'flare_data' зарегистрирована")
    
    def get_flares_for_year(self, year: int) -> pd.DataFrame:
        """
        Получает все вспышки за указанный год одним запросом.
        """
        print(f"\n{'='*60}")
        print(f"📅 ПОЛУЧЕНИЕ ВСПЫШЕК ЗА {year} ГОД")
        print(f"⭐ Минимальный класс: {self.min_flare_class}")
        print(f"{'='*60}")
        
        # Формируем временной диапазон на весь год
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)
        
        tstart = start_date.strftime('%Y/%m/%d 00:00')
        tend = end_date.strftime('%Y/%m/%d 23:59')
        
        print(f"⏰ Временной диапазон: {tstart} - {tend}")
        print(f"🔍 Выполнение запроса к HEK...")
        
        try:
            # Выполняем один запрос на весь год
            result = Fido.search(
                a.Time(tstart, tend),
                a.hek.EventType("FL"),
                a.hek.FL.GOESCls > self.min_flare_class
            )
            
            if len(result) == 0:
                print(f"📭 Вспышек за {year} год не найдено")
                return pd.DataFrame()
            
            hek_results = result['hek']
            print(f"✅ Найдено {len(hek_results)} вспышек за {year} год")
            
            flares_data = []
            for flare in hek_results:
                try:
                    flare_class = flare.get('fl_goescls', '').strip()
                    
                    if not flare_class:
                        continue
                    
                    start_dt = Time(flare['event_starttime']).to_datetime()
                    peak_dt = Time(flare['event_peaktime']).to_datetime()
                    end_dt = Time(flare['event_endtime']).to_datetime()
                    
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
                        'date': start_dt.date()  # Добавляем дату для группировки
                    })
                except Exception as e:
                    print(f"   ⚠️ Ошибка обработки вспышки: {e}")
                    continue
            
            df = pd.DataFrame(flares_data)
            
            if not df.empty:
                df = df.sort_values('class_value', ascending=False)
                
                # Группируем по датам для статистики
                dates_with_flares = df['date'].unique()
                print(f"📊 Распределение по дням:")
                print(f"   📅 Дней с вспышками: {len(dates_with_flares)}")
                print(f"   📈 Всего вспышек: {len(df)}")
                
                # Статистика по месяцам
                df['month'] = df['start_time'].dt.month
                monthly_stats = df.groupby('month').size()
                print(f"📅 Распределение по месяцам:")
                for month in range(1, 13):
                    count = monthly_stats.get(month, 0)
                    if count > 0:
                        month_name = datetime(year, month, 1).strftime('%B')
                        print(f"   {month_name:9s}: {count:3d} вспышек")
                
                # Самые сильные вспышки
                if len(df) > 0:
                    print(f"\n⭐ САМЫЕ СИЛЬНЫЕ ВСПЫШКИ {year} ГОДА:")
                    top_flares = df.head(5)
                    for i, (_, flare) in enumerate(top_flares.iterrows(), 1):
                        print(f"   {i}. {flare['class']:6s} - {flare['start_time'].strftime('%Y-%m-%d %H:%M')}")
            
            print(f"{'='*60}\n")
            return df
            
        except Exception as e:
            print(f"❌ Ошибка получения вспышек за {year} год: {e}")
            return pd.DataFrame()
    
    def process_year(self, year: int, force_redownload: bool = False) -> Dict[str, Any]:
        """
        Обрабатывает весь год: получает вспышки и скачивает данные для дней с вспышками.
        """
        print(f"\n{'='*70}")
        print(f"📅 ОБРАБОТКА {year} ГОДА")
        print(f"{'='*70}")
        
        # Получаем все вспышки за год
        yearly_flares = self.get_flares_for_year(year)
        
        if yearly_flares.empty:
            print(f"📭 Нет вспышек за {year} год")
            return {
                'year': year,
                'status': 'no_flares',
                'total_flares': 0,
                'days_with_flares': 0,
                'processed_days': 0
            }
        
        # Группируем по датам
        dates_with_flares = yearly_flares['date'].unique()
        dates_with_flares = sorted(dates_with_flares)
        
        print(f"\n📊 ОБНАРУЖЕНО:")
        print(f"   📈 Всего вспышек: {len(yearly_flares)}")
        print(f"   📅 Дней с вспышками: {len(dates_with_flares)}")
        print(f"   📅 С {dates_with_flares[0]} по {dates_with_flares[-1]}")
        
        # Обновляем кэш по годам
        year_str = str(year)
        self.state.setdefault("yearly_cache", {})
        self.state["yearly_cache"][year_str] = [d.strftime("%Y-%m-%d") for d in dates_with_flares]
        
        # Обрабатываем каждый день с вспышками
        print(f"\n📥 ОБРАБОТКА ДНЕЙ С ВСПЫШКАМИ:")
        
        processed_days = 0
        successful_days = 0
        total_flares_processed = 0
        
        for flare_date in dates_with_flares:
            date_str = flare_date.strftime("%Y-%m-%d")
            
            # Пропускаем если уже обработано
            if date_str in self.state["data_downloaded"] and not force_redownload:
                print(f"   ⏭️  {date_str}: уже обработан (пропуск)")
                continue
            
            print(f"\n   [{processed_days + 1}/{len(dates_with_flares)}] 📅 {date_str}")
            
            # Фильтруем вспышки для этой даты
            day_flares = yearly_flares[yearly_flares['date'] == flare_date]
            
            # Сохраняем данные о вспышках
            flare_file = self.data_manager.get_download_path('flares', flare_date, f"flares_{date_str}.csv")
            day_flares.drop(columns=['date', 'month']).to_csv(flare_file, index=False)
            
            print(f"      📊 Вспышек: {len(day_flares)}")
            
            # Обновляем состояние
            for _, flare in day_flares.iterrows():
                flare_id = f"{date_str}_{flare['class']}_{flare['start_time'].strftime('%H%M')}"
                if flare_id not in self.state["downloaded_flares"]:
                    self.state["downloaded_flares"].append(flare_id)
            
            if date_str not in self.state["flare_dates"]:
                self.state["flare_dates"].append(date_str)
            
            self.state["total_flares"] += len(day_flares)
            total_flares_processed += len(day_flares)
            
            # Скачиваем данные через DataManager
            print(f"      🌐 Скачивание данных...")
            download_results = self.data_manager.download_by_date(
                target_date=flare_date,
                sources=None,
                force_redownload=force_redownload
            )
            
            successful_sources = []
            for source_name, result in download_results.items():
                if result['status'] == 'success':
                    successful_sources.append(source_name)
            
            if successful_sources:
                if date_str not in self.state["data_downloaded"]:
                    self.state["data_downloaded"].append(date_str)
                    successful_days += 1
                print(f"      ✅ Данные скачаны: {', '.join(successful_sources)}")
            else:
                print(f"      ⚠️  Данные не скачаны")
            
            processed_days += 1
        
        # Обновляем последнюю дату проверки
        if dates_with_flares:
            last_date = dates_with_flares[-1]
            self.state["last_check_date"] = last_date.strftime("%Y-%m-%d")
        
        # Сохраняем состояние
        self._save_state()
        
        print(f"\n{'='*70}")
        print(f"🎉 ОБРАБОТКА {year} ГОДА ЗАВЕРШЕНА")
        print(f"{'='*70}")
        print(f"📊 РЕЗУЛЬТАТЫ:")
        print(f"   📅 Всего дней с вспышками: {len(dates_with_flares)}")
        print(f"   📅 Обработано дней: {processed_days}")
        print(f"   📅 Успешно скачано: {successful_days}")
        print(f"   📈 Всего вспышек: {total_flares_processed}")
        print(f"   📊 Всего вспышек в истории: {self.state['total_flares']}")
        print(f"{'='*70}\n")
        
        return {
            'year': year,
            'status': 'success',
            'total_flares': len(yearly_flares),
            'days_with_flares': len(dates_with_flares),
            'processed_days': processed_days,
            'successful_days': successful_days
        }
    
    def process_multiple_years(self, start_year: int, end_year: int = None, force_redownload: bool = False):
        """
        Обрабатывает несколько лет подряд.
        """
        if end_year is None:
            end_year = datetime.now().year
        
        print(f"\n{'='*70}")
        print(f"📅 ОБРАБОТКА НЕСКОЛЬКИХ ЛЕТ")
        print(f"📅 Период: {start_year} - {end_year}")
        print(f"{'='*70}")
        
        results = {}
        
        for year in range(start_year, end_year + 1):
            print(f"\n🎯 ГОД {year}")
            result = self.process_year(year, force_redownload)
            results[year] = result
            
            if result['status'] == 'no_flares':
                print(f"   📭 Пропуск {year} года - нет вспышек")
        
        # Итоговая статистика
        print(f"\n{'='*70}")
        print(f"📊 ИТОГОВАЯ СТАТИСТИКА")
        print(f"{'='*70}")
        
        total_years_processed = 0
        total_years_with_flares = 0
        total_flares = 0
        total_days_with_flares = 0
        
        for year, result in results.items():
            if result['status'] == 'success':
                total_years_processed += 1
                total_years_with_flares += 1
                total_flares += result['total_flares']
                total_days_with_flares += result['days_with_flares']
                print(f"   {year}: {result['total_flares']} вспышек, {result['days_with_flares']} дней")
            elif result['status'] == 'no_flares':
                total_years_processed += 1
                print(f"   {year}: нет вспышек")
        
        print(f"{'='*70}")
        print(f"📈 ВСЕГО:")
        print(f"   📅 Обработано лет: {total_years_processed}")
        print(f"   📅 Лет с вспышками: {total_years_with_flares}")
        print(f"   📈 Всего вспышек: {total_flares}")
        print(f"   📅 Всего дней с вспышками: {total_days_with_flares}")
        print(f"   📊 Всего вспышек в истории: {self.state['total_flares']}")
        print(f"{'='*70}\n")
        
        return results
    
    def get_missed_years(self) -> List[int]:
        """
        Возвращает список лет, которые еще не обработаны.
        """
        current_year = datetime.now().year
        missed_years = []
        
        # Проверяем каждый год с min_year по текущий
        for year in range(self.min_year, current_year + 1):
            year_str = str(year)
            
            # Проверяем есть ли этот год в кэше
            if year_str in self.state.get("yearly_cache", {}):
                # Год был обработан, проверяем есть ли непрошедшие дни
                cached_dates = self.state["yearly_cache"][year_str]
                all_dates_processed = all(date_str in self.state["data_downloaded"] for date_str in cached_dates)
                
                if not all_dates_processed:
                    missed_years.append(year)
            else:
                # Год вообще не обрабатывался
                missed_years.append(year)
        
        return missed_years
    
    def download_missed_data(self, limit_years: Optional[int] = None, limit_days: Optional[int] = None):
        """
        Скачивает пропущенные данные, начиная с проверки целых лет.
        """
        print(f"\n{'='*70}")
        print(f"📥 СКАЧИВАНИЕ ПРОПУЩЕННЫХ ДАННЫХ")
        print(f"{'='*70}")
        
        # Сначала получаем пропущенные годы
        missed_years = self.get_missed_years()
        
        if not missed_years:
            print(f"✅ НЕТ ПРОПУЩЕННЫХ ДАННЫХ")
            print(f"{'='*70}\n")
            return
        
        if limit_years:
            missed_years = missed_years[:limit_years]
        
        print(f"📅 ПРОПУЩЕННЫЕ ГОДЫ: {len(missed_years)}")
        for year in missed_years:
            print(f"   • {year}")
        
        print(f"\n🔄 НАЧИНАЕМ ОБРАБОТКУ...")
        
        # Обрабатываем каждый пропущенный год
        for i, year in enumerate(missed_years, 1):
            print(f"\n[{i}/{len(missed_years)}] 🎯 ОБРАБОТКА {year} ГОДА")
            self.process_year(year)
        
        print(f"\n{'='*70}")
        print(f"🎉 ВСЕ ПРОПУЩЕННЫЕ ДАННЫЕ ОБРАБОТАНЫ")
        print(f"{'='*70}\n")
    
    # Остальные методы остаются без изменений...
    def _get_flares_for_date_api(self, target_date: date) -> pd.DataFrame:
        """Заглушка для совместимости - теперь используем get_flares_for_year"""
        date_str = target_date.strftime("%Y-%m-%d")
        year = target_date.year
        
        print(f"   🔍 Запрос вспышек за {date_str} через годовой кэш")
        
        # Проверяем кэш года
        year_str = str(year)
        if year_str in self.state.get("yearly_cache", {}):
            cached_dates = self.state["yearly_cache"][year_str]
            if date_str in cached_dates:
                # Дата есть в кэше, читаем из файла
                flare_file = self.data_manager.get_download_path('flares', target_date, f"flares_{date_str}.csv")
                if flare_file.exists():
                    try:
                        df = pd.read_csv(flare_file, parse_dates=['start_time', 'peak_time', 'end_time'])
                        print(f"   ✅ Вспышки загружены из кэша: {len(df)} вспышек")
                        return df
                    except:
                        pass
        
        print(f"   ⚠️  Дата не найдена в кэше, возвращаем пустой DataFrame")
        return pd.DataFrame()
    
    def download_flare_data(self, target_date: date, force_redownload: bool = False) -> Dict[str, Any]:
        """Совместимый метод для обработки одного дня"""
        date_str = target_date.strftime("%Y-%m-%d")
        print(f"\n📥 ОБРАБОТКА ОДНОГО ДНЯ: {date_str}")
        
        # Используем годовую логику
        year = target_date.year
        year_result = self.process_year(year, force_redownload)
        
        return {
            'date': date_str,
            'year': year,
            'status': year_result['status'],
            'flares_count': year_result.get('total_flares', 0),
            'data_downloaded': []
        }
    
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
    