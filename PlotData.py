from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
from datetime import datetime

# Класс для вспышки
@dataclass
class FlareData:
    flare_id: int
    duration: float
    start_time: datetime
    peak_time: datetime
    end_time: datetime
    location: Tuple[float, float]

@dataclass
class PlotData:
    timestamps: List[datetime] = field(default_factory=list)
    product_values: List[tuple] = field(default_factory=list)

    xray_times: List[datetime] = field(default_factory=list)
    xray_values: List[float] = field(default_factory=list)

    euv_times: List[datetime] = field(default_factory=list)
    euv_values: List[float] = field(default_factory=list)

    index_times: List[datetime] = field(default_factory=list)
    index_times_by_product: Dict[str, List[datetime]] = field(default_factory=dict)
    indices: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)

    flare: List[FlareData] = field(default_factory=list)
    sun_image: Optional[object] = None
