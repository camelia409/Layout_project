import os, sqlite3, time, warnings, math
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
warnings.filterwarnings('ignore')

# SECTION 1 — IMPORTS AND CONSTANTS
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH    = os.path.join(BASE_DIR, 'db', 'floorplan.db')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

WALL_EXTERIOR = 0.23
WALL_INTERIOR = 0.115

FACING_MAP  = {'N': 0, 'S': 1, 'E': 2, 'W': 3}
CLIMATE_MAP = {'Hot_Humid': 0, 'Hot_Dry': 1, 'Composite': 2, 'Warm_Humid': 3}
ROAD_SIDE = {'N': 'north', 'S': 'south', 'E': 'east', 'W': 'west'}

ROOM_LISTS = {
    1: ['master_bedroom', 'toilet_attached', 'living', 'kitchen', 'verandah'],
    2: ['master_bedroom', 'toilet_attached', 'bedroom_2', 'living', 'dining', 'kitchen', 'toilet_common', 'utility', 'verandah'],
    3: ['master_bedroom', 'toilet_attached', 'bedroom_2', 'bedroom_3', 'living', 'dining', 'kitchen', 'toilet_common', 'utility', 'verandah', 'pooja'],
    4: ['master_bedroom', 'toilet_attached', 'bedroom_2', 'bedroom_3', 'bedroom_4', 'living', 'dining', 'kitchen', 'toilet_common', 'utility', 'verandah', 'pooja', 'store'],
}

NBC_MIN_AREA = {
    'master_bedroom': 9.5, 'bedroom_2': 7.5, 'bedroom_3': 7.5,
    'bedroom_4': 7.5, 'living': 9.5, 'dining': 5.0,
    'kitchen': 4.5, 'toilet_attached': 2.5, 'toilet_common': 2.5,
    'utility': 2.0, 'verandah': 4.0, 'pooja': 1.2, 'store': 2.0,
}

NBC_MIN_WIDTH = {
    'master_bedroom': 2.4, 'bedroom_2': 2.1, 'bedroom_3': 2.1,
    'bedroom_4': 2.1, 'living': 2.4, 'kitchen': 1.8,
    'toilet_attached': 1.2, 'toilet_common': 1.2,
    'verandah': 1.5, 'utility': 1.0, 'dining': 1.8,
}

REQUIRED_CONNECTIONS = [
    ('verandah', 'living', 'archway', 1.20),
    ('living', 'master_bedroom', 'swing', 0.90),
    ('master_bedroom', 'toilet_attached', 'swing', 0.75),
    ('living', 'kitchen', 'swing', 0.90),
    ('kitchen', 'utility', 'swing', 0.75),
    ('living', 'toilet_common', 'swing', 0.75),
    ('living', 'bedroom_2', 'swing', 0.90),
    ('living', 'bedroom_3', 'swing', 0.90),
    ('living', 'bedroom_4', 'swing', 0.90),
    ('living', 'pooja', 'archway', 0.75),
    ('dining', 'living', 'archway', 1.20),
]

PASSAGE_TYPE_MAP = {
    ('verandah', 'living'): 'ARCHWAY_LIVING_VERANDAH',
    ('living', 'master_bedroom'): 'BEDROOM_DOOR',
    ('master_bedroom', 'toilet_attached'): 'TOILET_DOOR',
    ('living', 'kitchen'): 'KITCHEN_DOOR',
    ('kitchen', 'utility'): 'UTILITY_DOOR',
    ('living', 'toilet_common'): 'TOILET_DOOR',
    ('living', 'bedroom_2'): 'BEDROOM_DOOR',
    ('living', 'bedroom_3'): 'BEDROOM_DOOR',
    ('living', 'bedroom_4'): 'BEDROOM_DOOR',
    ('living', 'pooja'): 'ARCHWAY_LIVING_VERANDAH',
    ('dining', 'living'): 'ARCHWAY_LIVING_DINING',
    ('outside', 'verandah'): 'MAIN_ENTRANCE_DOOR',
}

CURRENT_FACING = 'N'

# SECTION 2 — DATA CLASSES
@dataclass
class WallSegment:
    x1: float
    y1: float
    x2: float
    y2: float
    thickness: float
    wall_type: str
    room_left: str
    room_right: str
    has_opening: bool = False

    @property
    def length(self) -> float:
        return math.sqrt((self.x2 - self.x1) ** 2 + (self.y2 - self.y1) ** 2)

    @property
    def midpoint(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def direction(self) -> str:
        return 'H' if abs(self.y2 - self.y1) < 0.01 else 'V'


@dataclass
class DoorOpening:
    label: str
    wall: WallSegment
    position: float
    width: float
    door_type: str
    hinge_side: str
    swing_into: str
    room_from: str
    room_to: str


@dataclass
class WindowOpening:
    label: str
    wall: WallSegment
    position: float
    width: float
    height: float
    sill_height: float
    room_type: str
    is_ventilator: bool = False


@dataclass
class Room:
    room_type: str
    x: float
    y: float
    width: float
    depth: float
    area: float
    cx_pct: float
    cy_pct: float
    compass: str


@dataclass
class FloorPlan:
    plot_w: float
    plot_d: float
    bhk: int
    facing: str
    district: str
    net_w: float
    net_d: float
    setback_front: float
    setback_rear: float
    setback_side: float
    rooms: List = field(default_factory=list)
    walls: List = field(default_factory=list)
    doors: List = field(default_factory=list)
    windows: List = field(default_factory=list)
    score_valid: float = 0.0
    score_vastu: float = 0.0
    score_nbc: float = 0.0
    score_circulation: float = 0.0
    score_adjacency: float = 0.0
    score_overall: float = 0.0
    climate_zone: str = ''
    facing_code: int = 0
    climate_code: int = 0
    explanations: dict = field(default_factory=dict)
    shap_values: dict = field(default_factory=dict)
    materials: List = field(default_factory=list)
    baker_principles: List = field(default_factory=list)
    generation_time_s: float = 0.0
    seed: int = 42


# SECTION 3 — MODEL LOADER (singleton)
class ModelLoader:
    _clf = None
    _dim_model = None
    _explainer = None
    _loaded = False

    @classmethod
    def get(cls):
        if not cls._loaded:
            print('Loading models...', end=' ', flush=True)
            paths = {
                'scorer': os.path.join(MODELS_DIR, 'constraint_scorer.pkl'),
                'dims': os.path.join(MODELS_DIR, 'room_dimensions.h5'),
                'shap': os.path.join(MODELS_DIR, 'shap_explainer.pkl'),
            }
            for _, path in paths.items():
                if not os.path.exists(path):
                    raise FileNotFoundError(f'Missing model: {path}\nDownload from Colab and place in models\\')
            cls._clf = joblib.load(paths['scorer'])
            cls._dim_model = tf.keras.models.load_model(paths['dims'])
            cls._explainer = joblib.load(paths['shap'])
            cls._loaded = True
            print('OK')
        return cls._clf, cls._dim_model, cls._explainer


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _pair(a: str, b: str) -> frozenset:
    return frozenset([a, b])


def _compass(cx_pct: float, cy_pct: float) -> str:
    if cx_pct > 0.6 and cy_pct > 0.6:
        return 'NE'
    if cx_pct < 0.4 and cy_pct > 0.6:
        return 'NW'
    if cx_pct > 0.6 and cy_pct < 0.4:
        return 'SE'
    if cx_pct < 0.4 and cy_pct < 0.4:
        return 'SW'
    if cy_pct > 0.6:
        return 'N'
    if cy_pct < 0.4:
        return 'S'
    if cx_pct > 0.6:
        return 'E'
    if cx_pct < 0.4:
        return 'W'
    return 'C'


def _cardinal_for_wall(wall: WallSegment, net_w: float, net_d: float) -> str:
    tol = 0.05
    if wall.direction == 'H':
        return 'N' if abs(wall.y1 - net_d) < tol else 'S'
    return 'E' if abs(wall.x1 - net_w) < tol else 'W'


# SECTION 4 — DATABASE HELPERS
def get_setbacks(plot_area: float, district: str) -> Tuple[float, float, float]:
    q = '''
        SELECT front_setback_m, rear_setback_m,
               side_setback_left_m, side_setback_right_m
        FROM tn_setbacks
        WHERE plot_area_min_sqm <= ?
          AND (plot_area_max_sqm >= ? OR plot_area_max_sqm IS NULL)
        ORDER BY plot_area_min_sqm DESC LIMIT 1
    '''
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(q, (plot_area, plot_area)).fetchone()
    if not row:
        return (2.0, 1.5, 1.0)
    return (float(row[0]), float(row[1]), round((float(row[2]) + float(row[3])) / 2.0, 2))


def get_climate_zone(district: str) -> str:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute('SELECT climate_zone FROM climate_data WHERE district = ? LIMIT 1', (district,)).fetchone()
    return row[0] if row and row[0] else 'Composite'


def get_window_scores(district: str) -> dict:
    q = '''
        SELECT window_north_score, window_south_score,
               window_east_score, window_west_score
        FROM climate_data WHERE district = ? LIMIT 1
    '''
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(q, (district,)).fetchone()
    if not row:
        return {'N': 1.0, 'S': 0.6, 'E': 0.8, 'W': 0.3}
    return {'N': float(row[0]), 'S': float(row[1]), 'E': float(row[2]), 'W': float(row[3])}


def get_door_width_from_db(passage_type: str) -> float:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute('SELECT min_clear_width_m FROM passage_dimensions WHERE passage_type = ? LIMIT 1', (passage_type,)).fetchone()
    return float(row[0]) if row and row[0] is not None else 0.9


def get_materials(district: str, climate_zone: str) -> List[dict]:
    q = '''
        SELECT material_name, material_category,
               cost_per_unit_inr_avg, unit,
               thermal_performance, sustainability_rating,
               baker_recommended, climate_zone_suitability,
               wall_drawing_color_hex, hatch_pattern
        FROM materials_db
        WHERE (districts_available LIKE ? OR districts_available = 'ALL')
          AND nbc_approved = 1
        ORDER BY sustainability_rating DESC, cost_per_unit_inr_avg ASC
        LIMIT 12
    '''
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(f'%{district}%',))
    return df.to_dict(orient='records')


def get_baker_principles(plot_area: float, climate_zone: str) -> List[dict]:
    q = '''
        SELECT principle_name, category, description,
               cost_saving_pct, drawing_impact, wall_thickness_mm
        FROM baker_principles
        WHERE (plot_area_min_sqm <= ? OR plot_area_min_sqm IS NULL)
          AND (plot_area_max_sqm >= ? OR plot_area_max_sqm IS NULL)
          AND (climate_zones LIKE ? OR climate_zones LIKE '%ALL%')
        ORDER BY cost_saving_pct DESC LIMIT 6
    '''
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(q, conn, params=(plot_area, plot_area, f'%{climate_zone}%'))
    return df.to_dict(orient='records')


# SECTION 5 — DIMENSION MODEL PREDICTION
def predict_room_dims(plot_w, plot_d, bhk, facing_code, climate_code, net_w, net_d, dim_model) -> dict:
    arr = np.array([[plot_w, plot_d, plot_w * plot_d, net_w, net_d, net_w * net_d, bhk, facing_code, climate_code]], dtype=float)
    pred = np.asarray(dim_model.predict(arr, verbose=0)[0], dtype=float)
    raw = {
        'master_bedroom': (pred[0], pred[1]),
        'toilet_attached': (pred[3], pred[4]),
        'living': (pred[6], pred[7]),
        'kitchen': (pred[9], pred[10]),
        'verandah': (pred[12], pred[13]),
        'bedroom_2': (pred[15], pred[16]),
        'bedroom_3': (pred[18], pred[19]),
        'dining': (pred[21], pred[22]),
        'toilet_common': (pred[24], pred[25]),
        'utility': (pred[27], pred[28]),
        'pooja': (pred[30], pred[31]),
        'bedroom_4': (pred[33], pred[34]),
        'store': (pred[36], pred[37]),
    }
    out = {}
    for room, (w, d) in raw.items():
        w = max(float(w), NBC_MIN_WIDTH.get(room, 1.0))
        d = max(float(d), 1.0)
        if room in NBC_MIN_AREA and w * d < NBC_MIN_AREA[room]:
            d = NBC_MIN_AREA[room] / w + 0.1
        out[room] = (round(w, 2), round(d, 2))
    return out


# SECTION 6 — 4-BAND HORIZONTAL ROOM PLACEMENT
def place_rooms_in_bands(net_w, net_d, bhk, predicted_dims, facing) -> List[Room]:
    rooms_n = []
    b1_h = _clamp(predicted_dims.get('verandah', (net_w, 1.8))[1], 1.5, net_d * 0.22)
    b2_h = _clamp(predicted_dims.get('living', (3.5, 2.8))[1], 2.4, net_d * 0.32)
    b3_h = _clamp(predicted_dims.get('kitchen', (2.5, 2.2))[1], 2.0, net_d * 0.28)
    max_top = max(net_d - 2.5, 1.5)
    if b1_h + b2_h + b3_h > max_top:
        s = max_top / (b1_h + b2_h + b3_h)
        b1_h, b2_h, b3_h = b1_h * s, b2_h * s, b3_h * s
    b1_h, b2_h, b3_h = round(b1_h, 2), round(b2_h, 2), round(b3_h, 2)
    b4_h = round(max(net_d - b1_h - b2_h - b3_h, 2.5), 2)
    total = b1_h + b2_h + b3_h + b4_h
    if total > net_d:
        s = net_d / total
        b1_h, b2_h, b3_h = round(b1_h * s, 2), round(b2_h * s, 2), round(b3_h * s, 2)
        b4_h = round(net_d - b1_h - b2_h - b3_h, 2)

    y_b4 = 0.0
    y_b3 = b4_h
    y_b2 = round(b4_h + b3_h, 2)
    y_b1 = round(b4_h + b3_h + b2_h, 2)

    svc_basis = [predicted_dims.get(r, (0.0, 0.0))[0] for r in ('kitchen', 'utility', 'toilet_common') if r in ROOM_LISTS[bhk]]
    svc_w = round(_clamp((max(svc_basis) + 0.1) if svc_basis else net_w * 0.3, net_w * 0.28, net_w * 0.42), 2)
    lv_w = round(net_w - svc_w, 2)

    rooms_n.append(('verandah', 0.0, y_b1, net_w, b1_h))

    pooja_w = 0.0
    if 'pooja' in ROOM_LISTS[bhk]:
        pooja_w = round(min(predicted_dims.get('pooja', (1.2, 1.2))[0], svc_w), 2)
    band2_left = round((net_w - pooja_w) if pooja_w else lv_w, 2)

    living_w = band2_left if 'dining' not in ROOM_LISTS[bhk] else round(max(band2_left * 0.58, NBC_MIN_WIDTH['living']), 2)
    living_w = min(living_w, band2_left)
    rooms_n.append(('living', 0.0, y_b2, living_w, b2_h))

    if 'dining' in ROOM_LISTS[bhk]:
        dining_w = round(max(band2_left - living_w, 0.6), 2)
        rooms_n.append(('dining', living_w, y_b2, dining_w, b2_h))
    if 'pooja' in ROOM_LISTS[bhk]:
        rooms_n.append(('pooja', round(net_w - pooja_w, 2), y_b2, pooja_w, b2_h))

    tcom_w = 0.0
    if 'toilet_common' in ROOM_LISTS[bhk]:
        tcom_w = round(max(predicted_dims.get('toilet_common', (1.4, 1.4))[0], NBC_MIN_WIDTH['toilet_common']), 2)

    kitchen_w = round(max(predicted_dims.get('kitchen', (2.5, 2.2))[0], NBC_MIN_WIDTH['kitchen']), 2) if 'kitchen' in ROOM_LISTS[bhk] else 0.0
    utility_w = round(max(predicted_dims.get('utility', (1.5, 1.2))[0], NBC_MIN_WIDTH['utility']), 2) if 'utility' in ROOM_LISTS[bhk] else 0.0
    store_w = round(max(predicted_dims.get('store', (1.5, 1.2))[0], NBC_MIN_WIDTH.get('store', 0.9)), 2) if 'store' in ROOM_LISTS[bhk] else 0.0

    # Valid training layouts consistently keep the service zone on the east side.
    # For 2BHK, match that cluster more closely with a stacked east-side service column.
    if bhk == 2:
        svc_col_w = round(max(kitchen_w, utility_w, tcom_w, net_w * 0.26), 2)
        svc_col_w = round(_clamp(svc_col_w, net_w * 0.26, net_w * 0.34), 2)
        left_block_w = round(net_w - svc_col_w, 2)

        living_w = round(max(left_block_w * 0.62, NBC_MIN_WIDTH['living']), 2)
        living_w = round(min(living_w, left_block_w), 2)
        for idx, item in enumerate(rooms_n):
            if item[0] == 'living':
                rooms_n[idx] = ('living', 0.0, y_b2, living_w, b2_h)
            elif item[0] == 'dining':
                dining_w = round(max(left_block_w - living_w, 0.6), 2)
                rooms_n[idx] = ('dining', living_w, y_b2, dining_w, b2_h)

        svc_x = round(net_w - svc_col_w, 2)

        if 'kitchen' in ROOM_LISTS[bhk]:
            kitchen_d = round(min(predicted_dims.get('kitchen', (2.5, 2.2))[1], b2_h), 2)
            kitchen_d = round(max(kitchen_d, 2.0), 2)
            kitchen_y = round(y_b1 - kitchen_d, 2)
            rooms_n.append(('kitchen', svc_x, kitchen_y, svc_col_w, kitchen_d))

        y_cursor = y_b3
        if 'toilet_common' in ROOM_LISTS[bhk]:
            tcom_d = round(min(predicted_dims.get('toilet_common', (1.4, 1.4))[1], b3_h * 0.55), 2)
            tcom_d = round(max(tcom_d, 1.4), 2)
            rooms_n.append(('toilet_common', svc_x, y_cursor, svc_col_w, tcom_d))
            y_cursor = round(y_cursor + tcom_d, 2)

        if 'utility' in ROOM_LISTS[bhk]:
            util_d = round(max(predicted_dims.get('utility', (1.5, 1.2))[1], 1.2), 2)
            util_d = round(min(util_d, max(net_d - y_cursor, 0.6)), 2)
            rooms_n.append(('utility', svc_x, y_cursor, svc_col_w, util_d))

    else:
        gap = 0.20 if 'toilet_common' in ROOM_LISTS[bhk] else 0.0
        if 'toilet_common' in ROOM_LISTS[bhk]:
            tcom_w = round(min(predicted_dims.get('toilet_common', (1.4, 1.4))[0], max(net_w * 0.22, 1.2)), 2)
            rooms_n.append(('toilet_common', 0.0, y_b3, tcom_w, b3_h))

        service_rooms = [r for r in ('kitchen', 'utility', 'store') if r in ROOM_LISTS[bhk]]
        widths = {r: round(max(predicted_dims.get(r, (1.5, 1.5))[0], NBC_MIN_WIDTH.get(r, 0.9)), 2) for r in service_rooms}
        total_svc = sum(widths.values())
        svc_start = round(max(net_w - total_svc, tcom_w + gap), 2)
        x = svc_start
        for r in service_rooms:
            w = round(min(widths[r], max(net_w - x, 0.6)), 2)
            rooms_n.append((r, x, y_b3, w, b3_h))
            x = round(x + w, 2)


    # master_bedroom: SW corner, depth from h5 model prediction
    mb_w = round(max(
        min(predicted_dims.get('master_bedroom', (3.2, 3.0))[0],
            net_w * 0.45),
        NBC_MIN_WIDTH['master_bedroom']), 2)
    mb_d_pred = round(predicted_dims.get(
        'master_bedroom', (3.2, 2.8))[1], 2)
    # toilet_attached: depth from h5 model prediction
    ta_w = 0.0
    ta_d_pred = 0.0
    if 'toilet_attached' in ROOM_LISTS[bhk]:
        ta_w = round(max(
            min(predicted_dims.get('toilet_attached', (1.5, 1.5))[0],
                mb_w),
            NBC_MIN_WIDTH['toilet_attached']), 2)
        ta_d_pred = round(predicted_dims.get(
            'toilet_attached', (1.5, 1.5))[1], 2)
    # Scale both to fit b4_h using predicted proportions (no hardcoding)
    required_d = mb_d_pred + ta_d_pred
    if required_d > b4_h and required_d > 0:
        scale = b4_h / required_d
        mb_d_actual = round(max(mb_d_pred * scale,
            NBC_MIN_AREA['master_bedroom'] / mb_w), 2)
        ta_d_actual = round(max(ta_d_pred * scale, 1.2), 2)
    else:
        mb_d_actual = round(max(mb_d_pred,
            NBC_MIN_AREA['master_bedroom'] / mb_w), 2)
        mb_d_actual = round(min(mb_d_actual, b4_h - 1.2), 2)
        ta_d_actual = round(max(ta_d_pred, 1.2), 2)
        ta_d_actual = round(min(ta_d_actual, b4_h - mb_d_actual), 2)
    # Place master_bedroom at SW (x=0, y=0)
    rooms_n.append(('master_bedroom', 0.0, 0.0, mb_w, mb_d_actual))
    # Place toilet_attached ABOVE master_bedroom (shared horizontal wall)
    # This satisfies adjacency_rules MUST_SHARE_WALL constraint from DB
    if 'toilet_attached' in ROOM_LISTS[bhk]:
        rooms_n.append((
            'toilet_attached', 0.0, mb_d_actual,
            ta_w, ta_d_actual))

    extra = [r for r in ('bedroom_2', 'bedroom_3', 'bedroom_4')
             if r in ROOM_LISTS[bhk]]
    # Use h5 model predicted width for each bedroom
    pred_w = {r: round(max(
                  predicted_dims.get(r, (2.5, 2.5))[0],
                  NBC_MIN_WIDTH.get(r, 2.1)), 2)
              for r in extra}
    available = round(net_w - mb_w, 2)
    total_pred = sum(pred_w.values())
    # Scale proportionally if predicted widths exceed available space
    if total_pred > available and total_pred > 0:
        scale = available / total_pred
        pred_w = {r: round(max(w * scale,
                               NBC_MIN_WIDTH.get(r, 2.1)), 2)
                  for r, w in pred_w.items()}
    x = round(mb_w, 2)
    for r in extra:
        w = round(min(pred_w.get(r, 2.1), max(net_w - x, 0.6)), 2)
        rooms_n.append((r, x, 0.0, w, b4_h))
        x = round(x + w, 2)

    out = []
    for rt, x, y, w, d in rooms_n:
        if facing == 'S':
            nx, ny, nw, nd = x, net_d - y - d, w, d
        elif facing == 'E':
            x0, y0, w0, d0 = x / net_w, y / net_d, w / net_w, d / net_d
            nx, ny, nw, nd = net_w * (1 - (y0 + d0)), net_d * x0, net_w * d0, net_d * w0
        elif facing == 'W':
            x0, y0, w0, d0 = x / net_w, y / net_d, w / net_w, d / net_d
            nx, ny, nw, nd = net_w * y0, net_d * (1 - (x0 + w0)), net_w * d0, net_d * w0
        else:
            nx, ny, nw, nd = x, y, w, d
        nx, ny = round(_clamp(nx, 0.0, max(net_w - nw, 0.0)), 2), round(_clamp(ny, 0.0, max(net_d - nd, 0.0)), 2)
        nw, nd = round(max(nw, 0.6), 2), round(max(nd, 0.6), 2)
        area = round(nw * nd, 2)
        cx_pct = round((nx + nw / 2) / net_w, 3)
        cy_pct = round((ny + nd / 2) / net_d, 3)
        out.append(Room(rt, nx, ny, nw, nd, area, cx_pct, cy_pct, _compass(cx_pct, cy_pct)))
    return out


# SECTION 7 — WALL NETWORK BUILDER
def build_wall_network(rooms: List[Room], net_w: float, net_d: float) -> List[WallSegment]:
    tol = 0.05
    walls, seen = [], set()
    for i, a in enumerate(rooms):
        for b in rooms[i + 1:]:
            if abs((a.x + a.width) - b.x) < tol or abs((b.x + b.width) - a.x) < tol:
                y1, y2 = max(a.y, b.y), min(a.y + a.depth, b.y + b.depth)
                if y2 - y1 > tol:
                    x = round(b.x if abs((a.x + a.width) - b.x) < tol else a.x, 2)
                    west = a.room_type if (a.x + a.width) <= (b.x + tol) else b.room_type
                    east = b.room_type if west == a.room_type else a.room_type
                    key = ('V', x, round(y1, 2), round(y2, 2))
                    if key not in seen:
                        walls.append(WallSegment(x, round(y1, 2), x, round(y2, 2), WALL_INTERIOR, 'interior', west, east, False))
                        seen.add(key)
            if abs((a.y + a.depth) - b.y) < tol or abs((b.y + b.depth) - a.y) < tol:
                x1, x2 = max(a.x, b.x), min(a.x + a.width, b.x + b.width)
                if x2 - x1 > tol:
                    y = round(b.y if abs((a.y + a.depth) - b.y) < tol else a.y, 2)
                    south = a.room_type if (a.y + a.depth) <= (b.y + tol) else b.room_type
                    north = b.room_type if south == a.room_type else a.room_type
                    key = ('H', y, round(x1, 2), round(x2, 2))
                    if key not in seen:
                        walls.append(WallSegment(round(x1, 2), y, round(x2, 2), y, WALL_INTERIOR, 'interior', south, north, False))
                        seen.add(key)
    for r in rooms:
        edges = [
            ('N', r.x, r.y + r.depth, r.x + r.width, r.y + r.depth),
            ('S', r.x, r.y, r.x + r.width, r.y),
            ('W', r.x, r.y, r.x, r.y + r.depth),
            ('E', r.x + r.width, r.y, r.x + r.width, r.y + r.depth),
        ]
        for _, x1, y1, x2, y2 in edges:
            exterior = (abs(x1) < tol and abs(x2) < tol) or (abs(x1 - net_w) < tol and abs(x2 - net_w) < tol) or (abs(y1) < tol and abs(y2) < tol) or (abs(y1 - net_d) < tol and abs(y2 - net_d) < tol)
            if exterior:
                walls.append(WallSegment(round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2), WALL_EXTERIOR, 'exterior', r.room_type, 'outside', False))
    return walls


# SECTION 8 — DOOR PLACEMENT
def place_doors(rooms: List[Room], walls: List[WallSegment], bhk: int) -> List[DoorOpening]:
    room_map = {r.room_type: r for r in rooms}
    doors = []
    vwalls = [w for w in walls if w.wall_type == 'exterior' and w.room_left == 'verandah']
    if vwalls:
        if CURRENT_FACING == 'N':
            wall = max(vwalls, key=lambda w: w.midpoint[1])
        elif CURRENT_FACING == 'S':
            wall = min(vwalls, key=lambda w: w.midpoint[1])
        elif CURRENT_FACING == 'E':
            wall = max(vwalls, key=lambda w: w.midpoint[0])
        else:
            wall = min(vwalls, key=lambda w: w.midpoint[0])
        wall.has_opening = True
        doors.append(DoorOpening('MAIN ENTRANCE', wall, 0.5, round(get_door_width_from_db('MAIN_ENTRANCE_DOOR'), 2), 'swing', 'left', 'verandah', 'outside', 'verandah'))
    counter = 1
    for from_r, to_r, door_type, fallback in REQUIRED_CONNECTIONS:
        if from_r not in room_map or to_r not in room_map:
            continue
        matches = [w for w in walls if _pair(w.room_left, w.room_right) == _pair(from_r, to_r)]
        if not matches:
            continue
        wall = max(matches, key=lambda w: w.length)
        ptype = PASSAGE_TYPE_MAP.get((from_r, to_r), PASSAGE_TYPE_MAP.get((to_r, from_r), 'BEDROOM_DOOR'))
        width = get_door_width_from_db(ptype)
        if width == 0.9 and door_type == 'archway':
            width = 1.2
        if wall.length < width + 0.4:
            width = max(round(wall.length * 0.55, 2), 0.6)
        wall.has_opening = True
        doors.append(DoorOpening(f'D{counter}', wall, 0.5, round(width if width else fallback, 2), door_type, 'left', to_r, from_r, to_r))
        counter += 1
    return doors


# SECTION 9 — WINDOW PLACEMENT
def place_windows(rooms: List[Room], walls: List[WallSegment], window_scores: dict, facing: str) -> List[WindowOpening]:
    net_w = max(max(w.x1, w.x2) for w in walls) if walls else 0.0
    net_d = max(max(w.y1, w.y2) for w in walls) if walls else 0.0
    by_room = {}
    for w in walls:
        if w.wall_type == 'exterior':
            by_room.setdefault(w.room_left, []).append(w)
    habitable = {'master_bedroom', 'bedroom_2', 'bedroom_3', 'bedroom_4', 'living', 'dining', 'kitchen'}
    wet = {'toilet_attached', 'toilet_common', 'utility'}
    wins, counter = [], 1
    for r in rooms:
        rws = by_room.get(r.room_type, [])
        if not rws:
            continue
        if r.room_type in habitable:
            scored = sorted([(window_scores.get(_cardinal_for_wall(w, net_w, net_d), 0.5), w) for w in rws], key=lambda t: t[0], reverse=True)
            placed = 0
            for _, w in scored:
                if w.has_opening:
                    continue
                if placed == 0:
                    ww = max(min(w.length * 0.50, 1.50), 0.60)
                    wins.append(WindowOpening(f'W{counter}', w, 0.50, round(ww, 2), 1.20, 0.90, r.room_type, False))
                    counter += 1
                    placed += 1
                elif r.room_type.startswith('bedroom') or r.room_type == 'master_bedroom':
                    if w.length >= 0.8:
                        ww = max(min(w.length * 0.40, 1.05), 0.45)
                        wins.append(WindowOpening(f'W{counter}', w, 0.50, round(ww, 2), 1.20, 0.90, r.room_type, False))
                        counter += 1
                    break
        elif r.room_type in wet:
            for w in rws:
                if not w.has_opening:
                    wins.append(WindowOpening(f'W{counter}', w, 0.50, 0.45, 0.45, 1.50, r.room_type, True))
                    counter += 1
                    break
    return wins


# SECTION 10 — FEATURE VECTOR BUILDER
def build_feature_vector(fp: FloorPlan) -> pd.DataFrame:
    """
    74-column vector in EXACT order from clf.feature_names_in_
    Order verified by reading model directly.
    """
    room_map = {r.room_type: r for r in fp.rooms}

    # Exact order from clf.feature_names_in_ (positions 9-73)
    ROOM_ORDER = [
        ("master_bedroom",  "masterbedr"),   # 9-13
        ("toilet_attached", "toiletatta"),   # 14-18
        ("living",          "living"),       # 19-23
        ("kitchen",         "kitchen"),      # 24-28
        ("verandah",        "verandah"),     # 29-33
        ("bedroom_2",       "bedroom2"),     # 34-38
        ("bedroom_3",       "bedroom3"),     # 39-43
        ("dining",          "dining"),       # 44-48
        ("toilet_common",   "toiletcomm"),   # 49-53
        ("utility",         "utility"),      # 54-58
        ("pooja",           "pooja"),        # 59-63
        ("bedroom_4",       "bedroom4"),     # 64-68
        ("store",           "store"),        # 69-73
    ]

    row = {
        "plot_w":       fp.plot_w,
        "plot_d":       fp.plot_d,
        "plot_area":    round(fp.plot_w * fp.plot_d, 2),
        "net_w":        fp.net_w,
        "net_d":        fp.net_d,
        "net_area":     round(fp.net_w * fp.net_d, 2),
        "bhk":          fp.bhk,
        "facing_code":  fp.facing_code,
        "climate_code": fp.climate_code,
    }

    for room_type, prefix in ROOM_ORDER:
        if room_type in room_map:
            r = room_map[room_type]
            row[f"{prefix}_w"]      = round(r.width,  2)
            row[f"{prefix}_d"]      = round(r.depth,  2)
            row[f"{prefix}_area"]   = round(r.area,   2)
            row[f"{prefix}_cx_pct"] = round(r.cx_pct, 3)
            row[f"{prefix}_cy_pct"] = round(r.cy_pct, 3)
        else:
            row[f"{prefix}_w"]      = -1.0
            row[f"{prefix}_d"]      = -1.0
            row[f"{prefix}_area"]   = -1.0
            row[f"{prefix}_cx_pct"] = -1.0
            row[f"{prefix}_cy_pct"] = -1.0

    return pd.DataFrame([row])

def score_and_explain(fp: FloorPlan, clf, explainer) -> FloorPlan:
    X = build_feature_vector(fp)
    fp.score_valid = round(float(clf.predict_proba(X)[0][1]), 3)
    room_map = {r.room_type: r for r in fp.rooms}

    sv = 1.0
    if 'kitchen' in room_map:
        k = room_map['kitchen']
        if k.cx_pct < 0.35 and k.cy_pct < 0.35:
            sv -= 0.40
    if 'master_bedroom' in room_map:
        mb = room_map['master_bedroom']
        if mb.cx_pct > 0.65 and mb.cy_pct > 0.65:
            sv -= 0.40
    if 'pooja' in room_map:
        p = room_map['pooja']
        if not (p.cx_pct > 0.55 and p.cy_pct > 0.55):
            sv -= 0.20

    sn = 1.0
    n_rooms = len([r for r in fp.rooms if r.room_type in NBC_MIN_AREA])
    if n_rooms:
        for r in fp.rooms:
            if r.room_type in NBC_MIN_AREA and r.area < NBC_MIN_AREA[r.room_type] * 0.88:
                sn -= (1.0 / n_rooms)

    sc = 1.0
    pairs = [_pair(w.room_left, w.room_right) for w in fp.walls]
    if _pair('living', 'verandah') not in pairs:
        sc -= 0.35
    if _pair('master_bedroom', 'toilet_attached') not in pairs:
        sc -= 0.35
    if len(fp.doors) == 0:
        sc -= 0.30

    sa = 1.0
    penalties = {
        frozenset(['kitchen', 'master_bedroom']): 0.20,
        frozenset(['kitchen', 'bedroom_2']): 0.20,
        frozenset(['kitchen', 'bedroom_3']): 0.20,
        frozenset(['toilet_attached', 'kitchen']): 0.20,
        frozenset(['toilet_common', 'kitchen']): 0.20,
        frozenset(['toilet_attached', 'dining']): 0.15,
        frozenset(['toilet_common', 'dining']): 0.15,
    }
    seen = set()
    for w in fp.walls:
        if w.has_opening:
            continue
        pair = frozenset([w.room_left, w.room_right])
        if pair in penalties and pair not in seen:
            sa -= penalties[pair]
            seen.add(pair)

    clamp = lambda x: round(max(0.0, min(1.0, x)), 3)
    fp.score_vastu = clamp(sv)
    fp.score_nbc = clamp(sn)
    fp.score_circulation = clamp(sc)
    fp.score_adjacency = clamp(sa)
    fp.score_overall = clamp(0.30 * fp.score_vastu + 0.25 * fp.score_nbc + 0.25 * fp.score_circulation + 0.20 * fp.score_adjacency)

    try:
        shap_vals = explainer.shap_values(X)
        arr = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
        idxs = np.argsort(np.abs(arr))[::-1][:5]
        fp.shap_values = {X.columns[i]: round(float(arr[i]), 4) for i in idxs}
    except Exception:
        fp.shap_values = {}

    VASTU_EXPLAIN = {
        'NE': 'NE corner - vastu Ishanya zone, auspicious.',
        'SW': 'SW corner - vastu Nairuthi zone, stable.',
        'NW': 'NW corner - vastu Vayavya zone, movement.',
        'SE': 'SE corner - vastu Agni zone, fire element.',
        'N': 'North side - faces road, public zone.',
        'S': 'South side - private, away from road.',
        'E': 'East side - morning light, positive.',
        'W': 'West side - afternoon, service zone.',
        'C': 'Central position - core circulation.',
    }
    for r in fp.rooms:
        ok = r.area >= NBC_MIN_AREA.get(r.room_type, 0) * 0.88
        fp.explanations[r.room_type] = f"{r.room_type.replace('_', ' ').title()} ({r.width:.1f}m x {r.depth:.1f}m, {r.area:.1f}m²) - {VASTU_EXPLAIN.get(r.compass, '')} Area {'meets NBC' if ok else 'below NBC min'}."
    fp.explanations['overall'] = (
        f"Plan scores {fp.score_overall:.0%} overall. Vastu {fp.score_vastu:.0%}, NBC {fp.score_nbc:.0%}, Circulation {fp.score_circulation:.0%}. "
        f"{'All critical circulation paths connected.' if fp.score_circulation > 0.7 else 'Some circulation paths incomplete.'}"
    )
    return fp


# SECTION 12 — MAIN GENERATE FUNCTION + SELF TEST
def generate(params: dict) -> FloorPlan:
    t0 = time.time()
    clf, dim_model, explainer = ModelLoader.get()
    plot_w, plot_d = float(params['plot_w']), float(params['plot_d'])
    bhk, facing = int(params['bhk']), str(params['facing']).upper()
    district, seed = str(params['district']), int(params.get('seed', 42))
    plot_area = plot_w * plot_d
    front, rear, side = get_setbacks(plot_area, district)
    climate_zone = get_climate_zone(district)
    window_scores = get_window_scores(district)
    materials = get_materials(district, climate_zone)
    baker_principles = get_baker_principles(plot_area, climate_zone)
    net_w = round(max(plot_w - 2 * side, 3.0), 2)
    net_d = round(max(plot_d - front - rear, 3.0), 2)
    fp = FloorPlan(plot_w, plot_d, bhk, facing, district, net_w, net_d, front, rear, side, climate_zone=climate_zone, facing_code=FACING_MAP.get(facing, 0), climate_code=CLIMATE_MAP.get(climate_zone, 2), materials=materials, baker_principles=baker_principles, seed=seed)
    dims = predict_room_dims(plot_w, plot_d, bhk, fp.facing_code, fp.climate_code, net_w, net_d, dim_model)
    global CURRENT_FACING
    CURRENT_FACING = facing
    fp.rooms = place_rooms_in_bands(net_w, net_d, bhk, dims, facing)
    fp.walls = build_wall_network(fp.rooms, net_w, net_d)
    fp.doors = place_doors(fp.rooms, fp.walls, bhk)
    fp.windows = place_windows(fp.rooms, fp.walls, window_scores, facing)
    fp = score_and_explain(fp, clf, explainer)
    fp.generation_time_s = round(time.time() - t0, 3)
    return fp


if __name__ == '__main__':
    test_cases = [
        {'plot_w': 12, 'plot_d': 15, 'bhk': 2, 'facing': 'N', 'district': 'Coimbatore', 'seed': 42},
        {'plot_w': 9, 'plot_d': 12, 'bhk': 2, 'facing': 'S', 'district': 'Chennai', 'seed': 42},
        {'plot_w': 15, 'plot_d': 20, 'bhk': 3, 'facing': 'E', 'district': 'Madurai', 'seed': 42},
        {'plot_w': 20, 'plot_d': 25, 'bhk': 4, 'facing': 'W', 'district': 'Salem', 'seed': 42},
    ]
    for i, params in enumerate(test_cases):
        print(f"\n{'=' * 60}")
        print(f"Test {i+1}: {params['plot_w']}x{params['plot_d']}m  {params['bhk']}BHK  {params['facing']}-facing  {params['district']}")
        fp = generate(params)
        print(f"  Net area:  {fp.net_w}x{fp.net_d}m")
        print(f"  Setbacks:  front={fp.setback_front}m rear={fp.setback_rear}m side={fp.setback_side}m")
        print(f"  Climate:   {fp.climate_zone}")
        print(f"  Rooms:     {len(fp.rooms)}")
        print(f"  Walls:     {len(fp.walls)} segments ({sum(1 for w in fp.walls if w.wall_type=='exterior')} ext, {sum(1 for w in fp.walls if w.wall_type=='interior')} int)")
        print(f"  Doors:     {len(fp.doors)}")
        print(f"  Windows:   {len(fp.windows)}")
        print(f"  Scores:    valid={fp.score_valid:.2f}  vastu={fp.score_vastu:.2f}  nbc={fp.score_nbc:.2f}  circulation={fp.score_circulation:.2f}  overall={fp.score_overall:.2f}")
        print(f"  Time:      {fp.generation_time_s}s")
        print(f"\n  ROOMS:")
        for r in fp.rooms:
            nbc = NBC_MIN_AREA.get(r.room_type, 0)
            ok = '✓' if r.area >= nbc * 0.88 else '✗'
            print(f"    {r.room_type:<22} x={r.x:.1f} y={r.y:.1f} w={r.width:.1f} d={r.depth:.1f} area={r.area:.1f}m² {ok} compass={r.compass}")
        print(f"\n  WALLS:")
        for w in fp.walls:
            print(f"    [{w.wall_type[:3].upper()}] ({w.x1:.1f},{w.y1:.1f})-({w.x2:.1f},{w.y2:.1f}) len={w.length:.2f}m t={w.thickness*1000:.0f}mm {'[OPENING]' if w.has_opening else ''}")
        print(f"\n  DOORS:")
        for d in fp.doors:
            print(f"    {d.label:<14} {d.room_from} → {d.room_to}  type={d.door_type}  width={d.width}m")
        print(f"\n  WINDOWS:")
        for w in fp.windows:
            kind = 'ventilator' if w.is_ventilator else 'window'
            print(f"    {w.label:<6} {w.room_type:<22} width={w.width:.2f}m  sill={w.sill_height}m  [{kind}]")
        print(f"\n  OVERALL: {fp.explanations.get('overall', '')}")
        print(f"  Materials: {len(fp.materials)} recommended for {params['district']}")
        print(f"  Baker principles: {len(fp.baker_principles)} applicable")

