import os
import sqlite3
import time

import numpy as np
import pandas as pd

DB_PATH = "db/floorplan.db"
OUT_PATH = "training_data/floor_plan_samples.parquet"
N_SAMPLES = 50000
SEED = 42

PLOT_SIZES = [
    (3, 6), (4, 6), (4, 8), (5, 8), (5, 10), (6, 9), (6, 12),
    (7, 10), (8, 10), (8, 12), (9, 12), (9, 15), (10, 12),
    (10, 15), (12, 15), (12, 18), (12, 20), (15, 20),
    (15, 24), (18, 24), (20, 25), (20, 30)
]

BHK_TYPES = [1, 2, 3, 4]
FACINGS = ["N", "S", "E", "W"]
FACING_MAP = {"N": 0, "S": 1, "E": 2, "W": 3}
CLIMATE_MAP = {"Hot_Humid": 0, "Hot_Dry": 1, "Composite": 2, "Warm_Humid": 3}

ROOM_LISTS = {
    1: ["master_bedroom", "toilet_attached", "living",
        "kitchen", "verandah"],
    2: ["master_bedroom", "toilet_attached", "bedroom_2",
        "living", "dining", "kitchen", "toilet_common",
        "utility", "verandah"],
    3: ["master_bedroom", "toilet_attached", "bedroom_2",
        "bedroom_3", "living", "dining", "kitchen",
        "toilet_common", "utility", "verandah", "pooja"],
    4: ["master_bedroom", "toilet_attached", "bedroom_2",
        "bedroom_3", "bedroom_4", "living", "dining", "kitchen",
        "toilet_common", "utility", "verandah", "pooja", "store"],
}

NBC_MIN_AREA = {'master_bedroom': 9.5, 'bedroom_2': 7.5, 'bedroom_3': 7.5, 'bedroom_4': 7.5, 'living': 9.5, 'dining': 5.0, 'kitchen': 4.5, 'toilet_attached': 1.8, 'toilet_common': 2.5, 'utility': 2.0, 'verandah': 4.0, 'pooja': 1.2, 'store': 2.0}

NBC_MIN_WIDTH = {'master_bedroom': 2.4, 'bedroom_2': 2.1, 'bedroom_3': 2.1, 'bedroom_4': 2.1, 'living': 2.4, 'kitchen': 1.8, 'toilet_attached': 1.2, 'toilet_common': 1.2, 'verandah': 1.5, 'utility': 1.0, 'dining': 1.8, 'lobby': 1.8, 'main_entrance': 0.9}

FORBIDDEN_ADJ_PAIRS = {('pooja', 'toilet_common'), ('dining', 'toilet_attached'), ('verandah', 'toilet_attached'), ('toilet_attached', 'kitchen'), ('toilet_attached', 'courtyard'), ('toilet_attached', 'living'), ('living', 'toilet_attached'), ('toilet_common', 'pooja'), ('courtyard', 'toilet_common'), ('kitchen', 'staircase'), ('kitchen', 'master_bedroom'), ('kitchen', 'toilet_common'), ('toilet_common', 'living'), ('pooja', 'toilet_attached'), ('toilet_common', 'kitchen'), ('toilet_common', 'courtyard'), ('staircase', 'toilet_common'), ('toilet_attached', 'dining'), ('toilet_attached', 'staircase'), ('verandah', 'toilet_common'), ('dining', 'toilet_common'), ('toilet_attached', 'verandah'), ('living', 'toilet_common'), ('master_bedroom', 'kitchen'), ('courtyard', 'toilet_attached'), ('toilet_common', 'dining'), ('kitchen', 'toilet_attached'), ('toilet_attached', 'pooja'), ('toilet_common', 'staircase'), ('toilet_common', 'verandah'), ('staircase', 'toilet_attached'), ('staircase', 'kitchen')}

MUST_SHARE_WALL_PAIRS = {('dining', 'kitchen'), ('kitchen', 'utility'), ('passage', 'living'), ('dining', 'living'), ('verandah', 'living'), ('passage', 'bedroom_4'), ('living', 'courtyard'), ('utility', 'kitchen'), ('kitchen', 'dining'), ('bedroom_3', 'passage'), ('staircase', 'passage'), ('pooja', 'living'), ('living', 'dining'), ('passage', 'staircase'), ('living', 'passage'), ('toilet_attached', 'master_bedroom'), ('living', 'staircase'), ('bedroom_2', 'passage'), ('passage', 'toilet_common'), ('living', 'verandah'), ('master_bedroom', 'toilet_attached'), ('bedroom_4', 'passage'), ('courtyard', 'living'), ('passage', 'bedroom_3'), ('toilet_common', 'passage'), ('living', 'pooja'), ('passage', 'bedroom_2'), ('staircase', 'living')}

HARDCODED_DEFAULTS = {
    "master_bedroom": (3.0, 2.8),
    "bedroom_2": (2.7, 2.5),
    "bedroom_3": (2.4, 2.2),
    "bedroom_4": (2.4, 2.2),
    "living": (3.5, 3.0),
    "dining": (2.5, 2.0),
    "kitchen": (2.5, 2.2),
    "toilet_attached": (1.5, 1.5),
    "toilet_common": (1.4, 1.4),
    "utility": (1.5, 1.2),
    "verandah": (0.0, 1.8),
    "pooja": (1.2, 1.2),
    "store": (1.5, 1.2),
}


db_uri = f"file:{os.path.abspath(DB_PATH).replace(os.sep, '/')}?mode=ro"
with sqlite3.connect(db_uri, uri=True) as conn:
    setbacks_df = pd.read_sql_query("SELECT * FROM tn_setbacks", conn)
    climate_df = pd.read_sql_query(
        "SELECT district, climate_zone FROM climate_data",
        conn,
    )
    plot_confs_df = pd.read_sql_query("SELECT * FROM plot_configurations", conn)
    move_paths_df = pd.read_sql_query("SELECT * FROM movement_paths", conn)

DISTRICT_CLIMATE = (
    climate_df.dropna(subset=["district"])
    .drop_duplicates(subset=["district"], keep="first")
    .set_index("district")["climate_zone"]
    .to_dict()
)
districts = sorted(DISTRICT_CLIMATE.keys())

CRITICAL_PAIRS = {
    (row["from_space"], row["to_space"])
    for _, row in move_paths_df.loc[move_paths_df["is_critical_path"] == 1].iterrows()
}

FORBIDDEN_PAIRS = {
    (row["from_space"], row["to_space"])
    for _, row in move_paths_df.loc[move_paths_df["path_type"] == "FORBIDDEN"].iterrows()
}

print(f"Districts loaded: {len(districts)}")
print(f"Critical path pairs: {len(CRITICAL_PAIRS)}")
print(f"Forbidden path pairs: {len(FORBIDDEN_PAIRS)}")


def get_setbacks(plot_area, facing, setbacks_df):
    matches = setbacks_df[
        (setbacks_df["plot_area_min_sqm"] <= plot_area)
        & (
            setbacks_df["plot_area_max_sqm"].isna()
            | (setbacks_df["plot_area_max_sqm"] >= plot_area)
        )
    ]

    if matches.empty:
        return 2.0, 1.5, 1.0

    row = matches.sort_values("plot_area_min_sqm", ascending=False).iloc[0]
    front = float(row.get("front_setback_m", 2.0))
    rear = float(row.get("rear_setback_m", 1.5))
    left = row.get("side_setback_left_m", 1.0)
    right = row.get("side_setback_right_m", 1.0)
    if pd.isna(left):
        left = 1.0
    if pd.isna(right):
        right = 1.0
    side = float((float(left) + float(right)) / 2.0)
    return front, rear, side


def get_room_targets(plot_w, plot_d, bhk, plot_confs_df):
    matches = plot_confs_df[
        (plot_confs_df["plot_width_m"].sub(plot_w).abs() <= 1.5)
        & (plot_confs_df["plot_depth_m"].sub(plot_d).abs() <= 2.0)
        & (plot_confs_df["bhk_type"] == f"{bhk}BHK")
    ]

    room_column_map = {
        "master_bedroom": ("master_bedroom_target_w", "master_bedroom_target_d"),
        "bedroom_2": ("bedroom2_target_w", "bedroom2_target_d"),
        "bedroom_3": ("bedroom3_target_w", "bedroom3_target_d"),
        "living": ("living_target_w", "living_target_d"),
        "dining": ("dining_target_w", "dining_target_d"),
        "kitchen": ("kitchen_target_w", "kitchen_target_d"),
        "toilet_attached": ("toilet_att_target_w", "toilet_att_target_d"),
        "toilet_common": ("toilet_common_target_w", "toilet_common_target_d"),
        "utility": ("utility_target_w", "utility_target_d"),
        "verandah": ("verandah_target_w", "verandah_target_d"),
    }

    matched_row = matches.iloc[0] if not matches.empty else None
    targets = {}

    for room in ROOM_LISTS.get(bhk, []):
        default_w, default_d = HARDCODED_DEFAULTS[room]
        target_w = default_w
        target_d = default_d

        if matched_row is not None and room in room_column_map:
            width_col, depth_col = room_column_map[room]
            if width_col in matched_row.index and not pd.isna(matched_row[width_col]):
                target_w = float(matched_row[width_col])
            if depth_col in matched_row.index and not pd.isna(matched_row[depth_col]):
                target_d = float(matched_row[depth_col])

        if room == "verandah" and (pd.isna(target_w) or target_w == 0):
            target_w = plot_w * 0.6

        targets[room] = (float(target_w), float(target_d))

    return targets


def place_rooms(net_w, net_d, bhk, targets, rng, error_prob=0.40):
    """
    Zone-based placement. Computes zone sizes from room dims.
    Guarantees no overlap in base case (error_type=None).
    """
    rooms_needed = ROOM_LISTS.get(bhk, [])

    # ── STEP 1: Room dimensions with NBC hard floor ───────────────
    dims = {}
    for room in rooms_needed:
        tw, td = targets.get(room, HARDCODED_DEFAULTS.get(room, (2.0, 2.0)))
        w = float(tw) * rng.uniform(0.92, 1.08)
        d = float(td) * rng.uniform(0.92, 1.08)
        w = max(w, NBC_MIN_WIDTH.get(room, 1.0))
        min_area = NBC_MIN_AREA.get(room, 1.0)
        if w * d < min_area:
            d = min_area / w + 0.05
        dims[room] = (round(w, 2), round(d, 2))

    # ── STEP 2: Reduce room list for small plots ──────────────────
    net_area = net_w * net_d
    if net_area < 45.0:
        # Micro plot: merge living+dining, drop utility/pooja/store
        rooms_needed = [r for r in rooms_needed
                        if r not in ('dining', 'utility', 'pooja', 'store')]
        if 'living' in dims:
            lw, ld = dims['living']
            dims['living'] = (round(lw, 2), round(min(ld + 0.5, net_d * 0.3), 2))
    elif net_area < 80.0:
        rooms_needed = [r for r in rooms_needed
                        if r not in ('pooja', 'store')]

    # ── STEP 3: Assign rooms to zones ────────────────────────────
    PUBLIC  = ['verandah', 'living', 'dining']
    SERVICE = ['kitchen', 'utility', 'toilet_common', 'pooja', 'store']
    PRIVATE = ['master_bedroom', 'toilet_attached',
               'bedroom_2', 'bedroom_3', 'bedroom_4']

    pub_rooms = [r for r in PUBLIC  if r in rooms_needed]
    svc_rooms = [r for r in SERVICE if r in rooms_needed]
    prv_rooms = [r for r in PRIVATE if r in rooms_needed]

    # ── STEP 4: Compute zone sizes from room requirements ─────────
    # PUBLIC: height = verandah depth + living depth
    pub_d = sum(dims[r][1] for r in pub_rooms if r in ('verandah', 'living'))
    pub_d = round(max(pub_d, net_d * 0.25), 2)
    pub_d = min(pub_d, net_d * 0.40)

    # PRIVATE: needs to fit master_bed + toilet + all extra bedrooms
    prv_height_needed = 0.0
    if 'master_bedroom' in dims:
        prv_height_needed += dims['master_bedroom'][1]
    if 'toilet_attached' in dims:
        prv_height_needed += dims['toilet_attached'][1]
    extra_beds = [r for r in prv_rooms
                  if r not in ('master_bedroom', 'toilet_attached')]
    if extra_beds:
        max_bed_d = max(dims[r][1] for r in extra_beds)
        prv_height_needed += max_bed_d + 0.3
    prv_height_needed = round(prv_height_needed, 2)

    # SERVICE: width = widest service room + small margin
    svc_w = 0.0
    if svc_rooms:
        svc_w = max(dims[r][0] for r in svc_rooms) + 0.1
    svc_w = round(max(svc_w, net_w * 0.28), 2)
    svc_w = min(svc_w, net_w * 0.45)

    # Available dimensions after zone allocation
    prv_w = round(net_w - svc_w, 2)
    bottom_zone_d = round(net_d - pub_d, 2)

    # Scale down all rooms if private zone height is insufficient
    if prv_height_needed > bottom_zone_d * 0.95 and bottom_zone_d > 0:
        scale = (bottom_zone_d * 0.90) / prv_height_needed
        scale = max(scale, 0.70)
        for room in dims:
            w, d = dims[room]
            nw = max(round(w * scale, 2), NBC_MIN_WIDTH.get(room, 1.0))
            nd = round(d * scale, 2)
            min_area = NBC_MIN_AREA.get(room, 1.0)
            if nw * nd < min_area * 0.85:
                nd = round(min_area * 0.85 / nw, 2)
            dims[room] = (nw, nd)

    # Recompute after scaling
    svc_x = round(net_w - svc_w, 2)
    pub_y  = round(net_d - pub_d, 2)

    placement = {}

    # ── STEP 5: PUBLIC zone ───────────────────────────────────────
    if 'verandah' in rooms_needed:
        vw, vd = dims['verandah']
        vd = min(vd, pub_d * 0.55)
        placement['verandah'] = {
            'x': 0.0,
            'y': round(net_d - vd, 2),
            'w': round(net_w, 2),   # full width always
            'd': round(vd, 2),
        }
        living_y = round(net_d - vd - dims.get('living', (0, 2.5))[1], 2)
        living_y = max(living_y, pub_y)
    else:
        living_y = pub_y

    if 'living' in rooms_needed:
        lw, ld = dims['living']
        lw = min(lw, svc_x)
        ld = min(ld, net_d - living_y - (
            dims.get('verandah', (0, 0))[1] if 'verandah' in placement else 0))
        ld = max(ld, 2.0)
        placement['living'] = {
            'x': 0.0,
            'y': round(pub_y, 2),
            'w': round(lw, 2),
            'd': round(ld, 2),
        }

    if 'dining' in rooms_needed and 'living' in placement:
        dw, dd = dims['dining']
        lv = placement['living']
        dx = round(lv['x'] + lv['w'], 2)
        dw = min(dw, svc_x - dx)
        if dw >= 1.5:
            placement['dining'] = {
                'x': dx,
                'y': lv['y'],
                'w': round(dw, 2),
                'd': round(min(dd, lv['d']), 2),
            }

    # ── STEP 6: SERVICE zone (stack top → bottom, below verandah) ─
    vd_actual = placement.get('verandah', {}).get('d', 1.8)
    svc_cursor = round(net_d - vd_actual, 2)

    for room in svc_rooms:
        rw, rd = dims[room]
        rw = min(rw, svc_w)
        actual_y = round(svc_cursor - rd, 2)
        actual_y = max(0.0, actual_y)
        placement[room] = {
            'x': svc_x,
            'y': actual_y,
            'w': round(rw, 2),
            'd': round(rd, 2),
        }
        svc_cursor = actual_y

    # ── STEP 7: PRIVATE zone ──────────────────────────────────────
    # master_bedroom: bottom-left
    if 'master_bedroom' in rooms_needed:
        mbw, mbd = dims['master_bedroom']
        mbw = min(mbw, prv_w * 0.85)
        mbd = min(mbd, bottom_zone_d * 0.50)
        mbd = max(mbd, NBC_MIN_AREA.get('master_bedroom', 9.5) / mbw)
        placement['master_bedroom'] = {
            'x': 0.0, 'y': 0.0,
            'w': round(mbw, 2), 'd': round(mbd, 2),
        }

    # toilet_attached: directly above master_bedroom
    if 'toilet_attached' in rooms_needed and 'master_bedroom' in placement:
        mb = placement['master_bedroom']
        taw, tad = dims['toilet_attached']
        taw = min(taw, mb['w'])
        tad = min(tad, bottom_zone_d - mb['d'])
        tad = max(tad, 1.2)
        placement['toilet_attached'] = {
            'x': 0.0,
            'y': round(mb['d'], 2),
            'w': round(taw, 2),
            'd': round(tad, 2),
        }

    # Extra bedrooms
    mb  = placement.get('master_bedroom',  {'w': 0.0, 'd': 0.0})
    ta  = placement.get('toilet_attached', {'d': 0.0})
    stack_top  = round(mb['d'] + ta['d'], 2)
    right_of_mb = round(mb['w'], 2)
    space_right = round(prv_w - right_of_mb, 2)
    space_above = round(bottom_zone_d - stack_top, 2)

    bed_cursor_x = right_of_mb
    bed_cursor_y = 0.0
    row_h = 0.0
    used_above = False

    for room in ['bedroom_2', 'bedroom_3', 'bedroom_4']:
        if room not in rooms_needed:
            continue
        rw, rd = dims[room]
        rw = max(rw, NBC_MIN_WIDTH.get(room, 2.1))
        min_area = NBC_MIN_AREA.get(room, 7.5)
        if rw * rd < min_area:
            rd = round(min_area / rw + 0.1, 2)

        # Try to place to the RIGHT of master_bedroom first
        if space_right >= rw + 0.2 and rd <= bottom_zone_d + 0.1:
            rw = min(rw, space_right)
            rd = min(rd, bottom_zone_d)
            placement[room] = {
                'x': round(bed_cursor_x, 2),
                'y': round(bed_cursor_y, 2),
                'w': round(rw, 2),
                'd': round(rd, 2),
            }
            bed_cursor_y = round(bed_cursor_y + rd, 2)
            if bed_cursor_y + rd > bottom_zone_d:
                bed_cursor_x = round(bed_cursor_x + rw, 2)
                bed_cursor_y = 0.0
            space_right -= rw
        else:
            # Place ABOVE master_bedroom + toilet stack
            if not used_above:
                bed_cursor_x = 0.0
                bed_cursor_y = stack_top
                row_h = 0.0
                used_above = True
            rw = min(rw, prv_w)
            rd = min(rd, space_above)
            rd = max(rd, 1.5)
            if bed_cursor_x + rw > prv_w + 0.05:
                bed_cursor_y = round(bed_cursor_y + row_h, 2)
                bed_cursor_x = 0.0
                row_h = 0.0
            placement[room] = {
                'x': round(bed_cursor_x, 2),
                'y': round(bed_cursor_y, 2),
                'w': round(rw, 2),
                'd': round(rd, 2),
            }
            bed_cursor_x = round(bed_cursor_x + rw, 2)
            row_h = max(row_h, rd)

    # ── STEP 8: Clamp and centroids ───────────────────────────────
    for room, r in placement.items():
        r['w'] = round(max(r['w'], 0.5), 2)
        r['d'] = round(max(r['d'], 0.5), 2)
        r['x'] = round(max(0.0, min(r['x'], net_w - r['w'])), 2)
        r['y'] = round(max(0.0, min(r['y'], net_d - r['d'])), 2)
        r['cx'] = round(r['x'] + r['w'] / 2, 3)
        r['cy'] = round(r['y'] + r['d'] / 2, 3)

    # ── STEP 9: Error injection ───────────────────────────────────
    # Build a map of occupied regions BEFORE injecting errors
    # so error placement goes to genuinely empty spaces
    error_type = None
    if rng.random() < error_prob:
        error_choice = str(rng.choice([
            'overlap', 'vastu_kitchen_sw', 'detach_toilet',
            'nbc_size_violation', 'living_isolated',
        ]))
        error_type = error_choice

        if error_choice == 'overlap':
            # Move a random non-verandah room to partially overlap a neighbour
            candidates = [r for r in placement if r != 'verandah']
            if len(candidates) >= 2:
                target = str(rng.choice(candidates))
                others = [r for r in candidates if r != target]
                ref_name = str(rng.choice(others))
                ref = placement[ref_name]
                # Shift target so it overlaps ref by ~35% of ref width
                placement[target]['x'] = round(
                    max(0.0, ref['x'] + ref['w'] * 0.35), 2)
                placement[target]['y'] = ref['y']
                r = placement[target]
                r['cx'] = round(r['x'] + r['w'] / 2, 3)
                r['cy'] = round(r['y'] + r['d'] / 2, 3)

        elif error_choice == 'vastu_kitchen_sw':
            # Move kitchen to SW — but first move master_bedroom out of SW
            if 'kitchen' in placement:
                if 'master_bedroom' in placement:
                    # Shift master_bedroom right temporarily
                    placement['master_bedroom']['x'] = round(
                        net_w * 0.5, 2)
                    mb = placement['master_bedroom']
                    mb['cx'] = round(mb['x'] + mb['w'] / 2, 3)
                placement['kitchen']['x'] = 0.0
                placement['kitchen']['y'] = 0.0
                k = placement['kitchen']
                k['cx'] = round(k['w'] / 2, 3)
                k['cy'] = round(k['d'] / 2, 3)

        elif error_choice == 'detach_toilet':
            # Move toilet_attached to top-right corner (far from master_bedroom)
            if 'toilet_attached' in placement:
                ta = placement['toilet_attached']
                placement['toilet_attached']['x'] = round(
                    net_w - ta['w'], 2)
                placement['toilet_attached']['y'] = round(
                    net_d - ta['d'], 2)
                ta = placement['toilet_attached']
                ta['cx'] = round(ta['x'] + ta['w'] / 2, 3)
                ta['cy'] = round(ta['y'] + ta['d'] / 2, 3)

        elif error_choice == 'nbc_size_violation':
            for room in ['master_bedroom', 'bedroom_2']:
                if room in placement:
                    mw = NBC_MIN_WIDTH.get(room, 2.0) * 0.70
                    placement[room]['w'] = round(mw, 2)
                    placement[room]['d'] = round(mw * 1.05, 2)
                    r = placement[room]
                    r['cx'] = round(r['x'] + r['w'] / 2, 3)
                    r['cy'] = round(r['y'] + r['d'] / 2, 3)
                    break

        elif error_choice == 'living_isolated':
            if 'living' in placement:
                lv = placement['living']
                placement['living']['x'] = round(net_w - lv['w'], 2)
                placement['living']['y'] = 0.0
                lv = placement['living']
                lv['cx'] = round(lv['x'] + lv['w'] / 2, 3)
                lv['cy'] = round(lv['d'] / 2, 3)

    return placement, error_type

def check_violations(placement, bhk, facing, net_w, net_d):
    """
    Checks 10 violation rules and computes 5 scores.
    Returns flat dict with all viol_ flags and score_ values.
    """

    OVERLAP_TOL = 0.10   # rooms must overlap by >10cm to count
    WALL_TOL = 0.35   # rooms within 35cm are considered adjacent

    # ── Helper: do two rooms share a wall? ───────────────────────
    def shares_wall(a, b):
        """
        Returns True if rooms a and b share a wall segment.
        They share a wall when:
          - They overlap on one axis (shared wall runs along that axis)
          - They touch (within WALL_TOL) on the perpendicular axis
        """
        # Check horizontal wall (rooms stacked vertically)
        y_touch = (abs(a['y'] + a['d'] - b['y']) < WALL_TOL or
                   abs(b['y'] + b['d'] - a['y']) < WALL_TOL)
        x_overlap = (a['x'] < b['x'] + b['w'] - WALL_TOL and
                     a['x'] + a['w'] > b['x'] + WALL_TOL)

        # Check vertical wall (rooms side by side)
        x_touch = (abs(a['x'] + a['w'] - b['x']) < WALL_TOL or
                   abs(b['x'] + b['w'] - a['x']) < WALL_TOL)
        y_overlap = (a['y'] < b['y'] + b['d'] - WALL_TOL and
                     a['y'] + a['d'] > b['y'] + WALL_TOL)

        return (y_touch and x_overlap) or (x_touch and y_overlap)

    # ── Helper: do two rooms truly overlap? ──────────────────────
    def truly_overlaps(a, b):
        """
        Returns True only if rooms overlap by more than OVERLAP_TOL
        on BOTH axes simultaneously. Touching edges do NOT count.
        """
        overlap_x = min(a['x']+a['w'], b['x']+b['w']) - max(a['x'], b['x'])
        overlap_y = min(a['y']+a['d'], b['y']+b['d']) - max(a['y'], b['y'])
        if overlap_x <= OVERLAP_TOL or overlap_y <= OVERLAP_TOL:
            return False
        overlap_area = overlap_x * overlap_y
        return overlap_area > (OVERLAP_TOL * OVERLAP_TOL)

    # ── Initialise all flags to 0 ────────────────────────────────
    v = {
        'viol_overlap': 0,
        'viol_nbc_area': 0,
        'viol_nbc_width': 0,
        'viol_kitchen_bedroom': 0,
        'viol_toilet_kitchen': 0,
        'viol_toilet_dining': 0,
        'viol_vastu_kitchen_sw': 0,
        'viol_vastu_master_ne': 0,
        'viol_living_not_adjacent_verandah': 0,
        'viol_master_toilet_not_adjacent': 0,
    }

    room_list = list(placement.items())

    # ── CHECK 1: Overlap ─────────────────────────────────────────
    for i in range(len(room_list)):
        for j in range(i + 1, len(room_list)):
            _, ra = room_list[i]
            _, rb = room_list[j]
            if truly_overlaps(ra, rb):
                v['viol_overlap'] = 1
                break
        if v['viol_overlap']:
            break

    # ── CHECK 2: NBC minimum area (skip for micro plots) ────────
    if net_w * net_d >= 55.0:
        for room, r in placement.items():
            min_a = NBC_MIN_AREA.get(room, 0)
            if min_a > 0 and (r['w'] * r['d']) < min_a * 0.88:
                v['viol_nbc_area'] = 1
                break

    # ── CHECK 3: NBC minimum width ───────────────────────────────
    for room, r in placement.items():
        min_w = NBC_MIN_WIDTH.get(room, 0)
        if min_w > 0 and r['w'] < min_w * 0.88:
            v['viol_nbc_width'] = 1
            break

    # ── CHECKS 4-6: Forbidden adjacencies from DB ───────────────
    room_items = list(placement.items())
    for i in range(len(room_items)):
        for j in range(i+1, len(room_items)):
            ra_name, ra = room_items[i]
            rb_name, rb = room_items[j]
            if not shares_wall(ra, rb):
                continue
            pair = (ra_name, rb_name)
            pair_r = (rb_name, ra_name)
            if pair in FORBIDDEN_ADJ_PAIRS or pair_r in FORBIDDEN_ADJ_PAIRS:
                # Classify which violation flag to set
                toilets = {'toilet_attached', 'toilet_common'}
                bedrooms = {'master_bedroom','bedroom_2',
                            'bedroom_3','bedroom_4'}
                names = {ra_name, rb_name}
                if 'kitchen' in names and names & bedrooms:
                    v['viol_kitchen_bedroom'] = 1
                elif 'kitchen' in names and names & toilets:
                    v['viol_toilet_kitchen'] = 1
                elif 'dining' in names and names & toilets:
                    v['viol_toilet_dining'] = 1
                # Generic forbidden adjacency also blocks is_valid
                # even if not one of the named violation flags
                # We capture it by adding a catch-all:
                elif names & toilets and names & {'living','verandah','pooja'}:
                    v['viol_toilet_dining'] = 1  # reuse as generic forbidden

    # ── CHECK 7: Vastu — kitchen in SW quadrant ──────────────────
    if 'kitchen' in placement:
        k = placement['kitchen']
        cx_pct = (k['x'] + k['w']/2) / net_w
        cy_pct = (k['y'] + k['d']/2) / net_d
        if cx_pct < 0.35 and cy_pct < 0.35:
            v['viol_vastu_kitchen_sw'] = 1

    # ── CHECK 8: Vastu — master bedroom in NE quadrant ───────────
    if 'master_bedroom' in placement:
        mb = placement['master_bedroom']
        cx_pct = (mb['x'] + mb['w']/2) / net_w
        cy_pct = (mb['y'] + mb['d']/2) / net_d
        if cx_pct > 0.65 and cy_pct > 0.65:
            v['viol_vastu_master_ne'] = 1

    # ── CHECK 9: Living not adjacent to verandah ─────────────────
    if 'living' in placement and 'verandah' in placement:
        if not shares_wall(placement['living'], placement['verandah']):
            v['viol_living_not_adjacent_verandah'] = 1

    # ── CHECK 10: Master bedroom not adjacent to toilet_attached ──
    if 'master_bedroom' in placement and 'toilet_attached' in placement:
        if not shares_wall(placement['master_bedroom'],
                           placement['toilet_attached']):
            v['viol_master_toilet_not_adjacent'] = 1

    # ── SCORES ───────────────────────────────────────────────────
    s_vastu = 1.0
    if v['viol_vastu_kitchen_sw']:
        s_vastu -= 0.45
    if v['viol_vastu_master_ne']:
        s_vastu -= 0.45
    if v['viol_living_not_adjacent_verandah']:
        s_vastu -= 0.10

    s_nbc = 1.0
    if v['viol_nbc_area']:
        s_nbc -= 0.55
    if v['viol_nbc_width']:
        s_nbc -= 0.35
    if v['viol_overlap']:
        s_nbc -= 0.10

    s_circ = 1.0
    if v['viol_master_toilet_not_adjacent']:
        s_circ -= 0.50
    if v['viol_living_not_adjacent_verandah']:
        s_circ -= 0.35
    if v['viol_overlap']:
        s_circ -= 0.15

    s_adj = 1.0
    if v['viol_kitchen_bedroom']:
        s_adj -= 0.40
    if v['viol_toilet_kitchen']:
        s_adj -= 0.35
    if v['viol_toilet_dining']:
        s_adj -= 0.25

    def clamp(x):
        return round(max(0.0, min(1.0, x)), 3)

    s_overall = (0.30 * s_vastu +
                 0.25 * s_nbc +
                 0.25 * s_circ +
                 0.20 * s_adj)

    v['score_vastu'] = clamp(s_vastu)
    v['score_nbc'] = clamp(s_nbc)
    v['score_circulation'] = clamp(s_circ)
    v['score_adjacency'] = clamp(s_adj)
    v['score_overall'] = clamp(s_overall)
    v['is_valid'] = 1 if sum(
        v[k] for k in v if k.startswith('viol_')
    ) == 0 else 0

    return v


rng = np.random.default_rng(SEED)
records = []
skipped = 0
start = time.time()
elapsed = 0.0

while len(records) < N_SAMPLES:
    plot_w, plot_d = PLOT_SIZES[rng.integers(len(PLOT_SIZES))]
    bhk = int(rng.choice(BHK_TYPES))
    facing = FACINGS[rng.integers(len(FACINGS))]
    district = districts[rng.integers(len(districts))]

    area = plot_w * plot_d
    if bhk == 4 and area < 200:
        skipped += 1
        continue
    if bhk == 3 and area < 110:
        skipped += 1
        continue
    if bhk == 2 and area < 55:
        skipped += 1
        continue
    if bhk == 1 and area < 25:
        skipped += 1
        continue

    climate_zone = DISTRICT_CLIMATE.get(district, "Composite")

    front, rear, side = get_setbacks(area, facing, setbacks_df)
    net_w = round(plot_w - 2 * side, 2)
    net_d = round(plot_d - front - rear, 2)
    if net_w < 2.5 or net_d < 2.5:
        skipped += 1
        continue

    targets = get_room_targets(plot_w, plot_d, bhk, plot_confs_df)
    placement, error_type = place_rooms(
        net_w, net_d, bhk, targets, rng, error_prob=0.40
    )

    results = check_violations(placement, bhk, facing, net_w, net_d)

    row = {
        "plot_w": float(plot_w),
        "plot_d": float(plot_d),
        "plot_area": float(area),
        "net_w": float(net_w),
        "net_d": float(net_d),
        "net_area": round(net_w * net_d, 2),
        "bhk": int(bhk),
        "facing_code": FACING_MAP[facing],
        "climate_code": CLIMATE_MAP.get(climate_zone, 2),
        "error_type": error_type or "none",
    }

    for room in ROOM_LISTS.get(bhk, []):
        pfx = room.replace("_", "")[:10]
        if room in placement:
            room_data = placement[room]
            row[f"{pfx}_w"] = round(room_data["w"], 2)
            row[f"{pfx}_d"] = round(room_data["d"], 2)
            row[f"{pfx}_area"] = round(room_data["w"] * room_data["d"], 2)
            row[f"{pfx}_cx_pct"] = round(room_data["cx"] / net_w, 3)
            row[f"{pfx}_cy_pct"] = round(room_data["cy"] / net_d, 3)
        else:
            for suffix in ["_w", "_d", "_area", "_cx_pct", "_cy_pct"]:
                row[f"{pfx}{suffix}"] = -1.0

    row.update(results)
    records.append(row)

    if len(records) % 5000 == 0:
        elapsed = time.time() - start
        valid_pct = sum(record["is_valid"] for record in records) / len(records) * 100
        print(
            f"  {len(records):>6}/{N_SAMPLES} | "
            f"{valid_pct:.1f}% valid | "
            f"{elapsed:.0f}s elapsed | "
            f"{skipped} skipped"
        )


df = pd.DataFrame(records)
df = df.fillna(-1.0)
os.makedirs("training_data", exist_ok=True)
df.to_parquet(OUT_PATH, index=False, compression="snappy")
elapsed = time.time() - start

print("═══════════════════════════════════════")
print("TRAINING DATA GENERATION COMPLETE")
print("═══════════════════════════════════════")
print(f"Total samples:    {len(df)}")
print(f"Valid plans:      {df['is_valid'].sum()} ({df['is_valid'].mean()*100:.1f}%)")
print(f"Invalid plans:    {(df['is_valid']==0).sum()}")
print(f"Total columns:    {len(df.columns)}")
print(f"Skipped:          {skipped}")
print(f"Time taken:       {elapsed:.0f}s")
print(f"File size:        {os.path.getsize(OUT_PATH)/1024/1024:.1f} MB")
print(f"Saved to:         {os.path.abspath(OUT_PATH)}")

print("\nVIOLATION BREAKDOWN:")
for col in [column for column in df.columns if column.startswith("viol_")]:
    print(f"  {col}: {df[col].sum()} ({df[col].mean()*100:.1f}%)")

print("\nSCORE DISTRIBUTIONS:")
for col in [column for column in df.columns if column.startswith("score_")]:
    print(
        f"  {col}: mean={df[col].mean():.3f}  min={df[col].min():.3f}  max={df[col].max():.3f}"
    )

print("\nFIRST 3 ROWS (key columns only):")
print(
    df[
        ["plot_w", "plot_d", "bhk", "facing_code", "is_valid", "score_overall", "error_type"]
    ]
    .head(3)
    .to_string(index=False)
)

print("\n═══════════════════════════════════════")
print("Next: upload training_data\\floor_plan_samples.parquet to Google Colab")
print("═══════════════════════════════════════")
