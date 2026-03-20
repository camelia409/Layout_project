# fix_feature_vector.py
content = open('engine/engine.py', encoding='utf-8').read()

OLD = '''def build_feature_vector(fp: FloorPlan) -> pd.DataFrame:'''

# Find the full function and replace it
import re

# Find where build_feature_vector starts and ends
start = content.index('def build_feature_vector(fp: FloorPlan)')
# Find next function def after it
rest = content[start:]
next_def = rest.index('\ndef ', 4)
old_func = rest[:next_def]

NEW_FUNC = '''def build_feature_vector(fp: FloorPlan) -> pd.DataFrame:
    """
    Builds 74-column feature vector in EXACT training data order.
    Order must match generate_training_data.py ROOM_LISTS order.
    """
    room_map = {r.room_type: r for r in fp.rooms}

    # Exact room order used during training
    ROOM_ORDER = [
        ("master_bedroom",  "masterbedr"),
        ("toilet_attached", "toiletatta"),
        ("bedroom_2",       "bedroom2"),
        ("living",          "living"),
        ("dining",          "dining"),
        ("kitchen",         "kitchen"),
        ("toilet_common",   "toiletcomm"),
        ("utility",         "utility"),
        ("verandah",        "verandah"),
        ("bedroom_3",       "bedroom3"),
        ("bedroom_4",       "bedroom4"),
        ("pooja",           "pooja"),
        ("store",           "store"),
    ]

    row = {
        "plot_w":    fp.plot_w,
        "plot_d":    fp.plot_d,
        "plot_area": round(fp.plot_w * fp.plot_d, 2),
        "net_w":     fp.net_w,
        "net_d":     fp.net_d,
        "net_area":  round(fp.net_w * fp.net_d, 2),
        "bhk":       fp.bhk,
        "facing_code":  fp.facing_code,
        "climate_code": fp.climate_code,
    }

    for room_type, prefix in ROOM_ORDER:
        if room_type in room_map:
            r = room_map[room_type]
            row[f"{prefix}_w"]      = round(r.width, 2)
            row[f"{prefix}_d"]      = round(r.depth, 2)
            row[f"{prefix}_area"]   = round(r.area,  2)
            row[f"{prefix}_cx_pct"] = round(r.cx_pct, 3)
            row[f"{prefix}_cy_pct"] = round(r.cy_pct, 3)
        else:
            row[f"{prefix}_w"]      = -1.0
            row[f"{prefix}_d"]      = -1.0
            row[f"{prefix}_area"]   = -1.0
            row[f"{prefix}_cx_pct"] = -1.0
            row[f"{prefix}_cy_pct"] = -1.0

    return pd.DataFrame([row])
'''

new_content = content[:start] + NEW_FUNC + content[start + len(old_func):]
open('engine/engine.py', 'w', encoding='utf-8').write(new_content)
print("FIXED: build_feature_vector() rewritten with correct column order")
print("Verify: run python engine\\engine.py and check score_valid")
