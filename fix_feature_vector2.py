# fix_feature_vector2.py
content = open('engine/engine.py', encoding='utf-8').read()

start = content.index('def build_feature_vector(fp: FloorPlan)')
rest  = content[start:]
next_def = rest.index('\ndef ', 4)
old_func = rest[:next_def]

NEW_FUNC = '''def build_feature_vector(fp: FloorPlan) -> pd.DataFrame:
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
'''

new_content = content[:start] + NEW_FUNC + content[start + len(old_func):]
open('engine/engine.py', 'w', encoding='utf-8').write(new_content)
print("FIXED: build_feature_vector() uses exact model column order")
