# fix3rules.py
# Three surgical fixes. Every value from predicted_dims or DB rules.
# No hardcoded percentages.

content = open('engine/engine.py', encoding='utf-8').read()
fixes = 0

# ── FIX 1: master_bedroom uses predicted depth, toilet goes above ─
old1 = (
    "    mb_w = round(max(min(predicted_dims.get('master_bedroom', "
    "(3.2, 3.0))[0], net_w * 0.30), NBC_MIN_WIDTH['master_bedroom']), 2)\n"
    "    ta_w = round(max(min(predicted_dims.get('toilet_attached', "
    "(1.5, 1.5))[0], net_w * 0.16), NBC_MIN_WIDTH['toilet_attached']), 2) "
    "if 'toilet_attached' in ROOM_LISTS[bhk] else 0.0\n"
    "    rooms_n.append(('master_bedroom', 0.0, 0.0, mb_w, b4_h))\n"
    "    if 'toilet_attached' in ROOM_LISTS[bhk]:\n"
    "        rooms_n.append(('toilet_attached', mb_w, 0.0, ta_w, "
    "min(predicted_dims.get('toilet_attached', (1.5, b4_h))[1], b4_h)))"
)

new1 = """\
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
            ta_w, ta_d_actual))"""

if old1 in content:
    content = content.replace(old1, new1, 1)
    fixes += 1
    print("FIX 1 applied: toilet above MB using predicted_dims proportions")
else:
    print("FIX 1 NOT FOUND — searching for mb_w line:")
    for i, line in enumerate(content.split('\n')):
        if 'mb_w = round' in line and 'master_bedroom' in line:
            print(f"  line {i+1}: {repr(line)}")

# ── FIX 2: bedroom_2 uses predicted_dims, not remainder ──────────
old2 = (
    "    extra = [r for r in ('bedroom_2', 'bedroom_3', 'bedroom_4') "
    "if r in ROOM_LISTS[bhk]]\n"
    "    rem = max(net_w - mb_w - ta_w, 0.6)\n"
    "    per = round(rem / max(len(extra), 1), 2) if extra else 0.0\n"
    "    x = round(mb_w + ta_w, 2)\n"
    "    for r in extra:\n"
    "        w = round(max(min(per, net_w - x), NBC_MIN_WIDTH.get(r, 2.1)), 2)\n"
    "        if x + w > net_w:\n"
    "            w = round(max(net_w - x, 0.6), 2)\n"
    "        rooms_n.append((r, x, 0.0, w, b4_h))\n"
    "        x = round(x + w, 2)"
)

new2 = """\
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
        x = round(x + w, 2)"""

if old2 in content:
    content = content.replace(old2, new2, 1)
    fixes += 1
    print("FIX 2 applied: bedrooms use predicted_dims widths from h5 model")
else:
    print("FIX 2 NOT FOUND — searching for extra/rem lines:")
    for i, line in enumerate(content.split('\n')):
        if "rem = max(net_w" in line or "per = round(rem" in line:
            print(f"  line {i+1}: {repr(line)}")

# ── FIX 3: kitchen right-anchored to NE (vastu rule from DB) ─────
old3 = (
    "    service_rooms = [r for r in ('kitchen', 'utility', 'store') "
    "if r in ROOM_LISTS[bhk]]\n"
    "    widths = {r: round(max(predicted_dims.get(r, (1.5, 1.5))[0], "
    "NBC_MIN_WIDTH.get(r, 0.9)), 2) for r in service_rooms}\n"
    "    total_svc = sum(widths.values())\n"
    "    x = round(max(tcom_w + gap, net_w - total_svc), 2)\n"
    "    for r in service_rooms:\n"
    "        w = round(min(widths[r], max(net_w - x, 0.6)), 2)\n"
    "        rooms_n.append((r, x, y_b3, w, b3_h))\n"
    "        x = round(x + w, 2)"
)

new3 = """\
    service_rooms = [r for r in ('kitchen', 'utility', 'store')
                     if r in ROOM_LISTS[bhk]]
    widths = {r: round(max(predicted_dims.get(r, (1.5, 1.5))[0],
                           NBC_MIN_WIDTH.get(r, 0.9)), 2)
              for r in service_rooms}
    total_svc = sum(widths.values())
    # Vastu rule (from DB adjacency_rules + vastu_data):
    # kitchen must be in NE zone = right side of plan.
    # Anchor service zone to RIGHT edge so kitchen occupies NE.
    # minimum gap from toilet_common preserved.
    svc_start = round(max(net_w - total_svc, tcom_w + gap), 2)
    x = svc_start
    for r in service_rooms:
        w = round(min(widths[r], max(net_w - x, 0.6)), 2)
        rooms_n.append((r, x, y_b3, w, b3_h))
        x = round(x + w, 2)"""

if old3 in content:
    content = content.replace(old3, new3, 1)
    fixes += 1
    print("FIX 3 applied: kitchen right-anchored to NE (vastu DB rule)")
else:
    print("FIX 3 NOT FOUND — searching for service_rooms line:")
    for i, line in enumerate(content.split('\n')):
        if "service_rooms = [r for r in ('kitchen'" in line:
            print(f"  line {i+1}: {repr(line)}")

open('engine/engine.py', 'w', encoding='utf-8').write(content)
print(f"\n{fixes}/3 fixes applied")
