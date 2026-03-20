NEW_PLACE_ROOMS = '''def place_rooms(net_w, net_d, bhk, targets, rng, error_prob=0.40):
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
'''

content = open('generate_training_data.py', encoding='utf-8').read()
lines = content.split('\n')

start_line = None
end_line = None
for i, line in enumerate(lines):
    if 'def place_rooms(' in line:
        start_line = i
    if start_line and i > start_line + 5:
        if line.startswith('def ') and 'place_rooms' not in line:
            end_line = i
            break

if start_line is None or end_line is None:
    print(f"ERROR: could not find boundaries. start={start_line} end={end_line}")
else:
    before = '\n'.join(lines[:start_line])
    after  = '\n'.join(lines[end_line:])
    new_content = before + '\n' + NEW_PLACE_ROOMS + '\n' + after
    open('generate_training_data.py', 'w', encoding='utf-8').write(new_content)
    print(f"REPLACED place_rooms (was lines {start_line+1}-{end_line})")
    print(f"New file: {len(new_content.split(chr(10)))} lines")
