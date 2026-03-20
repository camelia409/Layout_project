content = open('generate_training_data.py', encoding='utf-8').read()

# Fix 1: Revert master bedroom caps to sensible values
old1 = '        mbw = min(mbw, prv_w_avail * 0.90)\n        mbd = min(mbd, prv_d_avail * 0.45)'
new1 = '        mbw = min(mbw, prv_w_avail * 0.85)\n        mbd = min(mbd, prv_d_avail * 0.52)'

# Fix 2: Make toilet depth aware of remaining space more carefully
old2 = '            tad = min(tad, prv_d_avail - mb["d"])'
new2 = '            tad = min(tad, prv_d_avail * 0.35)'

# Fix 3: After placing toilet, compute stack top correctly
# and ensure bedroom_2 gets enough vertical space
old3 = '''    # Remaining bedrooms: stacked ABOVE master_bedroom in private zone
    # This avoids horizontal overflow since prv_w may be narrow
    mb_placed = placement.get("master_bedroom", {"w":0,"d":0,"x":0,"y":0})
    ta_placed = placement.get("toilet_attached", {"d":0})
    # Start cursor above master_bedroom + toilet stack
    prv_stack_top = round(mb_placed["d"] + ta_placed["d"], 2)
    cursor_x = 0.0
    cursor_y = prv_stack_top
    row_height = 0.0

    for room in ["bedroom_2", "bedroom_3", "bedroom_4"]:
        if room not in rooms_needed:
            continue
        rw, rd = dims[room]
        # Enforce NBC minimums hard
        rw = max(rw, NBC_MIN_WIDTH.get(room, 2.1))
        min_area = NBC_MIN_AREA.get(room, 7.5)
        if rw * rd < min_area:
            rd = round(min_area / rw + 0.1, 2)
        # Clamp to available private width
        rw = min(rw, prv_w_avail)
        # If room does not fit horizontally, wrap to next row
        if cursor_x + rw > prv_w_avail + 0.05:
            cursor_y = round(cursor_y + row_height, 2)
            cursor_x = 0.0
            row_height = 0.0
        # Check vertical space remains
        if cursor_y + rd > prv_d_avail + 0.05:
            rd = round(prv_d_avail - cursor_y, 2)
            rd = max(rd, 1.0)
        placement[room] = {
            "x": round(cursor_x, 2),
            "y": round(cursor_y, 2),
            "w": round(rw, 2),
            "d": round(rd, 2),
        }
        cursor_x = round(cursor_x + rw, 2)
        row_height = max(row_height, rd)'''

new3 = '''    # Remaining bedrooms placed in private zone
    mb_placed = placement.get("master_bedroom", {"w":0,"d":0,"x":0,"y":0})
    ta_placed = placement.get("toilet_attached", {"d":0,"w":0})

    # Compute how much vertical space is left above the mb+toilet stack
    stack_top = round(mb_placed["d"] + ta_placed.get("d", 0), 2)
    space_above = round(prv_d_avail - stack_top, 2)

    # Compute how much horizontal space is to the RIGHT of master_bedroom
    space_right = round(prv_w_avail - mb_placed["w"], 2)

    extra_bedrooms = [r for r in ["bedroom_2","bedroom_3","bedroom_4"]
                      if r in rooms_needed]

    for idx, room in enumerate(extra_bedrooms):
        rw, rd = dims[room]
        # Hard NBC enforcement
        rw = max(rw, NBC_MIN_WIDTH.get(room, 2.1))
        min_area = NBC_MIN_AREA.get(room, 7.5)
        if rw * rd < min_area:
            rd = round(min_area / rw + 0.1, 2)

        # Decide placement strategy:
        # If space to the right of master_bedroom is enough → place beside it
        # Otherwise → place above the stack
        if space_right >= rw + 0.3 and rd <= prv_d_avail * 0.8:
            # Place beside master_bedroom
            bx = round(mb_placed["w"], 2)
            by = 0.0
            rw = min(rw, space_right)
            rd = min(rd, prv_d_avail)
            space_right -= rw
        else:
            # Place above the stack
            bx = round(idx * (prv_w_avail / max(len(extra_bedrooms), 1)), 2)
            by = stack_top
            rw = min(rw, prv_w_avail)
            rd = min(rd, space_above)
            if rd < 1.0:
                rd = 1.0

        placement[room] = {
            "x": round(bx, 2),
            "y": round(by, 2),
            "w": round(rw, 2),
            "d": round(rd, 2),
        }'''

fixes = [
    (old1, new1, "master bedroom caps"),
    (old2, new2, "toilet depth cap"),
    (old3, new3, "bedroom placement strategy"),
]

for old, new, label in fixes:
    if old in content:
        content = content.replace(old, new, 1)
        print(f"FIXED: {label}")
    else:
        print(f"NOT FOUND: {label}")
        for i, line in enumerate(content.split('\n')):
            if any(fragment in line for fragment in old.split('\n')[:2]):
                print(f"  line {i+1}: {repr(line)}")

open('generate_training_data.py', 'w', encoding='utf-8').write(content)
