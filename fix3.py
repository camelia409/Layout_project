content = open('generate_training_data.py', encoding='utf-8').read()

old = '''    # Remaining bedrooms: to the right of master_bedroom row
    mb_placed = placement.get("master_bedroom", {"w":0,"d":0,"x":0,"y":0})
    cursor_x = round(mb_placed["x"] + mb_placed["w"], 2)
    cursor_y = 0.0       

    for room in ["bedroom_2", "bedroom_3", "bedroom_4"]:
        if room not in rooms_needed:
            continue     
        rw, rd = dims[room]
        rw = min(rw, prv_w_avail - cursor_x)
        if rw < 1.0:     
            # wrap to next row above master_bedroom    
            cursor_x = 0.0
            mb_top = mb_placed["d"] + dims.get(        
                "toilet_attached", (0, 0))[1]
            cursor_y = round(mb_top, 2)
            rw = min(dims[room][0], prv_w_avail)       
        rd_max = prv_d_avail - cursor_y
        rd = min(rd, rd_max)
        rd = max(rd, 1.0)
        placement[room] = {
            "x": round(cursor_x, 2),
            "y": round(cursor_y, 2),
            "w": round(rw, 2),
            "d": round(rd, 2),
        }
        cursor_x = round(cursor_x + rw, 2)'''

new = '''    # Remaining bedrooms: stacked ABOVE master_bedroom in private zone
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

if old in content:
    content = content.replace(old, new, 1)
    open('generate_training_data.py', 'w', encoding='utf-8').write(content)
    print('FIXED: bedroom placement rewritten')
else:
    # Fallback: replace using stable start/end markers (handles whitespace drift)
    start_marker = '    # Remaining bedrooms: to the right of master_bedroom row'
    end_marker = '        cursor_x = round(cursor_x + rw, 2)'
    start_idx = content.find(start_marker)
    if start_idx != -1:
        end_idx = content.find(end_marker, start_idx)
        if end_idx != -1:
            end_line_idx = content.find('\n', end_idx)
            if end_line_idx == -1:
                end_line_idx = len(content)
            else:
                end_line_idx = end_line_idx
            content = content[:start_idx] + new + content[end_line_idx:]
            open('generate_training_data.py', 'w', encoding='utf-8').write(content)
            print('FIXED: bedroom placement rewritten')
        else:
            print('NOT FOUND')
    else:
        print('NOT FOUND')

    # Show lines 344-371 to debug
    for i, line in enumerate(content.split('\n')):
        if 344 <= i+1 <= 371:
            print(f'  {i+1}: {ascii(line)}')
