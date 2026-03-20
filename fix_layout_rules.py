# fix_layout_rules.py
# Three surgical fixes to place_rooms_in_bands()
# No hardcoded percentages. Pure architectural rules.

content = open('engine/engine.py', encoding='utf-8').read()
lines   = content.split('\n')

# ── FIX 1: Find toilet_attached placement and move it ABOVE master_bedroom ──
# Current wrong code places toilet beside master_bedroom (same y=0)
# Correct: toilet goes above master_bedroom (y = mb_depth)

old1 = '''    # toilet_attached: ABOVE master_bedroom (shared horizontal wall)
    if "toilet_attached" in rooms_needed:
        ta_d_actual = min(ta_d, middle_d * 0.55)
        ta_d_actual = max(ta_d_actual, 1.2)
        ta_w_actual = min(ta_w, left_w * 0.60)
        ta_w_actual = max(ta_w_actual, 1.2)
        placement["toilet_attached"] = (
            x_left,
            round(y_bottom + mb_d, 2),
            round(ta_w_actual, 2),
            round(ta_d_actual, 2))'''

# Check if this exact text is in file
if old1 in content:
    print("FIX 1: toilet_attached already written correctly")
else:
    print("FIX 1: searching for toilet_attached placement...")
    for i, line in enumerate(lines):
        if 'toilet_attached' in line and 'placement' in line:
            print(f"  line {i+1}: {line.strip()}")

print()

# Show the actual toilet_attached block in the file
start_ta = None
for i, line in enumerate(lines):
    if '"toilet_attached" in rooms_needed' in line:
        start_ta = i
        print(f"Found toilet_attached block at line {i+1}:")
        for j in range(i, min(i+12, len(lines))):
            print(f"  {j+1}: {lines[j]}")
        break

print()

# Show bedroom_2 placement block
for i, line in enumerate(lines):
    if '"bedroom_2" in rooms_needed' in line and 'placement' not in line:
        print(f"Found bedroom_2 block at line {i+1}:")
        for j in range(i, min(i+15, len(lines))):
            print(f"  {j+1}: {lines[j]}")
        break

print()

# Show kitchen placement block
for i, line in enumerate(lines):
    if '"kitchen" in rooms_needed' in line and 'placement' not in line:
        print(f"Found kitchen block at line {i+1}:")
        for j in range(i, min(i+12, len(lines))):
            print(f"  {j+1}: {lines[j]}")
        break
