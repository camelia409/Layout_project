import numpy as np

rng = np.random.default_rng(42)

# Simulate a typical 2BHK 12x15 North plot
net_w, net_d = 10.0, 12.0  # after setbacks
bhk = 2

# Step 2 zone fractions (from current code)
pub_d_frac = 0.28
svc_w_frac = 0.38

pub_d  = round(net_d * pub_d_frac, 2)   # 3.36
svc_w  = round(net_w * svc_w_frac, 2)   # 3.80
prv_w  = round(net_w - svc_w, 2)        # 6.20
prv_d  = round(net_d - pub_d, 2)        # 8.64

print(f"Net area:     {net_w} x {net_d}")
print(f"Public zone:  full width x {pub_d}m tall (top)")
print(f"Service zone: {svc_w}m wide x {prv_d}m tall (right)")
print(f"Private zone: {prv_w}m wide x {prv_d}m tall (left)")
print()

# Typical room dims for 2BHK
rooms = {
    "verandah":        (net_w, 1.8),
    "living":          (3.5, 2.8),
    "dining":          (2.5, 2.0),
    "kitchen":         (2.5, 2.5),
    "utility":         (1.5, 1.5),
    "toilet_common":   (1.4, 1.4),
    "master_bedroom":  (3.2, 2.8),
    "toilet_attached": (1.5, 1.5),
    "bedroom_2":       (2.8, 2.5),
}

# PRIVATE ZONE placement trace
mbw = min(rooms["master_bedroom"][0], prv_w * 0.90)  # fix4: 90%
mbd = min(rooms["master_bedroom"][1], prv_d * 0.45)  # fix4: 45%
taw = min(rooms["toilet_attached"][0], mbw)
tad = min(rooms["toilet_attached"][1], prv_d - mbd)

prv_stack_top = round(mbd + tad, 2)

print(f"master_bedroom: x=0, y=0, w={mbw:.2f}, d={mbd:.2f}")
print(f"  → top edge at y={mbd:.2f}")
print(f"toilet_attached: x=0, y={mbd:.2f}, w={taw:.2f}, d={tad:.2f}")
print(f"  → top edge at y={prv_stack_top:.2f}")
print(f"prv_stack_top = {prv_stack_top:.2f}")
print(f"prv_d_avail = {prv_d:.2f}")
print(f"Space above stack for bedroom_2: {round(prv_d - prv_stack_top, 2)}m")
print()

# bedroom_2 placement
b2w = min(rooms["bedroom_2"][0], prv_w)
b2d = rooms["bedroom_2"][1]
min_area = 7.5
if b2w * b2d < min_area:
    b2d = round(min_area / b2w + 0.1, 2)

print(f"bedroom_2 needs: w={b2w:.2f}, d={b2d:.2f}")
print(f"bedroom_2 placed: x=0, y={prv_stack_top:.2f}")
b2_top = prv_stack_top + b2d
print(f"bedroom_2 top edge: y={b2_top:.2f}")
print(f"prv_d_avail: {prv_d:.2f}")

if b2_top > prv_d + 0.05:
    print(f"OVERFLOW: bedroom_2 exceeds private zone by {b2_top - prv_d:.2f}m")
    print("→ This causes viol_overlap when Step 8 clamps y coordinate")
else:
    print("OK: bedroom_2 fits")

print()
print("Service zone stacking:")
svc_rooms = ["kitchen", "utility", "toilet_common"]
vd = rooms["verandah"][1]
cursor_y = round(net_d - vd, 2)
print(f"cursor_y starts at: {cursor_y} (net_d - verandah_d)")
for room in svc_rooms:
    rw, rd = rooms[room]
    rw = min(rw, svc_w)
    actual_y = round(cursor_y - rd, 2)
    actual_y = max(0.0, actual_y)
    right_edge = round(net_w - svc_w + rw, 2)
    print(f"  {room}: x={net_w-svc_w:.2f}, y={actual_y:.2f}, "
          f"w={rw:.2f}, d={rd:.2f}, right={right_edge:.2f}")
    cursor_y = actual_y
