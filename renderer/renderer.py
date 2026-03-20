import os, sys, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Arc
from matplotlib.lines import Line2D
from matplotlib.path import Path
import matplotlib.patheffects as pe

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from engine.engine import FloorPlan, Room, WallSegment, \
    DoorOpening, WindowOpening, generate

# Drawing constants
SCALE         = 50       # pixels per metre (1:50 scale)
MARGIN_LEFT   = 3.0      # metres of drawing margin
MARGIN_RIGHT  = 6.0      # metres (extra for legend)
MARGIN_TOP    = 3.0      # metres
MARGIN_BOTTOM = 4.0      # metres (for title block)
DPI           = 150      # output resolution

# Colours (match reference image)
COL_PAPER      = '#F5F0E8'   # cream background
COL_WALL_EXT   = '#1A1A1A'   # near-black exterior walls
COL_WALL_INT   = '#3A3A3A'   # dark grey interior walls
COL_ROOM_FILL  = '#FFFFFF'   # white room fill
COL_SETBACK    = '#888888'   # grey dashed setback line
COL_BOUNDARY   = '#444444'   # property boundary
COL_DIM_LINE   = '#333333'   # dimension lines
COL_TEXT       = '#1A1A1A'   # room labels
COL_NORTH      = '#CC2200'   # north arrow red

# Room fill colours (subtle, like reference image)
ROOM_COLOURS = {
    'master_bedroom':  '#EBF0FA',
    'bedroom_2':       '#EBF0FA',
    'bedroom_3':       '#EBF0FA',
    'bedroom_4':       '#EBF0FA',
    'living':          '#FAFAFA',
    'dining':          '#FAFAFA',
    'kitchen':         '#FFF8E7',
    'toilet_attached': '#E8F4F0',
    'toilet_common':   '#E8F4F0',
    'utility':         '#F0EBE0',
    'verandah':        '#EFF7EF',
    'pooja':           '#FFF0F5',
    'store':           '#F5F0E0',
}

ROOM_LABELS = {
    'master_bedroom':  'MASTER\nBEDROOM',
    'bedroom_2':       'BEDROOM 2',
    'bedroom_3':       'BEDROOM 3',
    'bedroom_4':       'BEDROOM 4',
    'living':          'LIVING/\nDINING',
    'dining':          'DINING',
    'kitchen':         'KITCHEN',
    'toilet_attached': 'ATTACHED\nTOILET',
    'toilet_common':   'COMMON\nTOILET',
    'utility':         'UTILITY',
    'verandah':        'VERANDAH',
    'pooja':           'NORTH-EAST\nPUJA',
    'store':           'STORE',
}


def m2p(metres):
    """Convert metres to points at current SCALE"""
    return metres * SCALE


def setup_figure(fp: FloorPlan):
    """
    Create matplotlib figure sized to fit the full drawing.
    Returns (fig, ax, origin_x, origin_y)
    where origin is the bottom-left corner of the NET area
    in figure coordinates (metres).
    """
    total_w = fp.plot_w + MARGIN_LEFT + MARGIN_RIGHT
    total_h = fp.plot_d + MARGIN_TOP + MARGIN_BOTTOM
    fig_w = total_w * SCALE / DPI
    fig_h = total_h * SCALE / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    ax.set_xlim(0, total_w)
    ax.set_ylim(0, total_h)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.patch.set_facecolor(COL_PAPER)
    ax.set_facecolor(COL_PAPER)
    origin_x = MARGIN_LEFT + fp.setback_side
    origin_y = MARGIN_BOTTOM + fp.setback_rear
    return fig, ax, origin_x, origin_y


def net2fig(rx, ry, origin_x, origin_y):
    """Convert net-area coordinates to figure coordinates"""
    return origin_x + rx, origin_y + ry


def _room_map(fp: FloorPlan):
    return {room.room_type: room for room in fp.rooms}


def _wall_key(wall: WallSegment):
    return (
        round(wall.x1, 3), round(wall.y1, 3),
        round(wall.x2, 3), round(wall.y2, 3),
        wall.wall_type, wall.room_left, wall.room_right,
    )


def _opening_matches_wall(opening, wall: WallSegment):
    if getattr(opening, 'wall', None) is wall:
        return True
    ow = getattr(opening, 'wall', None)
    if ow is None:
        return False
    return _wall_key(ow) == _wall_key(wall)


def _get_wall_openings(fp: FloorPlan, wall: WallSegment):
    openings = []
    for opening in list(fp.doors) + list(fp.windows):
        if _opening_matches_wall(opening, wall):
            openings.append(opening)
    return openings


def _wall_cardinal(wall: WallSegment, fp: FloorPlan):
    if wall.direction == 'H':
        y = max(wall.y1, wall.y2)
        return 'N' if abs(y - fp.net_d) < 0.08 else 'S'
    x = max(wall.x1, wall.x2)
    return 'E' if abs(x - fp.net_w) < 0.08 else 'W'


def _score_color(score: float):
    if score >= 0.8:
        return '#2E7D32'
    if score >= 0.5:
        return '#B26A00'
    return '#B42318'


def draw_boundaries(ax, fp, origin_x, origin_y):
    """
    Draws property boundary, setback lines, and outer dimensions.
    """
    plot_x0 = MARGIN_LEFT
    plot_y0 = MARGIN_BOTTOM
    plot_x1 = MARGIN_LEFT + fp.plot_w
    plot_y1 = MARGIN_BOTTOM + fp.plot_d

    boundary = mpatches.Rectangle(
        (plot_x0, plot_y0), fp.plot_w, fp.plot_d,
        fill=False, linewidth=0.8, linestyle='--',
        edgecolor=COL_BOUNDARY, zorder=1,
    )
    ax.add_patch(boundary)
    ax.text(
        (plot_x0 + plot_x1) / 2, plot_y1 + 0.18,
        'Property Boundary', ha='center', va='bottom',
        fontsize=5.0, color=COL_BOUNDARY, zorder=2,
    )

    front_side = str(fp.facing).upper()
    if front_side == 'N':
        front_y = plot_y1 - fp.setback_front
        rear_y = plot_y0 + fp.setback_rear
    elif front_side == 'S':
        front_y = plot_y0 + fp.setback_front
        rear_y = plot_y1 - fp.setback_rear
    else:
        front_y = plot_y1 - fp.setback_front
        rear_y = plot_y0 + fp.setback_rear

    left_x = plot_x0 + fp.setback_side
    right_x = plot_x1 - fp.setback_side

    for x1, y1, x2, y2, txt, tx, ty, rot in [
        (plot_x0, front_y, plot_x1, front_y, f'Front: {fp.setback_front:.1f}m', (plot_x0 + plot_x1) / 2, front_y + 0.08, 0),
        (plot_x0, rear_y, plot_x1, rear_y, f'Rear: {fp.setback_rear:.1f}m', (plot_x0 + plot_x1) / 2, rear_y + 0.08, 0),
        (left_x, plot_y0, left_x, plot_y1, f'Side: {fp.setback_side:.1f}m', left_x + 0.08, (plot_y0 + plot_y1) / 2, 90),
        (right_x, plot_y0, right_x, plot_y1, f'Side: {fp.setback_side:.1f}m', right_x + 0.08, (plot_y0 + plot_y1) / 2, 90),
    ]:
        ax.plot([x1, x2], [y1, y2], linestyle='--', linewidth=0.55, color=COL_SETBACK, zorder=1)
        ax.text(tx, ty, txt, ha='center', va='bottom', rotation=rot, fontsize=4.5, color=COL_SETBACK, zorder=2)

    top_dim_y = plot_y1 + 0.7
    left_dim_x = plot_x0 - 0.7
    draw_dim_line(ax, plot_x0, top_dim_y, plot_x1, top_dim_y, f'{fp.plot_w:.1f}m', offset=0.18, is_vertical=False)
    draw_dim_line(ax, left_dim_x, plot_y0, left_dim_x, plot_y1, f'{fp.plot_d:.1f}m', offset=0.18, is_vertical=True)


def _draw_wall_piece(ax, wall: WallSegment, start_along: float, end_along: float, origin_x: float, origin_y: float):
    if end_along <= start_along:
        return
    color = COL_WALL_EXT if wall.wall_type == 'exterior' else COL_WALL_INT
    if wall.direction == 'H':
        x0 = min(wall.x1, wall.x2) + start_along
        y0 = wall.y1 - wall.thickness / 2.0
        fx, fy = net2fig(x0, y0, origin_x, origin_y)
        rect = mpatches.Rectangle((fx, fy), end_along - start_along, wall.thickness,
                                  linewidth=0, facecolor=color, zorder=4)
    else:
        x0 = wall.x1 - wall.thickness / 2.0
        y0 = min(wall.y1, wall.y2) + start_along
        fx, fy = net2fig(x0, y0, origin_x, origin_y)
        rect = mpatches.Rectangle((fx, fy), wall.thickness, end_along - start_along,
                                  linewidth=0, facecolor=color, zorder=4)
    ax.add_patch(rect)


def draw_walls(ax, fp, origin_x, origin_y):
    """
    Draws all wall segments as filled rectangles.
    """
    for wall in fp.walls:
        openings = _get_wall_openings(fp, wall)
        if not openings:
            _draw_wall_piece(ax, wall, 0.0, wall.length, origin_x, origin_y)
            continue

        gaps = []
        for opening in openings:
            gap_w = float(opening.width)
            centre = float(opening.position) * wall.length
            gap_start = max(0.0, centre - gap_w / 2.0)
            gap_end = min(wall.length, centre + gap_w / 2.0)
            gaps.append((gap_start, gap_end))
        gaps.sort()

        merged = []
        for gs, ge in gaps:
            if not merged or gs > merged[-1][1]:
                merged.append([gs, ge])
            else:
                merged[-1][1] = max(merged[-1][1], ge)

        cursor = 0.0
        for gs, ge in merged:
            _draw_wall_piece(ax, wall, cursor, gs, origin_x, origin_y)
            cursor = ge
        _draw_wall_piece(ax, wall, cursor, wall.length, origin_x, origin_y)


def draw_rooms(ax, fp, origin_x, origin_y):
    """
    Draws room fills and labels before walls.
    """
    for room in fp.rooms:
        fx, fy = net2fig(room.x, room.y, origin_x, origin_y)
        rect = mpatches.Rectangle(
            (fx, fy), room.width, room.depth,
            linewidth=0, facecolor=ROOM_COLOURS.get(room.room_type, COL_ROOM_FILL),
            zorder=2,
        )
        ax.add_patch(rect)

        label = ROOM_LABELS.get(room.room_type, room.room_type.replace('_', ' ').upper())
        cx = fx + room.width / 2.0
        cy = fy + room.depth / 2.0
        if room.area >= 9:
            fontsize = 6.5
        elif room.area >= 4:
            fontsize = 5.5
        else:
            fontsize = 4.5

        ax.text(
            cx, cy + 0.15, label,
            ha='center', va='center', fontsize=fontsize,
            fontweight='bold', color=COL_TEXT, zorder=6,
            multialignment='center',
            path_effects=[pe.withStroke(linewidth=1.0, foreground='white')],
        )
        ax.text(
            cx, cy - 0.18, f'{room.width:.1f}m x {room.depth:.1f}m',
            ha='center', va='center', fontsize=max(fontsize - 1.0, 3.8),
            color='#555555', zorder=6,
            path_effects=[pe.withStroke(linewidth=0.8, foreground='white')],
        )

        if room.room_type.startswith('bedroom') or room.room_type == 'master_bedroom':
            bed_w = min(room.width * 0.7, 1.8)
            bed_d = min(room.depth * 0.3, 1.0)
            bx = fx + (room.width - bed_w) / 2.0
            by = fy + max(0.18, room.depth * 0.08)
            ax.add_patch(mpatches.Rectangle((bx, by), bed_w, bed_d,
                                            fill=False, edgecolor='#A5A5A5', linewidth=0.6, zorder=3))
            ax.plot([bx, bx + bed_w], [by + bed_d * 0.52, by + bed_d * 0.52], color='#B0B0B0', linewidth=0.5, zorder=3)
        elif room.room_type in ('living', 'dining'):
            tw = min(0.8, room.width * 0.3)
            td = min(0.6, room.depth * 0.25)
            tx = cx - tw / 2.0
            ty = cy - td / 2.0
            ax.add_patch(mpatches.Rectangle((tx, ty), tw, td, fill=False, edgecolor='#A5A5A5', linewidth=0.55, zorder=3))
            for dx, dy in [(-0.22, -0.16), (0.22, -0.16), (-0.22, 0.16), (0.22, 0.16)]:
                ax.add_patch(mpatches.Circle((cx + dx, cy + dy), 0.04, facecolor='#B0B0B0', edgecolor='none', zorder=3))
        elif room.room_type in ('toilet_attached', 'toilet_common'):
            wc_w, wc_d = 0.5, 0.35
            wx = fx + min(0.2, max(room.width - wc_w - 0.1, 0.05))
            wy = fy + min(0.2, max(room.depth - wc_d - 0.1, 0.05))
            ax.add_patch(mpatches.Rectangle((wx, wy), wc_w, wc_d, fill=False, edgecolor='#8E8E8E', linewidth=0.55, zorder=3))
        elif room.room_type == 'kitchen':
            counter = 0.08
            ax.add_patch(mpatches.Rectangle((fx + 0.05, fy + 0.05), max(room.width - 0.1, 0.2), counter,
                                            facecolor='#D7C7A3', edgecolor='none', zorder=3))
            ax.add_patch(mpatches.Rectangle((fx + 0.05, fy + 0.05), counter, max(room.depth - 0.1, 0.2),
                                            facecolor='#D7C7A3', edgecolor='none', zorder=3))


def _door_center(door: DoorOpening, origin_x: float, origin_y: float):
    wall = door.wall
    if wall.direction == 'H':
        x0 = min(wall.x1, wall.x2)
        cx = x0 + door.position * wall.length
        cy = wall.y1
    else:
        y0 = min(wall.y1, wall.y2)
        cx = wall.x1
        cy = y0 + door.position * wall.length
    return net2fig(cx, cy, origin_x, origin_y)


def draw_doors(ax, fp, origin_x, origin_y):
    """
    Draws door symbols: swing arc, jamb hints, and labels.
    """
    room_map = _room_map(fp)
    for door in fp.doors:
        wall = door.wall
        cx, cy = _door_center(door, origin_x, origin_y)
        label_fs = 4.5

        if wall.direction == 'H':
            x0 = cx - door.width / 2.0
            x1 = cx + door.width / 2.0
            y = cy
            swing_room = room_map.get(door.swing_into)
            swing_up = True
            if swing_room is not None:
                swing_up = (swing_room.y + swing_room.depth / 2.0) >= wall.y1
            sign = 1 if swing_up else -1
            if door.hinge_side == 'left':
                hinge_x = x0
                leaf_end_x = x0
                leaf_end_y = y + sign * door.width
                theta1, theta2 = (0, 90) if sign > 0 else (270, 360)
            else:
                hinge_x = x1
                leaf_end_x = x1
                leaf_end_y = y + sign * door.width
                theta1, theta2 = (90, 180) if sign > 0 else (180, 270)
            if door.door_type == 'archway':
                for px in (x0, x1):
                    ax.plot([px, px], [y - 0.12, y + 0.12], color='#444444', linewidth=0.6, zorder=5)
                ax.add_patch(Arc((cx, y), door.width, door.width * 0.55, theta1=0, theta2=180,
                                 linestyle='--', linewidth=0.6, color='#666666', zorder=5))
            else:
                ax.plot([hinge_x, leaf_end_x], [y, leaf_end_y], color='#444444', linewidth=0.8 if door.label == 'MAIN ENTRANCE' else 0.65, zorder=5)
                ax.add_patch(Arc((hinge_x, y), 2 * door.width, 2 * door.width,
                                 theta1=theta1, theta2=theta2,
                                 linewidth=0.95 if door.label == 'MAIN ENTRANCE' else 0.8,
                                 color='#444444', zorder=5))
            tx, ty = cx, y + sign * 0.38
        else:
            y0 = cy - door.width / 2.0
            y1 = cy + door.width / 2.0
            x = cx
            swing_room = room_map.get(door.swing_into)
            swing_right = True
            if swing_room is not None:
                swing_right = (swing_room.x + swing_room.width / 2.0) >= wall.x1
            sign = 1 if swing_right else -1
            if door.hinge_side == 'left':
                hinge_y = y0
                leaf_end_x = x + sign * door.width
                leaf_end_y = y0
                theta1, theta2 = (270, 360) if sign > 0 else (180, 270)
            else:
                hinge_y = y1
                leaf_end_x = x + sign * door.width
                leaf_end_y = y1
                theta1, theta2 = (0, 90) if sign > 0 else (90, 180)
            if door.door_type == 'archway':
                for py in (y0, y1):
                    ax.plot([x - 0.12, x + 0.12], [py, py], color='#444444', linewidth=0.6, zorder=5)
                ax.add_patch(Arc((x, cy), door.width * 0.55, door.width, theta1=90, theta2=270,
                                 linestyle='--', linewidth=0.6, color='#666666', zorder=5))
            else:
                ax.plot([x, leaf_end_x], [hinge_y, leaf_end_y], color='#444444', linewidth=0.8 if door.label == 'MAIN ENTRANCE' else 0.65, zorder=5)
                ax.add_patch(Arc((x, hinge_y), 2 * door.width, 2 * door.width,
                                 theta1=theta1, theta2=theta2,
                                 linewidth=0.95 if door.label == 'MAIN ENTRANCE' else 0.8,
                                 color='#444444', zorder=5))
            tx, ty = x + sign * 0.38, cy

        if door.label == 'MAIN ENTRANCE':
            ax.text(tx, ty, 'MAIN\nENTRANCE', ha='center', va='center', fontsize=4.6,
                    color='#333333', zorder=7,
                    path_effects=[pe.withStroke(linewidth=0.8, foreground=COL_PAPER)])
            if wall.direction == 'H':
                arrow_end = (cx, cy)
                arrow_start = (cx, cy + (0.8 if wall.y1 >= fp.net_d - 0.1 else -0.8))
            else:
                arrow_end = (cx, cy)
                arrow_start = (cx + (0.8 if wall.x1 >= fp.net_w - 0.1 else -0.8), cy)
            ax.add_patch(FancyArrowPatch(arrow_start, arrow_end, arrowstyle='->', mutation_scale=8,
                                         linewidth=0.7, color='#444444', zorder=7))
        else:
            ax.text(tx, ty, door.label, ha='center', va='center', fontsize=label_fs,
                    color='#333333', zorder=7,
                    path_effects=[pe.withStroke(linewidth=0.8, foreground=COL_PAPER)])


def draw_windows(ax, fp, origin_x, origin_y):
    """
    Draws window symbols in wall gaps.
    """
    for window in fp.windows:
        wall = window.wall
        cx, cy = _door_center(window, origin_x, origin_y)
        if wall.direction == 'H':
            x0 = cx - window.width / 2.0
            x1 = cx + window.width / 2.0
            offsets = [-wall.thickness * 0.32, 0.0, wall.thickness * 0.32]
            if window.is_ventilator:
                offsets = [-wall.thickness * 0.2, wall.thickness * 0.2]
            for off in offsets:
                ax.plot([x0, x1], [cy + off, cy + off], color='#777777' if window.is_ventilator else '#555555',
                        linewidth=0.8 if window.is_ventilator else 1.2, zorder=5)
            tx, ty = cx, cy + 0.32
        else:
            y0 = cy - window.width / 2.0
            y1 = cy + window.width / 2.0
            offsets = [-wall.thickness * 0.32, 0.0, wall.thickness * 0.32]
            if window.is_ventilator:
                offsets = [-wall.thickness * 0.2, wall.thickness * 0.2]
            for off in offsets:
                ax.plot([cx + off, cx + off], [y0, y1], color='#777777' if window.is_ventilator else '#555555',
                        linewidth=0.8 if window.is_ventilator else 1.2, zorder=5)
            tx, ty = cx + 0.32, cy
        ax.text(tx, ty, window.label, ha='center', va='center', fontsize=4.0,
                color='#555555', zorder=7,
                path_effects=[pe.withStroke(linewidth=0.8, foreground=COL_PAPER)])


def draw_dim_line(ax, x1, y1, x2, y2, text, offset, is_vertical):
    ax.plot([x1, x2], [y1, y2], color=COL_DIM_LINE, linewidth=0.6, zorder=6)
    tick = 0.15
    if is_vertical:
        ax.plot([x1 - tick, x1 + tick], [y1, y1], color=COL_DIM_LINE, linewidth=0.6, zorder=6)
        ax.plot([x2 - tick, x2 + tick], [y2, y2], color=COL_DIM_LINE, linewidth=0.6, zorder=6)
        ax.text(x1 - offset, (y1 + y2) / 2.0, text, rotation=90,
                ha='center', va='center', fontsize=5.0, color=COL_DIM_LINE, zorder=7,
                path_effects=[pe.withStroke(linewidth=0.8, foreground=COL_PAPER)])
    else:
        ax.plot([x1, x1], [y1 - tick, y1 + tick], color=COL_DIM_LINE, linewidth=0.6, zorder=6)
        ax.plot([x2, x2], [y2 - tick, y2 + tick], color=COL_DIM_LINE, linewidth=0.6, zorder=6)
        ax.text((x1 + x2) / 2.0, y1 + offset, text,
                ha='center', va='center', fontsize=5.0, color=COL_DIM_LINE, zorder=7,
                path_effects=[pe.withStroke(linewidth=0.8, foreground=COL_PAPER)])


def draw_dimensions(ax, fp, origin_x, origin_y):
    """
    Draws dimension annotations inside and outside the plan.
    """
    top_y = origin_y + fp.net_d + fp.setback_front + 0.5
    left_x = origin_x - fp.setback_side - 0.5
    draw_dim_line(ax, MARGIN_LEFT, top_y, MARGIN_LEFT + fp.plot_w, top_y,
                  f'{fp.plot_w:.1f}m', offset=0.18, is_vertical=False)
    ax.text((MARGIN_LEFT + MARGIN_LEFT + fp.plot_w) / 2.0, top_y - 0.28,
            f'Net: {fp.net_w:.1f}m', ha='center', va='center', fontsize=4.6,
            color=COL_DIM_LINE, zorder=7)
    draw_dim_line(ax, left_x, MARGIN_BOTTOM, left_x, MARGIN_BOTTOM + fp.plot_d,
                  f'{fp.plot_d:.1f}m', offset=0.18, is_vertical=True)

    for room in fp.rooms:
        if room.area <= 4.0:
            continue
        fx, fy = net2fig(room.x, room.y, origin_x, origin_y)
        top = fy + room.depth + 0.05
        right = fx + room.width + 0.05
        draw_dim_line(ax, fx + 0.1, top, fx + room.width - 0.1, top,
                      f'{room.width:.1f}m', offset=0.11, is_vertical=False)
        draw_dim_line(ax, right, fy + 0.1, right, fy + room.depth - 0.1,
                      f'{room.depth:.1f}m', offset=0.11, is_vertical=True)

    ext_wall = next((w for w in fp.walls if w.wall_type == 'exterior'), None)
    int_wall = next((w for w in fp.walls if w.wall_type == 'interior'), None)
    call_x = origin_x + fp.net_w + fp.setback_side + 1.0
    call_y = origin_y + fp.net_d - 0.2
    if ext_wall:
        mx, my = net2fig(*ext_wall.midpoint, origin_x, origin_y)
        ax.plot([call_x, mx], [call_y, my], color=COL_DIM_LINE, linewidth=0.5, zorder=6)
        ax.text(call_x + 0.1, call_y, '230mm', fontsize=5.0, color=COL_DIM_LINE, va='center', zorder=7)
    if int_wall:
        mx, my = net2fig(*int_wall.midpoint, origin_x, origin_y)
        ax.plot([call_x, mx], [call_y - 0.45, my], color=COL_DIM_LINE, linewidth=0.5, zorder=6)
        ax.text(call_x + 0.1, call_y - 0.45, '115mm', fontsize=5.0, color=COL_DIM_LINE, va='center', zorder=7)


def draw_north_arrow(ax, fp, origin_x, origin_y):
    """
    Draws a north arrow in the top-right area of the drawing.
    """
    cx = origin_x + fp.net_w + fp.setback_side + 1.0
    cy = origin_y + fp.net_d + fp.setback_front + 0.5
    radius = 0.4
    ax.add_patch(mpatches.Circle((cx, cy), radius, facecolor='white', edgecolor='#222222', linewidth=0.8, zorder=7))

    angle_map = {'N': 90, 'E': 0, 'S': 270, 'W': 180}
    theta = math.radians(angle_map.get(str(fp.facing).upper(), 90))
    tip = (cx + radius * 0.78 * math.cos(theta), cy + radius * 0.78 * math.sin(theta))
    left = (cx + radius * 0.22 * math.cos(theta + 2.35), cy + radius * 0.22 * math.sin(theta + 2.35))
    right = (cx + radius * 0.22 * math.cos(theta - 2.35), cy + radius * 0.22 * math.sin(theta - 2.35))
    ax.add_patch(mpatches.Polygon([tip, left, right], closed=True, facecolor=COL_NORTH, edgecolor='none', zorder=8))
    ax.text(cx, cy + 0.04, 'N', ha='center', va='center', fontsize=6.5, fontweight='bold', color='#222222', zorder=9)


def draw_legend(ax, fp, origin_x, origin_y):
    """
    Draws a legend on the right side of the drawing.
    """
    x = origin_x + fp.net_w + fp.setback_side + 0.5
    y = origin_y + fp.net_d - 1.0
    ax.text(x, y, 'LEGEND', fontsize=6.2, fontweight='bold', color=COL_TEXT, ha='left', va='top', zorder=7)
    y -= 0.35

    seen = []
    for room in fp.rooms:
        if room.room_type not in seen:
            seen.append(room.room_type)
    for room_type in seen:
        ax.add_patch(mpatches.Rectangle((x, y - 0.18), 0.3, 0.2,
                                        facecolor=ROOM_COLOURS.get(room_type, '#FFFFFF'),
                                        edgecolor='#777777', linewidth=0.4, zorder=7))
        ax.text(x + 0.38, y - 0.08, ROOM_LABELS.get(room_type, room_type).replace('\n', ' '),
                fontsize=5.0, color=COL_TEXT, ha='left', va='center', zorder=7)
        y -= 0.35

    y -= 0.1
    ax.add_patch(mpatches.Rectangle((x, y - 0.18), 0.32, 0.16, facecolor=COL_WALL_EXT, edgecolor='none', zorder=7))
    ax.text(x + 0.42, y - 0.10, 'Ext. Wall 230mm', fontsize=5.0, color=COL_TEXT, ha='left', va='center', zorder=7)
    y -= 0.3
    ax.add_patch(mpatches.Rectangle((x, y - 0.16), 0.26, 0.10, facecolor=COL_WALL_INT, edgecolor='none', zorder=7))
    ax.text(x + 0.42, y - 0.10, 'Int. Wall 115mm', fontsize=5.0, color=COL_TEXT, ha='left', va='center', zorder=7)

    y -= 0.45
    ax.text(x, y, 'PLAN SCORES', fontsize=5.8, fontweight='bold', color=COL_TEXT, ha='left', va='top', zorder=7)
    y -= 0.28
    scores = [
        ('Vastu', fp.score_vastu),
        ('NBC', fp.score_nbc),
        ('Circulation', fp.score_circulation),
        ('Adjacency', fp.score_adjacency),
        ('Overall', fp.score_overall),
    ]
    for name, score in scores:
        ax.text(x, y, f'{name}:', fontsize=5.0, color=COL_TEXT, ha='left', va='center', zorder=7)
        ax.text(x + 1.55, y, f'{score:.0%}', fontsize=5.0, color=_score_color(score), ha='right', va='center', zorder=7)
        y -= 0.24


def draw_title_block(ax, fp, origin_x, origin_y):
    """
    Draws title block at bottom of drawing.
    """
    plot_cx = MARGIN_LEFT + fp.plot_w / 2.0
    y = MARGIN_BOTTOM * 0.35
    ax.plot([MARGIN_LEFT, MARGIN_LEFT + fp.plot_w], [MARGIN_BOTTOM - 0.35, MARGIN_BOTTOM - 0.35],
            color='#6B6B6B', linewidth=0.8, zorder=7)
    ax.text(plot_cx, y + 0.55,
            f'{fp.bhk}BHK RESIDENCE FLOOR PLAN - {fp.facing}-FACING PLOT: {fp.plot_w:.0f}m x {fp.plot_d:.0f}m',
            ha='center', va='center', fontsize=9, fontweight='bold', color=COL_TEXT, zorder=7)
    ax.text(plot_cx, y + 0.20, f'{fp.district}, Tamil Nadu, India',
            ha='center', va='center', fontsize=7, color=COL_TEXT, zorder=7)
    ax.text(plot_cx, y - 0.12,
            f'Climate: {fp.climate_zone}  |  Net area: {fp.net_w:.1f}m x {fp.net_d:.1f}m  |  Scale 1:50',
            ha='center', va='center', fontsize=6, color='#444444', zorder=7)

    sx = MARGIN_LEFT + 0.2
    sy = y - 0.2
    seg = 1.0
    for i in range(5):
        fill = '#111111' if i % 2 == 0 else 'white'
        ax.add_patch(mpatches.Rectangle((sx + i * seg, sy), seg, 0.12,
                                        facecolor=fill, edgecolor='#111111', linewidth=0.5, zorder=7))
    for i in range(6):
        ax.plot([sx + i * seg, sx + i * seg], [sy, sy + 0.17], color='#111111', linewidth=0.5, zorder=7)
        ax.text(sx + i * seg, sy - 0.10, f'{i}', ha='center', va='top', fontsize=4.8, color=COL_TEXT, zorder=7)
    ax.text(sx + 5.2, sy + 0.06, '5m', ha='left', va='center', fontsize=5.0, color=COL_TEXT, zorder=7)


def render(fp: FloorPlan, output_path: str = None, show: bool = False) -> str:
    """
    Main render function. Takes FloorPlan, produces PNG.
    """
    fig, ax, ox, oy = setup_figure(fp)
    draw_rooms(ax, fp, ox, oy)
    draw_boundaries(ax, fp, ox, oy)
    draw_walls(ax, fp, ox, oy)
    draw_doors(ax, fp, ox, oy)
    draw_windows(ax, fp, ox, oy)
    draw_dimensions(ax, fp, ox, oy)
    draw_north_arrow(ax, fp, ox, oy)
    draw_legend(ax, fp, ox, oy)
    draw_title_block(ax, fp, ox, oy)

    if output_path is None:
        output_path = f'output_{fp.district}_{fp.bhk}BHK_{fp.facing}.png'
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    plt.savefig(output_path, dpi=DPI, bbox_inches='tight', facecolor=COL_PAPER)
    if show:
        plt.show()
    plt.close(fig)
    return output_path


if __name__ == '__main__':
    test_cases = [
        {'plot_w': 12, 'plot_d': 15, 'bhk': 2, 'facing': 'N', 'district': 'Coimbatore'},
        {'plot_w': 15, 'plot_d': 20, 'bhk': 3, 'facing': 'N', 'district': 'Chennai'},
        {'plot_w': 20, 'plot_d': 25, 'bhk': 4, 'facing': 'N', 'district': 'Madurai'},
    ]
    os.makedirs('outputs', exist_ok=True)
    for params in test_cases:
        print(f"Rendering {params['plot_w']}x{params['plot_d']} {params['bhk']}BHK {params['facing']} {params['district']}...")
        fp = generate(params)
        path = os.path.join('outputs', f"plan_{params['district']}_{params['bhk']}BHK_{params['facing']}.png")
        out = render(fp, output_path=path)
        print(f'  Saved: {out}')
        print(f'  Rooms: {len(fp.rooms)}  Walls: {len(fp.walls)}  Doors: {len(fp.doors)}  Windows: {len(fp.windows)}')
