import pygame
import colorsys
import hashlib
import math
import sys
from simulation import Simulation, SelectionCriteria
from individual import Sensor, Action

# ── Window layout ──
GRID_SIZE = 768                # pixels for the grid area (square)
PANEL_WIDTH = 340              # pixels for the control panel
WINDOW_WIDTH = GRID_SIZE + PANEL_WIDTH
WINDOW_HEIGHT = GRID_SIZE + 60  # extra space for bottom stats bar
FPS_SIM = 60                   # max frames per second during simulation
GRID_LINE_COLOR = (28, 28, 38)

# ── Colors ──
BG_COLOR = (18, 18, 24)
GRID_BG = (10, 10, 16)
PANEL_BG = (24, 24, 32)
TEXT_COLOR = (220, 220, 230)
DIM_TEXT = (120, 120, 140)
ACCENT = (80, 180, 255)
BTN_COLOR = (40, 40, 55)
BTN_HOVER = (55, 55, 75)
BTN_ACTIVE = (80, 180, 255)
SLIDER_BG = (40, 40, 55)
SLIDER_FG = (80, 180, 255)
SELECTION_ZONE_COLOR = (255, 255, 255, 25)
DIVIDER_COLOR = (50, 50, 65)
DROPDOWN_BG = (32, 32, 44)


def genome_to_color(genome):
    """Map genome to HSV color so similar genomes get similar colors.
    Uses continuous features (avg weight, source/sink ratios) instead of hashing."""
    genes = genome.genes
    n = len(genes)
    if n == 0:
        return (128, 128, 128)
    
    # Feature 1: average weight normalized to [0, 1]  (range is -4 to +4)
    avg_weight = sum(g.weight for g in genes) / n
    f_weight = (avg_weight + 4.0) / 8.0  # 0..1
    
    # Feature 2: fraction of genes with source_type == 1 (sensor sources)
    f_src = sum(1 for g in genes if g.source_type == 1) / n
    
    # Feature 3: fraction of genes with sink_type == 1 (action sinks)
    f_snk = sum(1 for g in genes if g.sink_type == 1) / n
    
    # Feature 4: average source_id / 127
    f_sid = sum(g.source_id for g in genes) / (n * 127.0)
    
    # Feature 5: average sink_id / 127
    f_did = sum(g.sink_id for g in genes) / (n * 127.0)
    
    # Feature 6: weight variance (how spread out the weights are)
    avg_w_raw = sum(g.weight for g in genes) / n
    f_var = min(1.0, sum((g.weight - avg_w_raw) ** 2 for g in genes) / (n * 16.0))
    
    # Map features to HSV — spread across full hue range
    # Hue: multiply by larger factors so small differences shift color noticeably
    hue = (f_weight * 0.6 + f_src * 0.5 + f_sid * 0.7 + f_var * 0.4) % 1.0
    # Saturation: keep vivid
    sat = 0.6 + f_snk * 0.25 + f_var * 0.15  # 0.6 - 1.0
    sat = min(1.0, sat)
    # Value: based on sink id and source spread
    val = 0.55 + f_did * 0.3 + f_src * 0.15  # 0.55 - 1.0
    val = min(1.0, val)
    
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return (int(r * 255), int(g * 255), int(b * 255))


class Button:
    def __init__(self, x, y, w, h, text, font):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.font = font
        self.hovered = False
        self.active = False
    
    def draw(self, surface):
        if self.active:
            color = BTN_ACTIVE
            text_color = (0, 0, 0)
        elif self.hovered:
            color = BTN_HOVER
            text_color = TEXT_COLOR
        else:
            color = BTN_COLOR
            text_color = TEXT_COLOR
        
        pygame.draw.rect(surface, color, self.rect, border_radius=6)
        label = self.font.render(self.text, True, text_color)
        lx = self.rect.centerx - label.get_width() // 2
        ly = self.rect.centery - label.get_height() // 2
        surface.blit(label, (lx, ly))
    
    def handle_mouse(self, pos):
        self.hovered = self.rect.collidepoint(pos)
    
    def clicked(self, pos):
        return self.rect.collidepoint(pos)


class Slider:
    def __init__(self, x, y, w, label, min_val, max_val, value, step=1, fmt="{:.0f}", font=None):
        self.x = x
        self.y = y
        self.w = w
        self.h = 20
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.step = step
        self.fmt = fmt
        self.font = font
        self.dragging = False
        self.track_rect = pygame.Rect(x, y + 22, w, 8)
    
    def draw(self, surface):
        # Label + value
        val_str = self.fmt.format(self.value)
        label_surf = self.font.render(f"{self.label}: {val_str}", True, DIM_TEXT)
        surface.blit(label_surf, (self.x, self.y))
        
        # Track
        pygame.draw.rect(surface, SLIDER_BG, self.track_rect, border_radius=4)
        
        # Fill
        ratio = (self.value - self.min_val) / max(self.max_val - self.min_val, 0.0001)
        fill_w = int(self.w * ratio)
        fill_rect = pygame.Rect(self.x, self.y + 22, fill_w, 8)
        pygame.draw.rect(surface, SLIDER_FG, fill_rect, border_radius=4)
        
        # Knob
        knob_x = self.x + fill_w
        knob_y = self.y + 26
        pygame.draw.circle(surface, (255, 255, 255), (knob_x, knob_y), 7)
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check if click is near the track
            expanded = self.track_rect.inflate(10, 20)
            if expanded.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])
    
    def _update_value(self, mouse_x):
        ratio = (mouse_x - self.x) / self.w
        ratio = max(0.0, min(1.0, ratio))
        raw = self.min_val + ratio * (self.max_val - self.min_val)
        # Snap to step
        self.value = round(raw / self.step) * self.step
        self.value = max(self.min_val, min(self.max_val, self.value))


class Dropdown:
    def __init__(self, x, y, w, options, selected_idx, font):
        self.x = x
        self.y = y
        self.w = w
        self.h = 32
        self.options = options
        self.selected_idx = selected_idx
        self.font = font
        self.open = False
        self.rect = pygame.Rect(x, y, w, self.h)
    
    def draw_closed(self, surface):
        # Main box only (always drawn in normal z-order)
        pygame.draw.rect(surface, BTN_COLOR, self.rect, border_radius=6)
        pygame.draw.rect(surface, DIVIDER_COLOR, self.rect, width=1, border_radius=6)
        label = self.font.render(self.options[self.selected_idx], True, TEXT_COLOR)
        surface.blit(label, (self.x + 10, self.y + 7))
        arrow = self.font.render("▼" if not self.open else "▲", True, DIM_TEXT)
        surface.blit(arrow, (self.x + self.w - 24, self.y + 7))
    
    def draw_popup(self, surface):
        # Draw the open dropdown list ON TOP of everything
        if not self.open:
            return
        total_h = self.h * len(self.options)
        # Solid background
        bg_rect = pygame.Rect(self.x, self.y + self.h, self.w, total_h)
        pygame.draw.rect(surface, DROPDOWN_BG, bg_rect, border_radius=4)
        pygame.draw.rect(surface, ACCENT, bg_rect, width=1, border_radius=4)
        
        mouse_pos = pygame.mouse.get_pos()
        for i, opt in enumerate(self.options):
            oy = self.y + self.h + i * self.h
            opt_rect = pygame.Rect(self.x, oy, self.w, self.h)
            if opt_rect.collidepoint(mouse_pos):
                pygame.draw.rect(surface, BTN_HOVER, opt_rect, border_radius=2)
            elif i == self.selected_idx:
                pygame.draw.rect(surface, (50, 50, 70), opt_rect, border_radius=2)
            opt_label = self.font.render(opt, True, TEXT_COLOR)
            surface.blit(opt_label, (self.x + 10, oy + 7))
    
    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.open:
                for i in range(len(self.options)):
                    oy = self.y + self.h + i * self.h
                    opt_rect = pygame.Rect(self.x, oy, self.w, self.h)
                    if opt_rect.collidepoint(event.pos):
                        self.selected_idx = i
                        self.open = False
                        return True
                self.open = False
            elif self.rect.collidepoint(event.pos):
                self.open = True
        return False


def draw_grid_lines(surface, grid_x, grid_y, cell_size, world_w, world_h):
    """Draw subtle grid lines."""
    # Only draw grid lines if cells are large enough to see them
    if cell_size < 2:
        return
    for gx in range(world_w + 1):
        x = grid_x + int(gx * cell_size)
        pygame.draw.line(surface, GRID_LINE_COLOR, (x, grid_y), (x, grid_y + GRID_SIZE))
    for gy in range(world_h + 1):
        y = grid_y + int(gy * cell_size)
        pygame.draw.line(surface, GRID_LINE_COLOR, (grid_x, y), (grid_x + GRID_SIZE, y))


def draw_selection_zone(surface, selection, grid_x, grid_y, cell_size, world_w, world_h):
    """Draw a translucent overlay showing the survival zone."""
    overlay = pygame.Surface((GRID_SIZE, GRID_SIZE), pygame.SRCALPHA)
    
    if selection == SelectionCriteria.RIGHT_HALF:
        x = int(world_w / 2 * cell_size)
        pygame.draw.rect(overlay, (80, 255, 80, 25), (x, 0, GRID_SIZE - x, GRID_SIZE))
    
    elif selection == SelectionCriteria.LEFT_HALF:
        x = int(world_w / 2 * cell_size)
        pygame.draw.rect(overlay, (80, 255, 80, 25), (0, 0, x, GRID_SIZE))
    
    elif selection == SelectionCriteria.CENTER_CIRCLE:
        cx = GRID_SIZE // 2
        cy = GRID_SIZE // 2
        radius = int(min(world_w, world_h) / 4 * cell_size)
        pygame.draw.circle(overlay, (80, 255, 80, 25), (cx, cy), radius)
    
    elif selection == SelectionCriteria.CORNERS:
        qw = int(world_w / 4 * cell_size)
        qh = int(world_h / 4 * cell_size)
        for rx, ry in [(0, 0), (GRID_SIZE - qw, 0), (0, GRID_SIZE - qh), (GRID_SIZE - qw, GRID_SIZE - qh)]:
            pygame.draw.rect(overlay, (80, 255, 80, 25), (rx, ry, qw, qh))
    
    elif selection == SelectionCriteria.RIGHT_QUARTER:
        x = int(world_w * 3 / 4 * cell_size)
        pygame.draw.rect(overlay, (80, 255, 80, 25), (x, 0, GRID_SIZE - x, GRID_SIZE))
    
    surface.blit(overlay, (grid_x, grid_y))


def draw_arrow(surface, color, start, end, width=2):
    """Draw a line with an arrowhead."""
    pygame.draw.line(surface, color, start, end, width)
    # Arrowhead
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1:
        return
    dx /= length
    dy /= length
    arrow_size = 8
    # Two points for arrowhead
    ax = end[0] - dx * arrow_size - dy * arrow_size * 0.5
    ay = end[1] - dy * arrow_size + dx * arrow_size * 0.5
    bx = end[0] - dx * arrow_size + dy * arrow_size * 0.5
    by = end[1] - dy * arrow_size - dx * arrow_size * 0.5
    pygame.draw.polygon(surface, color, [end, (int(ax), int(ay)), (int(bx), int(by))])


def analyze_brain(indiv, final_x, final_y, sim_selection, world_w, world_h):
    """Analyze a survivor's brain and explain why it probably survived."""
    sensor_names = [s.name for s in Sensor]
    action_names = [a.name for a in Action]
    num_sensors = len(sensor_names)
    num_actions = len(action_names)
    
    # Aggregate connections
    conn_weights = {}
    for (src_type, src_id, snk_type, snk_id, weight) in indiv.brain.connections:
        key = (src_type, src_id, snk_type, snk_id)
        conn_weights[key] = conn_weights.get(key, 0.0) + weight
    
    # Categorize connections
    sensor_to_action = {}
    sensor_to_neuron = {}
    neuron_to_action = {}
    
    for (src_type, src_id, snk_type, snk_id), weight in conn_weights.items():
        src_name = sensor_names[src_id] if src_type == 1 and src_id < num_sensors else None
        snk_name = action_names[snk_id] if snk_type == 1 and snk_id < num_actions else None
        
        if src_type == 1 and snk_type == 1 and src_name and snk_name:
            key = (src_name, snk_name)
            sensor_to_action[key] = sensor_to_action.get(key, 0) + weight
        elif src_type == 1 and snk_type == 0 and src_name:
            key = (src_name, snk_id)
            sensor_to_neuron[key] = sensor_to_neuron.get(key, 0) + weight
        elif src_type == 0 and snk_type == 1 and snk_name:
            key = (src_id, snk_name)
            neuron_to_action[key] = neuron_to_action.get(key, 0) + weight
    
    # Top direct connections
    top_direct = sorted(sensor_to_action.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
    
    # Indirect paths
    indirect_paths = []
    for (s_name, n_id), sw in sensor_to_neuron.items():
        for (n_id2, a_name), aw in neuron_to_action.items():
            if n_id == n_id2:
                indirect_paths.append((s_name, f"N{n_id}", a_name, sw * aw))
    indirect_paths.sort(key=lambda x: abs(x[3]), reverse=True)
    
    # Build explanation lines
    lines = []
    
    # Position info
    lines.append(f"Position: ({final_x}, {final_y}) — {len(indiv.genome.genes)} genes")
    
    # Would they survive? Based on selection criteria and position
    sel_name = sim_selection.name if sim_selection else "UNKNOWN"
    in_zone = False
    if sim_selection == SelectionCriteria.RIGHT_HALF:
        in_zone = final_x >= world_w // 2
        zone_desc = f"right half (x={final_x}, need >= {world_w//2})"
    elif sim_selection == SelectionCriteria.LEFT_HALF:
        in_zone = final_x < world_w // 2
        zone_desc = f"left half (x={final_x}, need < {world_w//2})"
    elif sim_selection == SelectionCriteria.RIGHT_QUARTER:
        in_zone = final_x >= world_w * 3 // 4
        zone_desc = f"right quarter (x={final_x}, need >= {world_w*3//4})"
    elif sim_selection == SelectionCriteria.CORNERS:
        qw, qh = world_w // 4, world_h // 4
        in_zone = (final_x < qw or final_x >= world_w - qw) and (final_y < qh or final_y >= world_h - qh)
        zone_desc = f"corner zone (x={final_x}, y={final_y})"
    elif sim_selection == SelectionCriteria.CENTER_CIRCLE:
        cx, cy = world_w / 2, world_h / 2
        r = min(world_w, world_h) / 4
        in_zone = (final_x - cx)**2 + (final_y - cy)**2 <= r**2
        zone_desc = f"center circle"
    else:
        zone_desc = "unknown zone"
    
    status = "IN SAFE ZONE" if in_zone else "OUTSIDE SAFE ZONE"
    lines.append(f"{status} ({sel_name}: {zone_desc})")
    
    # What drove it there
    lines.append("")
    if top_direct:
        lines.append("Key wiring (direct):")
        for (s, a), w in top_direct:
            verb = "activates" if w > 0 else "suppresses"
            lines.append(f"  {s} {verb} {a} ({w:+.2f})")
    
    if indirect_paths[:2]:
        lines.append("Key wiring (via neurons):")
        for s, n, a, w in indirect_paths[:2]:
            effect = "excites" if w > 0 else "inhibits"
            lines.append(f"  {s} -> {n} -> {a} ({effect}, {w:+.2f})")
    
    # Movement summary
    lines.append("")
    move_drivers = []
    all_paths = list(sensor_to_action.items()) + [((s, a), w) for s, _, a, w in indirect_paths]
    for (s, a), w in all_paths:
        if "MOVE" in str(a) and abs(w) > 0.3:
            move_drivers.append((s, a, w))
    move_drivers.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if move_drivers:
        s, a, w = move_drivers[0]
        if "EDGE" in s:
            lines.append(f"Strategy: Uses {s} to drive {a} — moves toward nearest edge")
        elif "LOC" in s:
            lines.append(f"Strategy: Uses {s} to drive {a} — moves based on grid position")
        elif "BOUNDARY" in s:
            lines.append(f"Strategy: Uses {s} to drive {a} — reacts to distance from walls")
        elif "BLOCKED" in s:
            lines.append(f"Strategy: Uses {s} to drive {a} — avoids obstacles")
        elif "RANDOM" in s:
            lines.append(f"Strategy: {s} drives {a} — partly random movement")
        else:
            lines.append(f"Strategy: {s} drives {a} ({w:+.2f})")
    else:
        lines.append("Strategy: No strong movement drivers — may have been lucky")
    
    return lines, conn_weights


def show_brain_viewer(screen, sim, clock, survivors_list):
    """Modal brain viewer showing 5 random survivors, paginated. Returns when closed."""
    if not survivors_list:
        return
    
    font_sm = pygame.font.SysFont("Menlo", 12)
    font_md = pygame.font.SysFont("Menlo", 14)
    font_lg = pygame.font.SysFont("Menlo", 18)
    font_title = pygame.font.SysFont("Menlo", 24, bold=True)
    
    W = screen.get_width()
    H = screen.get_height()
    margin = 40
    
    sensor_names = [s.name for s in Sensor]
    action_names = [a.name for a in Action]
    num_sensors = len(sensor_names)
    num_actions = len(action_names)
    
    # Pre-analyze all survivors
    survivor_data = []
    for indiv, fx, fy in survivors_list:
        num_internal = indiv.brain.num_internal
        lines, conn_weights = analyze_brain(indiv, fx, fy, sim.selection, sim.world_width, sim.world_height)
        survivor_data.append((indiv, fx, fy, num_internal, lines, conn_weights))
    
    current_page = 0
    
    # Layout columns
    col_sensor_x = margin + 100
    col_neuron_x = W // 2
    col_action_x = W - margin - 100
    
    # Nav button rects
    btn_prev_rect = pygame.Rect(margin, H - 40, 100, 30)
    btn_next_rect = pygame.Rect(W - margin - 100, H - 40, 100, 30)
    
    running_viewer = True
    while running_viewer:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running_viewer = False
                elif event.key == pygame.K_LEFT:
                    current_page = (current_page - 1) % len(survivor_data)
                elif event.key == pygame.K_RIGHT:
                    current_page = (current_page + 1) % len(survivor_data)
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                close_rect = pygame.Rect(W - 50, 10, 36, 36)
                if close_rect.collidepoint(event.pos):
                    running_viewer = False
                elif btn_prev_rect.collidepoint(event.pos):
                    current_page = (current_page - 1) % len(survivor_data)
                elif btn_next_rect.collidepoint(event.pos):
                    current_page = (current_page + 1) % len(survivor_data)
        
        indiv, fx, fy, num_internal, explain_lines, conn_weights = survivor_data[current_page]
        
        # Diagram area
        diagram_top = 90
        diagram_bottom = H - 200
        diagram_h = diagram_bottom - diagram_top
        
        def node_positions(count, x):
            positions = []
            if count == 0:
                return positions
            spacing = diagram_h / max(count, 1)
            for i in range(count):
                y = diagram_top + int((i + 0.5) * spacing)
                positions.append((x, y))
            return positions
        
        sensor_pos = node_positions(num_sensors, col_sensor_x)
        neuron_pos = node_positions(num_internal, col_neuron_x)
        action_pos = node_positions(num_actions, col_action_x)
        
        max_w = max((abs(w) for w in conn_weights.values()), default=1.0)
        if max_w < 0.001:
            max_w = 1.0
        
        # ── Draw ──
        screen.fill((14, 14, 20))
        
        # Title
        title = font_title.render(f"Individual {current_page + 1} of {len(survivor_data)}", True, ACCENT)
        screen.blit(title, (margin, 16))
        subtitle = font_md.render(f"Current pos ({fx}, {fy})  |  {len(indiv.genome.genes)} genes  |  {num_internal} neurons", True, DIM_TEXT)
        screen.blit(subtitle, (margin, 48))
        hint = font_sm.render("Arrow keys or click < > to navigate  |  ESC to close", True, (80, 80, 100))
        screen.blit(hint, (margin, 68))
        
        # Close button
        close_rect = pygame.Rect(W - 50, 10, 36, 36)
        pygame.draw.rect(screen, BTN_COLOR, close_rect, border_radius=6)
        x_label = font_lg.render("X", True, TEXT_COLOR)
        screen.blit(x_label, (close_rect.centerx - x_label.get_width() // 2, close_rect.centery - x_label.get_height() // 2))
        
        # Column headers
        sh = font_md.render("SENSORS", True, (100, 200, 100))
        screen.blit(sh, (col_sensor_x - sh.get_width() // 2, diagram_top - 20))
        nh = font_md.render("NEURONS", True, (200, 150, 255))
        screen.blit(nh, (col_neuron_x - nh.get_width() // 2, diagram_top - 20))
        ah = font_md.render("ACTIONS", True, (255, 150, 100))
        screen.blit(ah, (col_action_x - ah.get_width() // 2, diagram_top - 20))
        
        # Draw connections
        for (src_type, src_id, snk_type, snk_id), weight in conn_weights.items():
            if src_type == 1:
                if src_id < len(sensor_pos):
                    start = (sensor_pos[src_id][0] + 12, sensor_pos[src_id][1])
                else:
                    continue
            else:
                if src_id < len(neuron_pos):
                    start = (neuron_pos[src_id][0] + 12, neuron_pos[src_id][1])
                else:
                    continue
            
            if snk_type == 1:
                if snk_id < len(action_pos):
                    end = (action_pos[snk_id][0] - 12, action_pos[snk_id][1])
                else:
                    continue
            else:
                if snk_id < len(neuron_pos):
                    end = (neuron_pos[snk_id][0] - 12, neuron_pos[snk_id][1])
                else:
                    continue
            
            intensity = min(1.0, abs(weight) / max_w)
            line_width = max(1, int(intensity * 4))
            
            if weight > 0:
                color = (int(60 + intensity * 140), int(200 * intensity), int(60 + intensity * 80))
            else:
                color = (int(200 * intensity), int(60 + intensity * 40), int(60 + intensity * 40))
            
            draw_arrow(screen, color, start, end, line_width)
            
            mx = (start[0] + end[0]) // 2
            my = (start[1] + end[1]) // 2
            w_label = font_sm.render(f"{weight:.2f}", True, (100, 100, 120))
            screen.blit(w_label, (mx - w_label.get_width() // 2, my - 8))
        
        # Draw sensor nodes
        for i, (nx, ny) in enumerate(sensor_pos):
            pygame.draw.circle(screen, (60, 160, 60), (nx, ny), 10)
            pygame.draw.circle(screen, (100, 220, 100), (nx, ny), 10, 2)
            name = sensor_names[i] if i < len(sensor_names) else f"S{i}"
            label = font_sm.render(name, True, (100, 200, 100))
            screen.blit(label, (nx - label.get_width() - 16, ny - label.get_height() // 2))
        
        # Draw neuron nodes
        for i, (nx, ny) in enumerate(neuron_pos):
            pygame.draw.circle(screen, (100, 70, 180), (nx, ny), 12)
            pygame.draw.circle(screen, (160, 120, 255), (nx, ny), 12, 2)
            label = font_md.render(f"N{i}", True, (200, 150, 255))
            screen.blit(label, (nx - label.get_width() // 2, ny - label.get_height() // 2))
        
        # Draw action nodes
        for i, (nx, ny) in enumerate(action_pos):
            pygame.draw.circle(screen, (180, 80, 50), (nx, ny), 10)
            pygame.draw.circle(screen, (255, 140, 90), (nx, ny), 10, 2)
            name = action_names[i] if i < len(action_names) else f"A{i}"
            label = font_sm.render(name, True, (255, 150, 100))
            screen.blit(label, (nx + 16, ny - label.get_height() // 2))
        
        # ── Explanation panel at bottom ──
        explain_y = H - 190
        pygame.draw.line(screen, DIVIDER_COLOR, (margin, explain_y), (W - margin, explain_y))
        
        for j, line in enumerate(explain_lines):
            if line.startswith("Final position") or line.startswith("Survived"):
                color = ACCENT
                f = font_md
            elif line.startswith("Strategy:"):
                color = (255, 220, 100)
                f = font_md
            elif line.startswith("Key wiring"):
                color = TEXT_COLOR
                f = font_sm
            else:
                color = DIM_TEXT
                f = font_sm
            surf = f.render(line, True, color)
            screen.blit(surf, (margin, explain_y + 8 + j * 17))
        
        # ── Navigation buttons ──
        pygame.draw.rect(screen, BTN_COLOR, btn_prev_rect, border_radius=6)
        prev_label = font_md.render("< Prev", True, TEXT_COLOR)
        screen.blit(prev_label, (btn_prev_rect.centerx - prev_label.get_width() // 2, btn_prev_rect.centery - prev_label.get_height() // 2))
        
        pygame.draw.rect(screen, BTN_COLOR, btn_next_rect, border_radius=6)
        next_label = font_md.render("Next >", True, TEXT_COLOR)
        screen.blit(next_label, (btn_next_rect.centerx - next_label.get_width() // 2, btn_next_rect.centery - next_label.get_height() // 2))
        
        # Page dots
        dot_y = H - 24
        total = len(survivor_data)
        dot_start_x = W // 2 - total * 10
        for i in range(total):
            dx = dot_start_x + i * 20
            c = ACCENT if i == current_page else (60, 60, 80)
            pygame.draw.circle(screen, c, (dx, dot_y), 5)
        
        pygame.display.flip()
        clock.tick(30)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Evolution Simulator")
    clock = pygame.time.Clock()
    
    # Fonts
    font_sm = pygame.font.SysFont("Menlo", 13)
    font_md = pygame.font.SysFont("Menlo", 15)
    font_lg = pygame.font.SysFont("Menlo", 20)
    font_title = pygame.font.SysFont("Menlo", 26, bold=True)
    
    # ── Default settings ──
    settings = {
        "world_size": 128,
        "population": 500,
        "genome_length": 24,
        "internal_neurons": 3,
        "steps_per_gen": 300,
        "mutation_rate": 0.01,
    }
    
    # ── UI elements ──
    px = GRID_SIZE + 16  # panel x start
    pw = PANEL_WIDTH - 32  # panel widget width
    
    selection_options = [c.name.replace("_", " ").title() for c in SelectionCriteria]
    selection_dropdown = Dropdown(px, 74, pw, selection_options, 4, font_md)  # RIGHT_QUARTER default
    
    sliders = [
        Slider(px, 125, pw, "Population", 50, 2000, 500, step=50, fmt="{:.0f}", font=font_sm),
        Slider(px, 175, pw, "Genome Length", 4, 50, 24, step=1, fmt="{:.0f}", font=font_sm),
        Slider(px, 225, pw, "Internal Neurons", 1, 10, 3, step=1, fmt="{:.0f}", font=font_sm),
        Slider(px, 275, pw, "Steps/Gen", 50, 1000, 300, step=50, fmt="{:.0f}", font=font_sm),
        Slider(px, 325, pw, "Mutation Rate", 0.001, 0.1, 0.01, step=0.001, fmt="{:.3f}", font=font_sm),
        Slider(px, 375, pw, "Sim Speed", 1, 50, 1, step=1, fmt="{:.0f}", font=font_sm),
    ]
    
    btn_start = Button(px, 425, pw // 2 - 4, 36, "Start", font_md)
    btn_pause = Button(px + pw // 2 + 4, 425, pw // 2 - 4, 36, "Pause", font_md)
    btn_reset = Button(px, 470, pw, 36, "Reset", font_md)
    btn_brain = Button(px, 514, pw, 36, "View Brain", font_md)
    
    # ── State ──
    sim = None
    running = False
    paused = False
    creature_colors = {}  # cache: individual index -> RGB
    steps_done_this_frame = 0
    gen_phase = "idle"  # "idle", "stepping", "between_gens"
    last_survivors = []  # store survivors from last generation for brain viewer
    
    # Stats
    gen_num = 0
    step_num = 0
    survival_rate = 0.0
    num_survivors = 0
    avg_genes = 0.0
    
    def create_sim():
        nonlocal sim, creature_colors, gen_num, step_num, survival_rate, num_survivors, avg_genes, gen_phase
        sel_idx = selection_dropdown.selected_idx
        sel = list(SelectionCriteria)[sel_idx]
        
        sim = Simulation(
            world_width=128,
            world_height=128,
            population_size=int(sliders[0].value),
            genome_length=int(sliders[1].value),
            num_internal_neurons=int(sliders[2].value),
            steps_per_gen=int(sliders[3].value),
            mutation_rate=sliders[4].value,
            selection=sel,
        )
        sim.spawn_generation()
        
        # Pre-compute colors
        creature_colors = {}
        for i, indiv in enumerate(sim.individuals):
            creature_colors[i] = genome_to_color(indiv.genome)
        
        gen_num = 0
        step_num = 0
        survival_rate = 0.0
        num_survivors = 0
        avg_genes = sliders[1].value
        gen_phase = "stepping"
    
    def advance_generation():
        nonlocal creature_colors, gen_num, step_num, survival_rate, num_survivors, avg_genes, gen_phase, last_survivors
        
        survivors = sim.apply_selection()
        num_survivors = len(survivors)
        survival_rate = num_survivors / sim.population_size
        
        if survivors:
            avg_genes = sum(len(s.genome.genes) for s in survivors) / len(survivors)
        
        # Save up to 5 random survivors with their final positions for brain viewer
        import random as _rng
        sample_size = min(5, len(survivors))
        if sample_size > 0:
            sampled = _rng.sample(survivors, sample_size)
            last_survivors = [(s, s.x, s.y) for s in sampled]
        else:
            last_survivors = []
        
        # Record stats
        stats = {
            "generation": sim.generation,
            "survivors": num_survivors,
            "survival_rate": survival_rate,
            "kill_count": sim.kill_count,
            "avg_genome_length": avg_genes,
        }
        sim.history.append(stats)
        
        # Reproduce
        child_genomes = sim.reproduce(survivors)
        sim.generation += 1
        sim.spawn_generation(child_genomes)
        
        # Recompute colors
        creature_colors = {}
        for i, indiv in enumerate(sim.individuals):
            creature_colors[i] = genome_to_color(indiv.genome)
        
        gen_num = sim.generation
        step_num = 0
        gen_phase = "stepping"
    
    # ── Main loop ──
    while True:
        mouse_pos = pygame.mouse.get_pos()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # Dropdown
            if selection_dropdown.handle_event(event):
                pass
            
            # Sliders
            for s in sliders:
                s.handle_event(event)
            
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if btn_start.clicked(event.pos):
                    if not running:
                        running = True
                        paused = False
                        create_sim()
                    else:
                        paused = False
                
                elif btn_pause.clicked(event.pos):
                    if running:
                        paused = not paused
                
                elif btn_reset.clicked(event.pos):
                    running = False
                    paused = False
                    sim = None
                    gen_phase = "idle"
                    creature_colors = {}
                    gen_num = 0
                    step_num = 0
                    survival_rate = 0.0
                
                elif btn_brain.clicked(event.pos):
                    if sim and sim.individuals:
                        import random as _rng
                        alive = [ind for ind in sim.individuals if ind.alive]
                        sample_size = min(10, len(alive))
                        if sample_size > 0:
                            sampled = _rng.sample(alive, sample_size)
                            viewer_list = [(s, s.x, s.y) for s in sampled]
                            was_paused = paused
                            paused = True
                            show_brain_viewer(screen, sim, clock, viewer_list)
                            paused = was_paused
        
        # Update button states
        btn_start.handle_mouse(mouse_pos)
        btn_pause.handle_mouse(mouse_pos)
        btn_reset.handle_mouse(mouse_pos)
        btn_brain.handle_mouse(mouse_pos)
        btn_start.active = running and not paused
        btn_pause.active = paused
        btn_pause.text = "Resume" if paused else "Pause"
        
        # ── Simulation stepping (run multiple steps, draw once) ──
        if running and not paused and sim and gen_phase == "stepping":
            steps_per_frame = max(1, int(sliders[5].value))  # sim speed slider
            for _ in range(steps_per_frame):
                if sim.current_step < sim.steps_per_gen:
                    sim.run_step()
                    step_num = sim.current_step
                else:
                    advance_generation()
                    break
        
        # ── Draw ──
        screen.fill(BG_COLOR)
        
        # Grid area
        grid_x = 0
        grid_y = 0
        pygame.draw.rect(screen, GRID_BG, (grid_x, grid_y, GRID_SIZE, GRID_SIZE))
        
        # Grid lines
        if sim:
            cell_size = GRID_SIZE / sim.world_width
            draw_grid_lines(screen, grid_x, grid_y, cell_size, sim.world_width, sim.world_height)
        
        # Selection zone overlay
        if sim:
            cell_size = GRID_SIZE / sim.world_width
            draw_selection_zone(screen, sim.selection, grid_x, grid_y, cell_size, sim.world_width, sim.world_height)
        
        # Draw creatures — round, filling the cell
        if sim:
            cell_size = GRID_SIZE / sim.world_width
            radius = max(1, int(cell_size * 0.5))
            
            for i, indiv in enumerate(sim.individuals):
                if not indiv.alive:
                    continue
                cx = grid_x + int((indiv.x + 0.5) * cell_size)
                cy = grid_y + int((indiv.y + 0.5) * cell_size)
                color = creature_colors.get(i, (200, 200, 200))
                pygame.draw.circle(screen, color, (cx, cy), radius)
        
        # ── Panel ──
        pygame.draw.rect(screen, PANEL_BG, (GRID_SIZE, 0, PANEL_WIDTH, WINDOW_HEIGHT))
        pygame.draw.line(screen, DIVIDER_COLOR, (GRID_SIZE, 0), (GRID_SIZE, WINDOW_HEIGHT))
        
        # Title
        title = font_title.render("Simulator", True, ACCENT)
        screen.blit(title, (px, 14))
        subtitle = font_sm.render("Evolution Simulator", True, DIM_TEXT)
        screen.blit(subtitle, (px, 44))
        
        # Selection label
        sel_label = font_sm.render("Selection:", True, DIM_TEXT)
        screen.blit(sel_label, (px, 60))
        selection_dropdown.draw_closed(screen)
        
        # Sliders
        for s in sliders:
            s.draw(screen)
        
        # Buttons
        btn_start.draw(screen)
        btn_pause.draw(screen)
        btn_reset.draw(screen)
        btn_brain.draw(screen)
        
        # ── Stats area ──
        stats_y = 560
        pygame.draw.line(screen, DIVIDER_COLOR, (px, stats_y), (px + pw, stats_y))
        
        stats_label = font_md.render("Stats", True, ACCENT)
        screen.blit(stats_label, (px, stats_y + 8))
        
        info_lines = [
            f"Generation:  {gen_num}",
            f"Step:        {step_num}/{int(sliders[3].value)}",
            f"Survivors:   {num_survivors}/{int(sliders[0].value)} ({survival_rate:.1%})",
            f"Avg Genes:   {avg_genes:.1f}",
        ]
        for j, line in enumerate(info_lines):
            surf = font_sm.render(line, True, TEXT_COLOR)
            screen.blit(surf, (px, stats_y + 32 + j * 20))
        
        # ── Survival graph (mini) ──
        if sim and sim.history:
            graph_y = stats_y + 120
            graph_h = 90
            graph_w = pw
            
            pygame.draw.rect(screen, (15, 15, 20), (px, graph_y, graph_w, graph_h), border_radius=4)
            
            # Axis labels
            g_label = font_sm.render("Survival %", True, DIM_TEXT)
            screen.blit(g_label, (px, graph_y - 16))
            
            history = sim.history
            n = len(history)
            if n > 1:
                max_gens = max(n, 10)
                points = []
                for k, h in enumerate(history):
                    gx = px + int(k / max_gens * graph_w)
                    gy = graph_y + graph_h - int(h["survival_rate"] * graph_h)
                    points.append((gx, gy))
                
                if len(points) >= 2:
                    pygame.draw.lines(screen, ACCENT, False, points, 2)
            
            # 25% / 50% / 75% lines
            for pct in [0.25, 0.5, 0.75]:
                ly = graph_y + graph_h - int(pct * graph_h)
                pygame.draw.line(screen, (40, 40, 50), (px, ly), (px + graph_w, ly), 1)
                pct_label = font_sm.render(f"{pct:.0%}", True, (60, 60, 70))
                screen.blit(pct_label, (px + graph_w - 30, ly - 7))
        
        # ── Bottom bar ──
        bar_y = GRID_SIZE
        pygame.draw.rect(screen, PANEL_BG, (0, bar_y, GRID_SIZE, 60))
        pygame.draw.line(screen, DIVIDER_COLOR, (0, bar_y), (GRID_SIZE, bar_y))
        
        if running:
            status = "PAUSED" if paused else "RUNNING"
        else:
            status = "IDLE — Press Start"
        
        status_surf = font_md.render(status, True, ACCENT if running else DIM_TEXT)
        screen.blit(status_surf, (14, bar_y + 8))
        
        if sim:
            pop_alive = sum(1 for ind in sim.individuals if ind.alive)
            bottom_info = f"Alive: {pop_alive}  |  Gen: {gen_num}  |  Step: {step_num}"
            bottom_surf = font_sm.render(bottom_info, True, DIM_TEXT)
            screen.blit(bottom_surf, (14, bar_y + 34))
        
        # ── Dropdown popup drawn LAST (on top of everything) ──
        selection_dropdown.draw_popup(screen)
        
        pygame.display.flip()
        clock.tick(FPS_SIM)


if __name__ == "__main__":
    main()
