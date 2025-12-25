from __future__ import annotations

from typing import Dict, Tuple
from PIL import Image, ImageDraw, ImageFont


COLOR_MAP = {
    "red": (220, 60, 60),
    "blue": (60, 120, 220),
    "green": (60, 180, 120),
    "agent": (40, 40, 40),
    "grid": (230, 230, 230),
    "bg": (255, 255, 255),
    "text": (20, 20, 20),
    "holding": (255, 215, 0),
}


def render_obs(
    obs: Dict,
    cell_px: int = 64,
    pad_px: int = 16,
    header_px: int = 70,
) -> Image.Image:
    """
    Renders the env state to an RGB image.
    Layout:
    - header: instruction + holding
    - grid: cells with objects + agent
    """
    size = _infer_grid_size(obs)
    w = pad_px * 2 + size * cell_px
    h = pad_px * 2 + header_px + size * cell_px

    img = Image.new("RGB", (w, h), COLOR_MAP["bg"])
    draw = ImageDraw.Draw(img)

    # Header
    instr = obs.get("instruction", "")
    holding = obs.get("holding", None)
    header_text = f"instruction: {instr}"
    holding_text = f"holding: {holding if holding is not None else 'nothing'}"
    draw.text((pad_px, pad_px), header_text, fill=COLOR_MAP["text"])
    draw.text((pad_px, pad_px + 28), holding_text, fill=COLOR_MAP["text"])

    grid_top = pad_px + header_px
    grid_left = pad_px

    # Grid lines
    for r in range(size + 1):
        y = grid_top + r * cell_px
        draw.line([(grid_left, y), (grid_left + size * cell_px, y)], fill=COLOR_MAP["grid"], width=2)
    for c in range(size + 1):
        x = grid_left + c * cell_px
        draw.line([(x, grid_top), (x, grid_top + size * cell_px)], fill=COLOR_MAP["grid"], width=2)

    # Objects
    for o in obs.get("objects", []):
        color = o["color"]
        rr, cc = o["pos"]
        _draw_cell_marker(draw, grid_left, grid_top, rr, cc, cell_px, COLOR_MAP.get(color, (128, 128, 128)))

    # Agent
    ar, ac = obs.get("agent_pos", (0, 0))
    _draw_agent(draw, grid_left, grid_top, ar, ac, cell_px)

    # If holding, add a small badge on the agent
    if holding is not None:
        _draw_holding_badge(draw, grid_left, grid_top, ar, ac, cell_px)

    return img


def _infer_grid_size(obs: Dict) -> int:
    # best-effort: infer from max coordinate present
    coords = [obs.get("agent_pos", (0, 0))]
    for o in obs.get("objects", []):
        coords.append(tuple(o["pos"]))
    m = max(max(r, c) for r, c in coords)
    return int(m + 1)


def _cell_bounds(grid_left: int, grid_top: int, r: int, c: int, cell_px: int) -> Tuple[int, int, int, int]:
    x0 = grid_left + c * cell_px
    y0 = grid_top + r * cell_px
    x1 = x0 + cell_px
    y1 = y0 + cell_px
    return x0, y0, x1, y1


def _draw_cell_marker(draw: ImageDraw.ImageDraw, grid_left: int, grid_top: int, r: int, c: int, cell_px: int, rgb):
    x0, y0, x1, y1 = _cell_bounds(grid_left, grid_top, r, c, cell_px)
    margin = int(cell_px * 0.18)
    draw.rounded_rectangle(
        [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
        radius=10,
        fill=rgb,
        outline=None,
    )


def _draw_agent(draw: ImageDraw.ImageDraw, grid_left: int, grid_top: int, r: int, c: int, cell_px: int):
    x0, y0, x1, y1 = _cell_bounds(grid_left, grid_top, r, c, cell_px)
    margin = int(cell_px * 0.25)
    draw.ellipse(
        [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
        fill=COLOR_MAP["agent"],
        outline=None,
    )


def _draw_holding_badge(draw: ImageDraw.ImageDraw, grid_left: int, grid_top: int, r: int, c: int, cell_px: int):
    x0, y0, x1, y1 = _cell_bounds(grid_left, grid_top, r, c, cell_px)
    badge_r = int(cell_px * 0.12)
    cx = x1 - int(cell_px * 0.22)
    cy = y0 + int(cell_px * 0.22)
    draw.ellipse([cx - badge_r, cy - badge_r, cx + badge_r, cy + badge_r], fill=COLOR_MAP["holding"])
