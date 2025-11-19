"""Extract inventory definitions from Sugar Glass inventory screenshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import struct
import zlib

import main  # reuse the shape library

ShapeMask = List[List[int]]


def read_rgb(path: Path) -> List[List[Tuple[int, int, int]]]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("Only PNG images are supported")

    offset = 8
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while offset < len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width = int.from_bytes(chunk_data[0:4], "big")
            height = int.from_bytes(chunk_data[4:8], "big")
            bit_depth = chunk_data[8]
            color_type = chunk_data[9]
            if bit_depth != 8:
                raise ValueError("Only 8-bit PNGs are supported")
            if color_type not in (2, 6):
                raise ValueError("Only RGB/RGBA PNGs are supported")
            bpp = {2: 3, 6: 4}[color_type]
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    raw = zlib.decompress(idat)
    stride = width * bpp
    rows: List[bytearray] = []
    pos = 0
    for _ in range(height):
        filter_type = raw[pos]
        pos += 1
        rowdata = bytearray(raw[pos : pos + stride])
        pos += stride
        recon = bytearray(stride)
        for i in range(stride):
            left = recon[i - bpp] if i >= bpp else 0
            up = rows[-1][i] if rows else 0
            up_left = rows[-1][i - bpp] if (rows and i >= bpp) else 0
            if filter_type == 0:
                val = rowdata[i]
            elif filter_type == 1:
                val = (rowdata[i] + left) & 0xFF
            elif filter_type == 2:
                val = (rowdata[i] + up) & 0xFF
            elif filter_type == 3:
                val = (rowdata[i] + (left + up) // 2) & 0xFF
            elif filter_type == 4:
                pa = left + up - up_left
                pa_val = abs(pa - left)
                pb_val = abs(pa - up)
                pc_val = abs(pa - up_left)
                if pa_val <= pb_val and pa_val <= pc_val:
                    predictor = left
                elif pb_val <= pc_val:
                    predictor = up
                else:
                    predictor = up_left
                val = (rowdata[i] + predictor) & 0xFF
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type}")
            recon[i] = val
        rows.append(recon)

    rgb_rows: List[List[Tuple[int, int, int]]] = []
    for row in rows:
        rgb_row: List[Tuple[int, int, int]] = []
        for i in range(0, len(row), bpp):
            rgb_row.append((row[i], row[i + 1], row[i + 2]))
        rgb_rows.append(rgb_row)
    return rgb_rows


def to_gray(rgb_rows: Sequence[Sequence[Tuple[int, int, int]]]) -> List[List[int]]:
    gray: List[List[int]] = []
    for row in rgb_rows:
        gray_row = []
        for r, g, b in row:
            gray_row.append((299 * r + 587 * g + 114 * b) // 1000)
        gray.append(gray_row)
    return gray


def detect_content_bounds(gray: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    rows = len(gray)
    cols = len(gray[0])
    bright_rows = [i for i, row in enumerate(gray) if sum(row) / cols > 140]
    bright_cols = [i for i in range(cols) if sum(gray[r][i] for r in range(rows)) / rows > 140]
    if not bright_rows or not bright_cols:
        raise ValueError("Could not detect grid area")
    top = max(0, min(bright_rows) - 5)
    bottom = min(rows, max(bright_rows) + 5)
    left = max(0, min(bright_cols) - 5)
    right = min(cols, max(bright_cols) + 5)
    return top, bottom, left, right


def split_tiles(
    rgb_rows: Sequence[Sequence[Tuple[int, int, int]]],
    gray_rows: Sequence[Sequence[int]],
    rows: int,
    cols: int,
    bounds: Tuple[int, int, int, int] | None,
) -> List[Tuple[List[List[int]], Tuple[int, int, int]]]:
    if bounds is None:
        top, bottom, left, right = detect_content_bounds(gray_rows)
    else:
        top, bottom, left, right = bounds
    tile_h = (bottom - top) / rows
    tile_w = (right - left) / cols
    tiles: List[Tuple[List[List[int]], Tuple[int, int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            y0 = max(0, int(top + r * tile_h) - 4)
            y1 = min(len(gray_rows), int(top + (r + 1) * tile_h) + 4)
            x0 = max(0, int(left + c * tile_w) - 4)
            x1 = min(len(gray_rows[0]), int(left + (c + 1) * tile_w) + 4)
            sub_gray = [row[x0:x1] for row in gray_rows[y0:y1]]
            sample_color = rgb_rows[min(len(rgb_rows) - 1, y0 + 15)][
                min(len(rgb_rows[0]) - 1, x0 + 15)
            ]
            tiles.append((sub_gray, sample_color))
    return tiles


def extract_mask(tile_gray: Sequence[Sequence[int]]) -> ShapeMask:
    h = len(tile_gray)
    w = len(tile_gray[0])
    visited = [[False] * w for _ in range(h)]
    cells: List[Tuple[float, float]] = []
    for y in range(h):
        for x in range(w):
            if tile_gray[y][x] <= 200 or visited[y][x]:
                continue
            q = [ (y, x) ]
            visited[y][x] = True
            min_y = max_y = y
            min_x = max_x = x
            pixels = 0
            while q:
                cy, cx = q.pop()
                pixels += 1
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny][nx] and tile_gray[ny][nx] > 200:
                        visited[ny][nx] = True
                        q.append((ny, nx))
            if pixels > 15:
                cells.append(((min_y + max_y) / 2, (min_x + max_x) / 2))

    if not cells:
        return []

    def normalize(vals: List[float]) -> List[float]:
        vals = sorted(vals)
        groups: List[List[float]] = []
        for val in vals:
            if not groups or val - groups[-1][-1] > 12:
                groups.append([val])
            else:
                groups[-1].append(val)
        return [sum(group) / len(group) for group in groups]

    row_centers = normalize([cy for cy, _ in cells])
    col_centers = normalize([cx for _, cx in cells])
    grid = [[0] * len(col_centers) for _ in range(len(row_centers))]
    for cy, cx in cells:
        ry = min(range(len(row_centers)), key=lambda i: abs(row_centers[i] - cy))
        cx_idx = min(range(len(col_centers)), key=lambda i: abs(col_centers[i] - cx))
        grid[ry][cx_idx] = 1
    return grid


def mask_signature(mask: ShapeMask) -> Tuple[Tuple[int, int], ...]:
    coords: List[Tuple[int, int]] = []
    for r, row in enumerate(mask):
        for c, value in enumerate(row):
            if value:
                coords.append((r, c))
    if not coords:
        return tuple()
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    normalized = sorted((r - min_r, c - min_c) for r, c in coords)
    return tuple(normalized)


def coords_signature(coords: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    if not coords:
        return tuple()
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


SIGNATURE_TO_SHAPE: Dict[Tuple[Tuple[int, int], ...], str] = {
    coords_signature(main.shape_library.coords(name)): name
    for name in main.shape_library._coords  # type: ignore[attr-defined]
}


def classify_grade(color: Tuple[int, int, int]) -> str:
    r, g, b = color
    if r - max(g, b) > 35:
        return "superepic"
    if r > 200 and g > 200 and b < 120:
        return "unique"
    if b > r and b > g and g > r:
        return "rare"
    return "epic"


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("modifier")
    parser.add_argument("output", type=Path)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--bounds", type=int, nargs=4, metavar=("TOP", "BOTTOM", "LEFT", "RIGHT"))
    parser.add_argument("--role", default="dealer")
    args = parser.parse_args()

    rgb = read_rgb(args.image)
    gray = to_gray(rgb)
    bounds = tuple(args.bounds) if args.bounds else None
    tiles = split_tiles(rgb, gray, args.rows, args.cols, bounds)

    pieces = []
    for idx, (tile, color) in enumerate(tiles):
        mask = extract_mask(tile)
        signature = mask_signature(mask)
        name = SIGNATURE_TO_SHAPE.get(signature)
        if not name:
            raise RuntimeError(f"Unknown shape for tile {idx}: mask={mask}")
        grade = classify_grade(color)
        pieces.append(
            {
                "name": f"{args.modifier}_{idx}",
                "shape": name,
                "grade": grade,
                "modifier": args.modifier,
                "role": args.role,
            }
        )

    args.output.write_text(json.dumps(pieces, ensure_ascii=False, indent=2))
    print(f"Wrote {len(pieces)} pieces to {args.output}")


if __name__ == "__main__":
    main_cli()
