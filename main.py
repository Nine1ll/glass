"""Sugar Glass board solver.

This module searches for the highest scoring placement of Sugar Glass pieces on a
7x7 board with locked cells.  Shapes are not rotatable, and the scoring rules
follow the description supplied in the prompt:

* Grade per-cell scores: Rare=30, Epic=60, Super Epic=120, Unique=250.
* Set bonus: placing at least 9 cells of the chosen modifier yields +265 points
  and every additional 3 cells (up to 21) adds another +265.
* Only modifiers selected for the current build are eligible for base points and
  set bonus contributions.

The solver can easily be extended to support shapes detected from images.  For
now, a curated library of shapes (1~5 cells plus the unique patterns shown in the
prompt) is provided along with an example inventory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

# ---------------------------------------------------------------------------
# Shape library
# ---------------------------------------------------------------------------


class ShapeLibrary:
    """Stores every non-rotatable shape and pre-computed metadata."""

    def __init__(self) -> None:
        self._masks: Dict[str, Tuple[Tuple[int, ...], ...]] = {}
        self._coords: Dict[str, Tuple[Tuple[int, int], ...]] = {}
        self._dims: Dict[str, Tuple[int, int]] = {}
        self._cell_counts: Dict[str, int] = {}

    def register(self, name: str, mask: Sequence[Sequence[int]]) -> None:
        if name in self._masks:
            raise ValueError(f"Shape '{name}' already registered")

        rows = tuple(tuple(int(value) for value in row) for row in mask)
        coords: List[Tuple[int, int]] = []
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                if value:
                    coords.append((r, c))

        self._masks[name] = rows
        self._coords[name] = tuple(coords)
        self._dims[name] = (len(rows), len(rows[0]) if rows else 0)
        self._cell_counts[name] = len(coords)

    def coords(self, name: str) -> Tuple[Tuple[int, int], ...]:
        return self._coords[name]

    def dims(self, name: str) -> Tuple[int, int]:
        return self._dims[name]

    def cell_count(self, name: str) -> int:
        return self._cell_counts[name]


shape_library = ShapeLibrary()


# Helper to make registrations slightly more compact
register = shape_library.register

# 1~3 cell shapes -----------------------------------------------------------
register("1_dot", [[1]])
register("2_bar_h", [[1, 1]])
register("2_bar_v", [[1], [1]])
register("3_bar_h", [[1, 1, 1]])
register("3_bar_v", [[1], [1], [1]])
register("3_L_nw", [[1, 0], [1, 1]])
register("3_L_ne", [[0, 1], [1, 1]])
register("3_L_sw", [[1, 1], [1, 0]])
register("3_L_se", [[1, 1], [0, 1]])

# 4 cell shapes (tetromino family) -----------------------------------------
register("4_square", [[1, 1], [1, 1]])
register("4_bar_h", [[1, 1, 1, 1]])
register("4_bar_v", [[1], [1], [1], [1]])
register("4_T_up", [[0, 1, 0], [1, 1, 1]])
register("4_T_down", [[1, 1, 1], [0, 1, 0]])
register("4_T_left", [[0, 1], [1, 1], [0, 1]])
register("4_T_right", [[1, 0], [1, 1], [1, 0]])
register("4_L_tall", [[1, 0], [1, 0], [1, 1]])
register("4_J_tall", [[0, 1], [0, 1], [1, 1]])
register("4_L_wide", [[1, 1, 1], [1, 0, 0]])
register("4_J_wide", [[1, 1, 1], [0, 0, 1]])
register("4_S_h", [[0, 1, 1], [1, 1, 0]])
register("4_S_v", [[1, 0], [1, 1], [0, 1]])
register("4_Z_h", [[1, 1, 0], [0, 1, 1]])
register("4_Z_v", [[0, 1], [1, 1], [1, 0]])

# 5 cell shapes (pentomino style) ------------------------------------------
register("5_plus", [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
register("5_L_tall", [[1, 0], [1, 0], [1, 0], [1, 1]])
register("5_L_wide", [[1, 1, 1, 1], [1, 0, 0, 0]])
register("5_T_long", [[1, 1, 1, 1, 1], [0, 0, 1, 0, 0]])
register("5_U", [[1, 0, 1], [1, 1, 1]])
register("5_V", [[1, 0, 0], [1, 0, 0], [1, 1, 1]])
register("5_W", [[1, 0, 0], [1, 1, 0], [0, 1, 1]])
register("5_P", [[1, 1], [1, 1], [1, 0]])
register("5_F", [[0, 1, 1], [1, 1, 0], [0, 1, 0]])
register("5_Y", [[0, 1], [1, 1], [0, 1], [0, 1]])
register("5_Z", [[1, 1, 0], [0, 1, 0], [0, 1, 1]])
register("5_S", [[0, 1, 1], [0, 1, 0], [1, 1, 0]])
register("5_N", [[1, 1, 0], [1, 0, 0], [0, 1, 1]])
register("5_arrow", [[0, 1, 0], [1, 1, 1], [0, 1, 1]])
register("5_hook", [[1, 1, 1], [1, 0, 0], [1, 0, 0]])

# Unique shapes from the prompt (non-rotatable) -----------------------------
register("unique_arrow", [[0, 1, 1, 0], [1, 1, 1, 1]])
register("unique_hook", [[1, 1, 1], [1, 0, 0], [1, 0, 0], [1, 0, 0]])
register("unique_split", [[1, 1, 0], [1, 0, 0], [1, 1, 1]])
register("unique_wave", [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])


# ---------------------------------------------------------------------------
# Core domain objects
# ---------------------------------------------------------------------------


SCORE_PER_GRADE = {
    "rare": 30,
    "epic": 60,
    "superepic": 120,
    "unique": 250,
}

BONUS_THRESHOLD = 9
BONUS_STEP_SIZE = 3
BONUS_VALUE = 265
BONUS_CAP = 21


@dataclass(frozen=True)
class GlassPiece:
    name: str
    shape: str
    grade: str
    modifier: str  # e.g. 광휘, 관통, 원소, 파쇄, 축복 등
    role: str      # dealer / striker / supporter (for bookkeeping)

    def cell_count(self) -> int:
        return shape_library.cell_count(self.shape)

    def base_score(self) -> int:
        return SCORE_PER_GRADE[self.grade] * self.cell_count()


@dataclass(frozen=True)
class Placement:
    piece: GlassPiece
    row: int
    col: int


class Board:
    def __init__(self, locked_map: Sequence[Sequence[int]]) -> None:
        if not locked_map:
            raise ValueError("Locked map cannot be empty")
        width = len(locked_map[0])
        for row in locked_map:
            if len(row) != width:
                raise ValueError("All rows must share the same length")

        self.rows = len(locked_map)
        self.cols = width
        self.locked: Tuple[Tuple[bool, ...], ...] = tuple(
            tuple(bool(cell) for cell in row) for row in locked_map
        )

    def empty_grid(self) -> List[List[Optional[GlassPiece]]]:
        return [
            [None if not self.locked[r][c] else _LOCKED for c in range(self.cols)]
            for r in range(self.rows)
        ]

    def can_place(
        self, grid: Sequence[Sequence[Optional[GlassPiece]]], coords: Iterable[Tuple[int, int]], row: int, col: int
    ) -> bool:
        for dr, dc in coords:
            rr, cc = row + dr, col + dc
            if rr < 0 or cc < 0 or rr >= self.rows or cc >= self.cols:
                return False
            if grid[rr][cc] is not None:
                return False
        return True

    def place(
        self,
        grid: List[List[Optional[GlassPiece]]],
        coords: Iterable[Tuple[int, int]],
        row: int,
        col: int,
        piece: GlassPiece,
    ) -> None:
        for dr, dc in coords:
            rr, cc = row + dr, col + dc
            grid[rr][cc] = piece

    @staticmethod
    def copy_grid(grid: Sequence[Sequence[Optional[GlassPiece]]]) -> List[List[Optional[GlassPiece]]]:
        return [list(row) for row in grid]

    @staticmethod
    def count_empty(grid: Sequence[Sequence[Optional[GlassPiece]]]) -> int:
        return sum(cell is None for row in grid for cell in row)


_LOCKED = object()


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------


class PlacementSolver:
    def __init__(
        self,
        board: Board,
        inventory: Sequence[GlassPiece],
        target_modifiers: Optional[Iterable[str]] = None,
    ) -> None:
        self.board = board
        self.target_modifiers: Optional[Set[str]] = (
            set(target_modifiers) if target_modifiers is not None else None
        )

        self.pieces: List[GlassPiece] = sorted(
            inventory,
            key=lambda piece: (piece.base_score(), piece.cell_count()),
            reverse=True,
        )

        self._remaining_scores = self._build_suffix(lambda p: self._piece_score(p))
        self._remaining_cells = self._build_suffix(lambda p: self._eligible_cell_count(p))

        self.best_score = 0
        self.best_grid = board.empty_grid()
        self.best_placements: List[Placement] = []

    def _build_suffix(self, getter) -> List[int]:
        suffix: List[int] = [0] * (len(self.pieces) + 1)
        running = 0
        for idx in range(len(self.pieces) - 1, -1, -1):
            running += getter(self.pieces[idx])
            suffix[idx] = running
        return suffix

    def _piece_score(self, piece: GlassPiece) -> int:
        if not self._counts_for_score(piece):
            return 0
        return piece.base_score()

    def _eligible_cell_count(self, piece: GlassPiece) -> int:
        if not self._counts_for_score(piece):
            return 0
        return piece.cell_count()

    def _counts_for_score(self, piece: GlassPiece) -> bool:
        if self.target_modifiers is None:
            return True
        return piece.modifier in self.target_modifiers

    def solve(self) -> Tuple[int, List[List[Optional[GlassPiece]]], List[Placement]]:
        grid = self.board.copy_grid(self.best_grid)
        self._search(0, grid, 0, 0, [])
        return self.best_score, self.best_grid, self.best_placements

    def _search(
        self,
        idx: int,
        grid: List[List[Optional[GlassPiece]]],
        accumulated_score: int,
        counted_cells: int,
        placements: List[Placement],
    ) -> None:
        current_bonus = compute_set_bonus(counted_cells)
        total = accumulated_score + current_bonus
        if total > self.best_score:
            self.best_score = total
            self.best_grid = Board.copy_grid(grid)
            self.best_placements = placements.copy()

        if idx >= len(self.pieces):
            return

        optimistic = accumulated_score + self._remaining_scores[idx]
        best_possible_cells = min(BONUS_CAP, counted_cells + self._remaining_cells[idx])
        optimistic_bonus = compute_set_bonus(best_possible_cells)
        if optimistic + optimistic_bonus <= self.best_score:
            return

        piece = self.pieces[idx]
        if self.target_modifiers is not None and piece.modifier not in self.target_modifiers:
            self._search(idx + 1, grid, accumulated_score, counted_cells, placements)
            return

        coords = shape_library.coords(piece.shape)
        height, width = shape_library.dims(piece.shape)
        placed_any = False

        for row in range(self.board.rows - height + 1):
            for col in range(self.board.cols - width + 1):
                if not self.board.can_place(grid, coords, row, col):
                    continue

                new_grid = Board.copy_grid(grid)
                self.board.place(new_grid, coords, row, col, piece)
                new_score = accumulated_score + self._piece_score(piece)
                new_cells = counted_cells + piece.cell_count()
                new_placements = placements + [Placement(piece, row, col)]
                placed_any = True
                self._search(idx + 1, new_grid, new_score, new_cells, new_placements)

        # also consider skipping the piece (useful when shape cannot fit)
        if not placed_any or True:
            self._search(idx + 1, grid, accumulated_score, counted_cells, placements)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def compute_set_bonus(count: int) -> int:
    if count < BONUS_THRESHOLD:
        return 0
    capped = min(count, BONUS_CAP)
    bonus_groups = 1 + (capped - BONUS_THRESHOLD) // BONUS_STEP_SIZE
    return bonus_groups * BONUS_VALUE


# ---------------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------------


def example_inventory() -> List[GlassPiece]:
    return [
        GlassPiece("광휘 네모 A", "4_square", "epic", "광휘", "dealer"),
        GlassPiece("광휘 네모 B", "4_square", "epic", "광휘", "dealer"),
        GlassPiece("광휘 3줄", "3_bar_h", "rare", "광휘", "dealer"),
        GlassPiece("광휘 세로", "3_bar_v", "rare", "광휘", "dealer"),
        GlassPiece("광휘 ㄴ", "3_L_ne", "rare", "광휘", "dealer"),
        GlassPiece("관통 ㅜ", "4_T_down", "superepic", "관통", "dealer"),
        GlassPiece("관통 ㅡ", "4_bar_h", "superepic", "관통", "dealer"),
        GlassPiece("원소 ㄴ", "4_L_tall", "epic", "원소", "striker"),
        GlassPiece("축복 유니크", "unique_arrow", "unique", "축복", "supporter"),
    ]


def render_grid(grid: Sequence[Sequence[Optional[GlassPiece]]]) -> str:
    grade_to_icon = {
        "rare": "🟦",
        "epic": "🟪",
        "superepic": "🟥",
        "unique": "🟨",
    }
    lines: List[str] = []
    for row in grid:
        line = []
        for cell in row:
            if cell is _LOCKED:
                line.append("⬛")
            elif cell is None:
                line.append("⬜")
            else:
                line.append(grade_to_icon[cell.grade])
        lines.append("".join(line))
    return "\n".join(lines)


def main() -> None:
    locked_map = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 0, 0, 1],
        [1, 1, 1, 1, 0, 0, 1],
    ]

    board = Board(locked_map)
    inventory = example_inventory()

    print("[광휘 세트 전용 배치]")
    solver = PlacementSolver(board, inventory, target_modifiers={"광휘"})
    best_score, best_grid, placements = solver.solve()
    print(f"최대 점수: {best_score}")
    print(render_grid(best_grid))
    for placement in placements:
        print(
            f" - {placement.piece.name} ({placement.piece.grade}, {placement.piece.modifier}) -> row {placement.row}, col {placement.col}"
        )

    print("\n[관통 세트 전용 배치]")
    pierce_solver = PlacementSolver(board, inventory, target_modifiers={"관통"})
    pierce_score, pierce_grid, pierce_placements = pierce_solver.solve()
    print(f"최대 점수: {pierce_score}")
    print(render_grid(pierce_grid))
    for placement in pierce_placements:
        print(
            f" - {placement.piece.name} ({placement.piece.grade}, {placement.piece.modifier}) -> row {placement.row}, col {placement.col}"
        )


if __name__ == "__main__":
    main()
