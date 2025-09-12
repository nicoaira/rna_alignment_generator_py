"""Utilities for updating bulge graph indices after sequence modifications."""

from .models import BulgeGraph, GraphNode


class BulgeGraphUpdater:
    """Update BulgeGraph positions without reparsing the structure."""

    @staticmethod
    def _shift_indices(bg: BulgeGraph, index: int, delta: int) -> None:
        """Shift all coordinates after a given index by delta.

        Args:
            bg: BulgeGraph to modify.
            index: 0-based position where the modification occurred.
            delta: Amount to shift indices (>0 for insert, <0 for delete).
        """
        threshold = index + 1
        for node in bg.elements.values():
            node.positions = [
                pos + delta if pos >= threshold else pos
                for pos in node.positions
            ]
            if node.start is not None and node.start >= threshold:
                node.start += delta
            if node.end is not None and node.end >= threshold:
                node.end += delta

    @staticmethod
    def _recompute_bounds(node: GraphNode) -> None:
        if node.positions:
            node.start = min(node.positions)
            node.end = max(node.positions)
        else:
            node.start = None
            node.end = None

    @classmethod
    def insert_loop_base(
        cls, bg: BulgeGraph, node_name: str, index: int
    ) -> None:
        """Update graph for a loop base insertion."""
        cls._shift_indices(bg, index, 1)
        node = bg.elements.get(node_name)
        if node is not None:
            node.positions.append(index + 1)
            node.positions.sort()
            cls._recompute_bounds(node)

    @classmethod
    def delete_loop_base(
        cls, bg: BulgeGraph, node_name: str, index: int
    ) -> None:
        """Update graph for a loop base deletion."""
        node = bg.elements.get(node_name)
        if node is not None:
            coord = index + 1
            # When a loop node only stores two coordinates it represents a
            # start/end range rather than explicit positions.  For these nodes
            # we update the bounds while keeping the two-coordinate
            # representation.
            if len(node.positions) == 2:
                start, end = node.positions
                if start <= coord <= end:
                    if coord == start:
                        start += 1
                    else:
                        end -= 1
                node.positions = [start, end]
                cls._recompute_bounds(node)
            elif len(node.positions) == 4:
                ls, le, rs, re = node.positions
                if ls <= coord <= le:
                    if coord == ls:
                        ls += 1
                    else:
                        le -= 1
                elif rs <= coord <= re:
                    if coord == rs:
                        rs += 1
                    else:
                        re -= 1
                node.positions = [ls, le, rs, re]
                cls._recompute_bounds(node)
            elif coord in node.positions:
                node.positions.remove(coord)
                cls._recompute_bounds(node)

        cls._shift_indices(bg, index, -1)

    @classmethod
    def insert_stem_pair(
        cls, bg: BulgeGraph, node_name: str, left_index: int, right_index: int
    ) -> None:
        """Update graph for a stem pair insertion."""
        node = bg.elements.get(node_name)
        orig = None
        if node is not None and len(node.positions) == 4:
            orig = list(node.positions)
            ls, le, rs, re = orig
            # Clamp diff-based indices to expected ranges
            left_index = min(max(left_index, ls - 1), le)
            right_index = min(max(right_index, rs - 1), re)

        if left_index < right_index:
            cls._shift_indices(bg, right_index, 1)
            cls._shift_indices(bg, left_index, 1)
        else:
            cls._shift_indices(bg, left_index, 1)
            cls._shift_indices(bg, right_index, 1)

        if orig is not None:
            ls, le, rs, re = orig
            node.positions = [ls, le + 1, rs + 1, re + 2]
            node.start = node.positions[0]
            node.end = node.positions[-1]

    @classmethod
    def delete_stem_pair(
        cls, bg: BulgeGraph, node_name: str, left_index: int, right_index: int
    ) -> None:
        """Update graph for a stem pair deletion."""
        node = bg.elements.get(node_name)
        orig = None
        if node is not None and len(node.positions) == 4:
            orig = list(node.positions)
            ls, le, rs, re = orig
            left_index = min(max(left_index, ls - 1), le - 1)
            right_index = min(max(right_index, rs - 1), re - 1)

        if left_index < right_index:
            cls._shift_indices(bg, left_index, -1)
            cls._shift_indices(bg, right_index - 1, -1)
        else:
            cls._shift_indices(bg, right_index, -1)
            cls._shift_indices(bg, left_index - 1, -1)

        if node is None or orig is None:
            return

        ls, le, rs, re = orig
        is_single = ls == le and rs == re

        if is_single:
            # Remove the stem and renumber subsequent stems
            index = int(node_name[1:])
            del bg.elements[node_name]
            i = index + 1
            while f"s{i}" in bg.elements:
                bg.elements[f"s{i-1}"] = bg.elements.pop(f"s{i}")
                i += 1

            # Merge adjacent internal loops when possible
            left_name = f"i{index-2}"
            right_name = f"i{index-1}"
            left_loop = bg.elements.get(left_name)
            right_loop = bg.elements.get(right_name)
            if left_loop and right_loop:
                merged = sorted(set(left_loop.positions + right_loop.positions))
                if len(merged) > 2:
                    merged = [merged[0], merged[-1]]
                left_loop.positions = merged
                cls._recompute_bounds(left_loop)
                del bg.elements[right_name]
                j = index
                while f"i{j}" in bg.elements:
                    bg.elements[f"i{j-1}"] = bg.elements.pop(f"i{j}")
                    j += 1
        else:
            node.positions = [ls, le - 1, rs - 1, re - 2]
            node.start = node.positions[0]
            node.end = node.positions[-1]
