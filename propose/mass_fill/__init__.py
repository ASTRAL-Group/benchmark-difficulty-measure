"""Iterative max-gap mass fill for math benchmarks (AIME×30 / AMC×40)."""

from mass_fill.distribution import (  # noqa: F401
    MassFillState,
    VirtualPoint,
    default_target_count,
    init_state,
    load_state,
    pick_max_gap,
    results_dir,
    save_state,
)
