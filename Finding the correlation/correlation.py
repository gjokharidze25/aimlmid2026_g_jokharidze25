#!/usr/bin/env python3

import math
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# --- 1) Put your points here -------------------------------------------------
points: List[Tuple[float, float]] = [
    (-10, -10),
    (-5,  -5),
    (-3,  -1),
    (-5,   2),
    (-1,   1),
    (3,    1),
    (1,   -2),
    (5,   -3),
    (7,   -2),
]
# -----------------------------------------------------------------------------

def pearson_r_manual(xs: List[float], ys: List[float]) -> float:
    """Pearson r using the standard centered formula."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("xs and ys must have the same length and contain at least 2 points.")

    x_bar = sum(xs) / len(xs)
    y_bar = sum(ys) / len(ys)

    num = sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
    den_x = sum((x - x_bar) ** 2 for x in xs)
    den_y = sum((y - y_bar) ** 2 for y in ys)
    den = math.sqrt(den_x * den_y)

    if den == 0:
        raise ValueError("Zero variance in x or y; correlation is undefined.")

    return num / den


def main() -> None:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Manual Pearson r
    r_manual = pearson_r_manual(xs, ys)

    # NumPy Pearson r (same result)
    r_np = float(np.corrcoef(xs, ys)[0, 1])

    print("Data points:", points)
    print(f"Pearson r (manual): {r_manual:.6f}")
    print(f"Pearson r (NumPy) : {r_np:.6f}")

    # Scatter plot
    plt.figure(figsize=(7, 5))
    plt.scatter(xs, ys)
    plt.title("Scatter Plot")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("scatter.png", dpi=200)
    print("Saved plot to scatter.png")


if __name__ == "__main__":
    main()

