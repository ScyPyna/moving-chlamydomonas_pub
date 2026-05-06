from __future__ import annotations

import numpy as np
from scipy.stats import sem


def calculate_energy_density(power: float, radius: float, inner_radius: float = 0.0) -> float:
    if inner_radius == 0:
        return power / (np.pi * radius * radius)
    return power / (np.pi * (radius * radius - inner_radius * inner_radius))


def bin_data(x, y, n_bins: int, use_quantile_bins: bool = False):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size == 0:
        return np.array([]), np.array([]), np.array([])

    if use_quantile_bins:
        quantiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(x, quantiles)
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(np.min(x), np.max(x), n_bins + 1)

    if len(bin_edges) < 2:
        return np.array([]), np.array([]), np.array([])

    x_bins = np.digitize(x, bins=bin_edges)

    mean_x = np.full(len(bin_edges) - 1, np.nan)
    mean_y = np.full(len(bin_edges) - 1, np.nan)
    std_y = np.full(len(bin_edges) - 1, np.nan)

    for i in range(1, len(bin_edges)):
        indices = np.where(x_bins == i)[0]
        if len(indices) > 0:
            mean_x[i - 1] = np.mean(x[indices])
            mean_y[i - 1] = np.mean(y[indices])
            std_y[i - 1] = sem(y[indices]) if len(indices) > 1 else 0.0

    return mean_x, mean_y, std_y


def count_points_within_distance(distance_values, signal_values, max_distance: float, num_intervals: int = 20):
    distance_values = np.asarray(distance_values, dtype=float)
    signal_values = np.asarray(signal_values, dtype=float)

    interval_size = max_distance / num_intervals
    centers, count, lower_bounds, upper_bounds = [], [], [], []

    for i in range(num_intervals):
        lower = i * interval_size
        upper = (i + 1) * interval_size
        center = 0.5 * (lower + upper)

        idx = np.where((distance_values >= lower) & (distance_values < upper))[0]

        centers.append(center)
        count.append(len(idx))
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return np.array(centers), np.array(count), np.array(lower_bounds), np.array(upper_bounds)


def count_points_within_distance_normalized(distance_values, signal_values, max_distance: float, num_intervals: int = 20):
    distance_values = np.asarray(distance_values, dtype=float)
    signal_values = np.asarray(signal_values, dtype=float)

    interval_size = max_distance / num_intervals
    centers, count, lower_bounds, upper_bounds = [], [], [], []

    for i in range(num_intervals):
        lower = i * interval_size
        upper = (i + 1) * interval_size
        center = 0.5 * (lower + upper)

        idx = np.where((distance_values >= lower) & (distance_values < upper))[0]
        raw_count = len(idx)
        area = np.pi * (upper**2 - lower**2)
        normalized_count = raw_count / area if area > 0 else np.nan

        centers.append(center)
        count.append(normalized_count)
        lower_bounds.append(lower)
        upper_bounds.append(upper)

    return np.array(centers), np.array(count), np.array(lower_bounds), np.array(upper_bounds)


def hex_to_rgb(hex_color: str):
    return tuple(int(hex_color[i:i+2], 16) / 255 for i in (1, 3, 5))


COLOR_PINK = hex_to_rgb('#FD57E7')
COLOR_ORANGE = hex_to_rgb('#F4D40C')
COLOR_GREEN = hex_to_rgb('#7FF64D')