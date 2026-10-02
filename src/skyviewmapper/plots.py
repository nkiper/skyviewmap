"""Static PNG maps of a region's results (matplotlib, light theme).

Colour follows the job: magnitudes (probability, totality duration) use one
blue ramp, light to dark; the ordered terrain classes "some" / "every typical
spot sees the Sun" take two steps of the same ramp. Cells where terrain
blocks every spot are the same neutral grey on every map, outside the ramp. Text and
outlines use ink colours, never the data ramp.
"""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from numpy.typing import ArrayLike  # noqa: E402
from matplotlib.artist import Artist  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colorbar import Colorbar  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.image import AxesImage  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# Sequential blue ramp, steps 100 -> 700 (reference palette).
BLUE_RAMP = (
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
)
SOME_ALL = ("#5598e7", "#104281")  # ramp steps 350 / 650: some / every typical spot sees the Sun
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
BLOCKED_GREY = "#c3c2b7"

SEQUENTIAL = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)

REFERENCE_TOWNS: dict[str, list[tuple[str, float, float]]] = {
    "iberia": [
        ("A Coruña", 43.36, -8.41), ("Oviedo", 43.36, -5.85), ("Bilbao", 43.26, -2.93),
        ("León", 42.60, -5.57), ("Burgos", 42.34, -3.70), ("Zaragoza", 41.65, -0.89),
        ("Madrid", 40.42, -3.70), ("Valencia", 39.47, -0.38), ("Palma", 39.57, 2.65),
        ("Barcelona", 41.39, 2.17), ("Seville", 37.39, -5.98), ("Lisbon", 38.72, -9.14),
    ],
    "iceland": [
        ("Reykjavík", 64.15, -21.94), ("Ólafsvík", 64.89, -23.71), ("Ísafjörður", 66.07, -23.13),
        ("Akureyri", 65.68, -18.09), ("Vík", 63.42, -19.01), ("Egilsstaðir", 65.27, -14.39),
    ],
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.edgecolor": "#e1e0d9",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def _extent(ds: xr.Dataset) -> tuple[float, float, float, float]:
    lat, lon = ds["lat"].values, ds["lon"].values
    dlat, dlon = lat[1] - lat[0], lon[1] - lon[0]
    return (lon[0] - dlon / 2, lon[-1] + dlon / 2, lat[0] - dlat / 2, lat[-1] + dlat / 2)


def _lon_label(x: float, _pos: int | None = None) -> str:
    return f"{abs(x):g}°{'W' if x < 0 else 'E'}"


def _lat_label(y: float, _pos: int | None = None) -> str:
    return f"{abs(y):g}°{'S' if y < 0 else 'N'}"


def _base_map(ds: xr.Dataset, title: str, subtitle: str) -> tuple[Figure, Axes, Axes]:
    """Figure laid out in inches: title block, map at its true aspect, key row, footer.

    Returns the figure, the map axes and an axes for the colour bar.
    """
    lat = ds["lat"].values
    w, e, s, n = _extent(ds)
    aspect = 1.0 / np.cos(np.radians(lat.mean()))
    left, right, top, bottom = 0.7, 0.25, 0.95, 1.45
    width = 11.0
    map_w = width - left - right
    map_h = map_w * (n - s) * aspect / (e - w)
    height = top + map_h + bottom
    fig = plt.figure(figsize=(width, height), dpi=150)
    ax = fig.add_axes((left / width, bottom / height, map_w / width, map_h / height))
    ax.set_aspect(aspect)
    ax.set_xlim(w, e)
    ax.set_ylim(s, n)
    ax.tick_params(labelsize=8, length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(_lon_label))
    ax.yaxis.set_major_formatter(FuncFormatter(_lat_label))
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes((left / width, 0.85 / height, 0.5 * map_w / width, 0.12 / height))
    fig.text(0.15 / width, 1 - 0.22 / height, title, ha="left", va="top", fontsize=14, weight="bold", color=INK_PRIMARY)
    fig.text(0.15 / width, 1 - 0.55 / height, subtitle, ha="left", va="top", fontsize=9.5, color=INK_SECONDARY)
    return fig, ax, cax


def _image(ax: Axes, data: ArrayLike, **kwargs: Any) -> AxesImage:
    """``imshow`` that keeps the map's latitude-corrected aspect (imshow would reset it to 1)."""
    return ax.imshow(np.ma.masked_invalid(np.asarray(data, dtype=float)), aspect=ax.get_aspect(), **kwargs)


def _overlays(ax: Axes, ds: xr.Dataset, fade_outside_path: bool = True) -> None:
    """Fade land outside the path, then coastline, path outline and towns."""
    lat, lon = ds["lat"].values, ds["lon"].values
    land = np.isfinite(ds["best_margin_deg"].values)
    path = ds["in_totality"].values.astype(bool)
    if fade_outside_path:
        outside = np.where(land & ~path, 1.0, np.nan)
        _image(ax, (outside), origin="lower", extent=_extent(ds),
                  cmap=ListedColormap([SURFACE]), vmin=0, vmax=1, alpha=0.6, interpolation="nearest")
    ax.contour(lon, lat, land.astype(float), levels=[0.5], colors=INK_MUTED, linewidths=0.6)
    ax.contour(lon, lat, path.astype(float), levels=[0.5], colors=INK_PRIMARY, linewidths=1.3, linestyles="--")
    w, e, s, n = _extent(ds)
    for name, la, lo in REFERENCE_TOWNS.get(str(ds.attrs.get("region", "")), []):
        if w < lo < e and s < la < n:
            ax.plot(lo, la, "o", ms=3.5, color=INK_PRIMARY, mec=SURFACE, mew=1.0, zorder=5)
            ax.annotate(name, (lo, la), xytext=(4, 3), textcoords="offset points", fontsize=8,
                        color=INK_PRIMARY, zorder=5,
                        bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.75})


def _key(fig: Figure, ax: Axes, handles: list[Artist]) -> None:
    """Legend in the key row, right-aligned under the map."""
    box = ax.get_position()
    fig.legend(handles=handles, loc="lower right", bbox_to_anchor=(box.x1, 0.0), bbox_transform=fig.transFigure,
               fontsize=8, frameon=False, borderaxespad=2.2)


def _path_handles(fade: bool = True) -> list[Artist]:
    handles: list[Artist] = [Line2D([], [], color=INK_PRIMARY, lw=1.3, ls="--", label="Edge of the path of totality")]
    if fade:
        handles.append(Patch(facecolor=BLUE_RAMP[4], alpha=0.4, label="Faded: outside the path"))
    return handles


def _colorbar(fig: Figure, im: AxesImage, cax: Axes, label: str) -> Colorbar:
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.ax.tick_params(labelsize=8, length=0, colors=INK_SECONDARY)
    cb.outline.set_visible(False)
    cb.set_label(label, fontsize=9, color=INK_SECONDARY)
    return cb


def _footer(fig: Figure, ds: xr.Dataset, extra: str = "") -> None:
    note = f"Cloud: {ds.attrs.get('cloud_data', 'ERA5')}. Terrain: {ds.attrs.get('terrain_data', 'DEM')}. {extra}"
    fig.text(0.01, 0.01, note, ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)


def plot_probability(ds: xr.Dataset, var: str, path: Path, title: str, subtitle: str) -> Path:
    fig, ax, cax = _base_map(ds, title, subtitle)
    cmap = SEQUENTIAL.with_extremes(bad=SURFACE)
    im = _image(ax, (ds[var].values), origin="lower", extent=_extent(ds), cmap=cmap,
                   vmin=0, vmax=1, interpolation="nearest")
    blocked = np.where(ds["terrain_blocked"].values.astype(bool), 1.0, np.nan)
    _image(ax, (blocked), origin="lower", extent=_extent(ds),
              cmap=ListedColormap([BLOCKED_GREY]), vmin=0, vmax=1, interpolation="nearest")
    _overlays(ax, ds)
    cb = _colorbar(fig, im, cax, "Chance of a clear view of the eclipse")
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{round(t * 100)}%" for t in ticks])
    _key(fig, ax, [Patch(color=BLOCKED_GREY, label="Terrain hides the Sun from every spot"), *_path_handles()])
    _footer(fig, ds)
    return _save(fig, path)


def plot_totality(ds: xr.Dataset, path: Path) -> Path:
    fig, ax, cax = _base_map(ds, "Duration of totality", "Seconds of total eclipse at each place; blank outside the path")
    tot = ds["totality_s"].values.astype(float)
    cmap = SEQUENTIAL.with_extremes(bad=SURFACE)
    vmax = float(np.nanmax(tot)) if np.nanmax(tot) > 0 else 1.0
    im = _image(ax, (np.where(tot > 0, tot, np.nan)), origin="lower", extent=_extent(ds),
                   cmap=cmap, vmin=0, vmax=vmax, interpolation="nearest")
    _overlays(ax, ds, fade_outside_path=False)
    _colorbar(fig, im, cax, "Totality (seconds)")
    _key(fig, ax, _path_handles(fade=False))
    _footer(fig, ds)
    return _save(fig, path)


def plot_terrain(ds: xr.Dataset, path: Path) -> Path:
    fig, ax, cax = _base_map(ds, "Does the terrain hide the Sun?",
                             "Of the spots tested in each ~1 km cell (9 typical + the highest point), "
                             "how many see the Sun at maximum eclipse")
    cax.set_visible(False)
    frac = ds["clear_fraction"].values
    best = ds["best_margin_deg"].values
    cls = np.full(frac.shape, np.nan)
    land = np.isfinite(best)
    cls[land & (best <= 0)] = 0  # no spot
    cls[land & (best > 0) & ~(frac >= 1)] = 1  # some spots
    cls[land & (best > 0) & (frac >= 1)] = 2  # every typical spot
    cmap = ListedColormap([BLOCKED_GREY, *SOME_ALL]).with_extremes(bad=SURFACE)
    _image(ax, (cls), origin="lower", extent=_extent(ds), cmap=cmap, vmin=-0.5, vmax=2.5,
              interpolation="nearest")
    _overlays(ax, ds)
    labels = ("No spot sees the Sun", "Some spots see it", "Every typical spot sees it")
    handles: list[Artist] = [Patch(color=c, label=lab) for c, lab in zip((BLOCKED_GREY, *SOME_ALL), labels)]
    _key(fig, ax, handles + _path_handles())
    _footer(fig, ds, "Eye height 2 m; standard refraction.")
    return _save(fig, path)


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def write_maps(ds: xr.Dataset, out_dir: Path) -> list[Path]:
    region = str(ds.attrs.get("region", "region"))
    label = region.capitalize()
    return [
        plot_probability(
            ds, "p_clear_view", out_dir / f"map_p_clear_view_{region}.png",
            f"{label}: chance of seeing the 12 Aug 2026 eclipse",
            "Cloud along the sight line (with slant-path correction) × somewhere in the ~1 km cell sees the Sun over the terrain",
        ),
        plot_probability(
            ds, "p_clear_view_no_slant", out_dir / f"map_p_clear_view_no_slant_{region}.png",
            f"{label}: chance of seeing the eclipse (optimistic)",
            "As the headline map, but without the slant-path cloud correction",
        ),
        plot_totality(ds, out_dir / f"map_totality_{region}.png"),
        plot_terrain(ds, out_dir / f"map_terrain_{region}.png"),
    ]
