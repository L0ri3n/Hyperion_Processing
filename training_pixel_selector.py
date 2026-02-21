"""
training_pixel_selector.py — Interactive Training Pixel Selection GUI
======================================================================

Interactive matplotlib window for visually labelling training pixels in a
hyperspectral EO-1 Hyperion reflectance cube.  Results are saved as a .npz
file that maps each class name to a numpy integer array of (row, col) pixel
indices restricted to the soil mask.

This selector is configured for **binary classification**:

  * AMD_FeOx  — AMD ferro-oxide lithology (goethite, hematite, jarosite,
                schwertmannite and other Fe³⁺ secondary minerals collectively).
                Select areas visually dominated by rusty, orange-brown or
                yellow-ochre colour; high Fe-Oxide Ratio values.
  * Background — unaltered rock, waste rock, bare soil, or any area with no
                clear AMD iron-oxide signature.

Usage
-----
    python training_pixel_selector.py          # standalone
    # or imported:
    from training_pixel_selector import TrainingPixelSelector
    sel = TrainingPixelSelector("path/to/image.hdr")
    sel.show()

Controls
--------
* Left-click + drag on the image to draw a rectangular ROI.
* Use the View radio buttons to switch between composites.
  The Fe-Oxide Ratio view (900 / 660 nm) is most diagnostic for AMD_FeOx.
* Use the Class radio buttons to select AMD_FeOx or Background.
* "Toggle Soil Mask" shows/hides the semi-transparent yellow overlay.
* "Clear Class" removes all ROIs for the currently selected class.
* "Clear All" removes every drawn ROI.
* "Save & Continue" serialises training pixels and closes the window.

Output
------
training_pixels.npz — one key per class; value is a (n, 2) int32 array
                       of [row, col] indices for soil-masked pixels inside
                       the drawn rectangles.

Notes
-----
* Only pixels whose (row, col) falls inside the binary soil mask are included.
* A soil_mask.npy file is expected in the same directory as the .hdr file.
  If not found, a valid-data mask (non-zero spectral norm) is used as fallback.
* The reflectance cube is assumed to be stored as scaled integers (/10000) if
  any value exceeds 2.0; otherwise values are taken as-is (already [0, 1]).
"""

import numpy as np
import spectral as sp
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import RectangleSelector, RadioButtons, Button
from pathlib import Path
from scipy.ndimage import binary_dilation

# =============================================================================
# CONFIGURATION
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
HDR_FILE = str(
    BASE_DIR / "amd_mapping" / "data" / "hyperion"
    / "EO1H2020342013284110KF_reflectance.hdr"
)
OUTPUT_FILE = str(
    BASE_DIR / "amd_mapping" / "outputs" / "training_pixels.npz"
)

# Wavelengths (nm) used for composite construction and band-index extraction.
KEY_WAVELENGTHS_NM = [480, 550, 660, 850, 900]

# Diagnostic wavelengths shown as vertical reference lines on the spectral
# profile panel.  Values are in nm; dict values are short display labels.
# These bands are diagnostic for Fe³⁺ secondary minerals (AMD_FeOx class).
DIAG_WAVELENGTHS = {
    430: "Fe³⁺ abs",   # Fe³⁺ charge-transfer absorption (jarosite/hematite)
    660: "Fe CT",      # Fe charge-transfer shoulder
    875: "Fe³⁺ CFT",  # Fe³⁺ crystal-field transition (goethite/hematite)
}

# Binary training classes for the AMD ferro-oxide lithology classifier.
# AMD_FeOx  — any pixel dominated by iron-oxide secondary AMD minerals.
# Background — unaltered rock, waste, or soil with no AMD FeOx signature.
AMD_CLASSES = [
    "AMD_FeOx",
    "Background",
]

# Per-class display colours (hex).
# AMD_FeOx uses a rust/burnt-orange colour evocative of iron-oxide mineralogy.
CLASS_COLORS = {
    "AMD_FeOx":   "#C05A0A",   # rust / burnt orange
    "Background": "#7F7F7F",   # neutral grey
}

# Hyperion reflectance scale factor (raw integers → [0, 1])
REFLECTANCE_SCALE = 10000.0

# Dark-background rcParams applied globally before building the figure.
DARK_THEME = {
    "figure.facecolor":  "#1a1a2e",
    "axes.facecolor":    "#16213e",
    "axes.edgecolor":    "#777799",
    "axes.labelcolor":   "#ccccdd",
    "xtick.color":       "#999999",
    "ytick.color":       "#999999",
    "text.color":        "#ddddee",
    "grid.color":        "#2a2a4a",
    "grid.alpha":        0.6,
    "axes.titlecolor":   "#ddddee",
    "legend.facecolor":  "#1a1a2e",
    "legend.edgecolor":  "#555577",
    "legend.labelcolor": "#ccccdd",
}

# =============================================================================
# HELPERS
# =============================================================================


def _find_band(wavelengths, target_nm):
    """Return the index of the wavelength channel closest to *target_nm* (nm).

    Parameters
    ----------
    wavelengths : ndarray (bands,)
    target_nm : float

    Returns
    -------
    int
    """
    return int(np.argmin(np.abs(wavelengths - target_nm)))


def _pct_stretch(arr, pct=(2, 98)):
    """Apply 2nd–98th percentile linear stretch; returns values in [0, 1].

    Only finite, positive values are used to compute the percentile range so
    that nodata zeros (common in Hyperion scenes) do not bias the stretch.

    Parameters
    ----------
    arr : ndarray (rows, cols), float
    pct : tuple (lo_pct, hi_pct)

    Returns
    -------
    ndarray (rows, cols), float32
    """
    finite_pos = arr[np.isfinite(arr) & (arr > 0)]
    if len(finite_pos) == 0:
        return np.zeros_like(arr, dtype=np.float32)
    lo, hi = np.percentile(finite_pos, pct)
    return np.clip((arr - lo) / (hi - lo + 1e-10), 0, 1).astype(np.float32)


def _stretch_rgb(rgb):
    """Per-channel 2–98 % percentile stretch for a 3-channel array.

    Parameters
    ----------
    rgb : ndarray (rows, cols, 3), float

    Returns
    -------
    ndarray (rows, cols, 3), float32
    """
    out = np.empty_like(rgb, dtype=np.float32)
    for ch in range(3):
        out[:, :, ch] = _pct_stretch(rgb[:, :, ch].astype(np.float64))
    return out


def _style_radio(radio):
    """Apply dark-theme colouring to a RadioButtons widget.

    Works with matplotlib >= 3.5.  The ``circles`` attribute check is
    wrapped in a try/except for forward-compatibility with API changes.

    Parameters
    ----------
    radio : matplotlib.widgets.RadioButtons
    """
    try:
        for circle in radio.circles:
            circle.set_facecolor("#2a2a4a")
            circle.set_edgecolor("#9999cc")
    except AttributeError:
        pass  # newer matplotlib may expose circles differently
    for label in radio.labels:
        label.set_color("#ccccdd")
        label.set_fontsize(8.5)
    radio.ax.set_facecolor("#16213e")


def load_cube_and_mask(hdr_file):
    """Load the reflectance cube and soil mask from *hdr_file*.

    The soil mask is expected as ``soil_mask.npy`` in the same directory as
    the .hdr file.  If the file is absent a valid-data mask (all pixels whose
    spectral L2-norm > 0) is used and a warning is printed.

    Parameters
    ----------
    hdr_file : str or Path
        Path to the ENVI .hdr file of the reflectance cube.

    Returns
    -------
    cube : ndarray (rows, cols, bands), float32  — reflectance [0, 1]
    wavelengths : ndarray (bands,), float         — wavelengths in nm
    soil_mask : ndarray (rows, cols), bool
    """
    print(f"  Loading image: {Path(hdr_file).name}")
    img = sp.open_image(str(hdr_file))
    cube = img.load().astype(np.float32)
    wavelengths = np.array(img.metadata["wavelength"], dtype=float)

    print(f"  Shape: {cube.shape[0]} rows × {cube.shape[1]} cols × {cube.shape[2]} bands")
    print(f"  Wavelength range: {wavelengths.min():.0f}–{wavelengths.max():.0f} nm")

    # Scale to reflectance if values are stored as scaled integers
    if cube.max() > 2.0:
        cube /= REFLECTANCE_SCALE
        print(f"  Scaled by 1/{REFLECTANCE_SCALE:.0f} → reflectance range "
              f"[{cube.min():.4f}, {cube.max():.4f}]")

    # --- soil mask ---
    mask_path = Path(hdr_file).parent / "soil_mask.npy"
    if mask_path.exists():
        soil_mask = np.load(str(mask_path)).astype(bool)
        n_soil = int(soil_mask.sum())
        print(f"  Loaded soil mask ({n_soil:,} soil pixels) from {mask_path.name}")
    else:
        # Fallback: pixels with non-zero spectral norm
        norms = np.linalg.norm(
            cube.reshape(-1, cube.shape[2]), axis=1
        )
        soil_mask = (norms > 0).reshape(cube.shape[0], cube.shape[1])
        print("  WARNING: soil_mask.npy not found — using valid-data mask as "
              "fallback.  Run the SAM workflow first to generate soil_mask.npy.")

    return cube, wavelengths, soil_mask


# =============================================================================
# MAIN CLASS
# =============================================================================


class TrainingPixelSelector:
    """Interactive matplotlib GUI for labelling training pixels.

    The window displays three switchable image composites (RGB, NIR false
    colour, and iron oxide ratio), a live spectral profile panel, and a suite
    of matplotlib.widgets controls.  The user draws rectangular ROIs on the
    image; pixels within the soil mask are collected and saved to a .npz file.

    Parameters
    ----------
    hdr_file : str or Path
        Path to the ENVI .hdr file of the reflectance cube.
    output_file : str or Path, optional
        Destination .npz file for the training pixel dictionary.
        Defaults to ``OUTPUT_FILE`` (defined at module level).
    """

    def __init__(self, hdr_file, output_file=None):
        # Apply dark theme before creating any figure
        plt.rcParams.update(DARK_THEME)

        # ----- load data -------------------------------------------------
        self.cube, self.wavelengths, self.soil_mask = load_cube_and_mask(
            hdr_file
        )
        self.rows, self.cols, self.bands = self.cube.shape
        self.output_file = output_file or OUTPUT_FILE

        # Band indices for the five key wavelengths
        self.band_idx = {
            nm: _find_band(self.wavelengths, nm) for nm in KEY_WAVELENGTHS_NM
        }
        print("  Key band indices (wavelength → band):")
        for nm in KEY_WAVELENGTHS_NM:
            actual = float(self.wavelengths[self.band_idx[nm]])
            print(f"    {nm} nm → band {self.band_idx[nm]} ({actual:.1f} nm)")

        # ----- GUI state -------------------------------------------------
        self.current_view = "RGB"       # 'RGB' | 'NIR' | 'Iron_Oxide'
        self.current_class = AMD_CLASSES[0]
        self.overlay_visible = True

        # ROI storage: {class_name -> [[r1,c1,r2,c2], ...]}
        self.rectangles = {cls: [] for cls in AMD_CLASSES}

        # matplotlib patch / text handles for each ROI rectangle
        self.rect_patches = {cls: [] for cls in AMD_CLASSES}
        self.rect_labels  = {cls: [] for cls in AMD_CLASSES}

        # Spectral fill_between artist (replaced on each ROI selection)
        self.std_fill = None

        # Throttle hover events: only redraw if the pixel changed
        self._last_hover_pos = (-1, -1)

        # ----- precompute composites and overlay arrays ------------------
        self._precompute_composites()

        # ----- build the figure ------------------------------------------
        self._setup_figure()

    # ------------------------------------------------------------------
    # Composite pre-computation
    # ------------------------------------------------------------------

    def _precompute_composites(self):
        """Precompute the three image composites and RGBA overlay arrays.

        Composites are stored as float32 arrays in ``self.composites``:

        * ``'RGB'``        — stretched 3-channel array (rows, cols, 3)
        * ``'NIR'``        — stretched 3-channel array (rows, cols, 3)
        * ``'Iron_Oxide'`` — stretched single-band greyscale (rows, cols)

        Two pre-built RGBA layers are also created:

        * ``self._boundary_rgba`` — thin white border of the soil mask, used
          only in Iron Oxide view.
        * ``self._soil_overlay_rgba`` — semi-transparent yellow fill of soil
          pixels, toggled by the user.
        """
        b = self.band_idx
        self.composites = {}

        # ---- RGB: 660 / 550 / 480 nm ------------------------------------
        rgb = np.stack([
            self.cube[:, :, b[660]].astype(np.float64),
            self.cube[:, :, b[550]].astype(np.float64),
            self.cube[:, :, b[480]].astype(np.float64),
        ], axis=2)
        self.composites["RGB"] = _stretch_rgb(rgb)

        # ---- NIR false colour: 850 / 660 / 550 nm -----------------------
        nir = np.stack([
            self.cube[:, :, b[850]].astype(np.float64),
            self.cube[:, :, b[660]].astype(np.float64),
            self.cube[:, :, b[550]].astype(np.float64),
        ], axis=2)
        self.composites["NIR"] = _stretch_rgb(nir)

        # ---- Iron oxide ratio: band_900 / band_660 ----------------------
        # High ratio → strong Fe³⁺ absorption relative to red reflectance.
        # Diagnostic for goethite, hematite, and jarosite.
        denom = self.cube[:, :, b[660]].astype(np.float64)
        numer = self.cube[:, :, b[900]].astype(np.float64)
        ratio = numer / (denom + 1e-10)
        self.composites["Iron_Oxide"] = _pct_stretch(ratio)

        # ---- soil mask boundary (white, 1 px) ---------------------------
        # Outer boundary = dilation minus the original mask.
        boundary = binary_dilation(self.soil_mask) & ~self.soil_mask
        self._boundary_rgba = np.zeros(
            (*self.soil_mask.shape, 4), dtype=np.float32
        )
        self._boundary_rgba[boundary] = [1.0, 1.0, 1.0, 0.85]

        # ---- soil mask overlay (yellow, 25 % alpha) ---------------------
        self._soil_overlay_rgba = np.zeros(
            (*self.soil_mask.shape, 4), dtype=np.float32
        )
        self._soil_overlay_rgba[self.soil_mask] = [1.0, 1.0, 0.0, 0.25]

    # ------------------------------------------------------------------
    # Figure construction
    # ------------------------------------------------------------------

    def _setup_figure(self):
        """Create the figure layout, image layers, spectral panel, and widgets."""
        fig = plt.figure(figsize=(19, 11))
        fig.suptitle(
            "Training Pixel Selector — AMD Ferro-oxide Lithology (binary: AMD_FeOx / Background)",
            fontsize=11, fontweight="bold", color="#ddddee", y=0.985
        )
        self.fig = fig

        # ── Main panels ────────────────────────────────────────────────
        # ax_img:  left 53 %, top 73 % of figure
        # ax_spec: right 40 %, top 73 % of figure
        self.ax_img  = fig.add_axes([0.02, 0.22, 0.52, 0.74])
        self.ax_spec = fig.add_axes([0.57, 0.22, 0.41, 0.74])

        # ── Image: three layered imshow artists ────────────────────────
        # Layer 1: base composite image (RGB / NIR / grayscale)
        self.img_artist = self.ax_img.imshow(
            self.composites["RGB"],
            interpolation="nearest", aspect="auto", zorder=1
        )
        # Layer 2: soil mask boundary (white contour, Iron Oxide view only)
        self.boundary_artist = self.ax_img.imshow(
            self._boundary_rgba,
            interpolation="nearest", aspect="auto", zorder=2, visible=False
        )
        # Layer 3: soil mask overlay (yellow semi-transparent, toggleable)
        self.overlay_artist = self.ax_img.imshow(
            self._soil_overlay_rgba,
            interpolation="nearest", aspect="auto", zorder=3, visible=True
        )

        self.ax_img.set_title(
            "RGB Composite  (660 / 550 / 480 nm)", fontsize=10
        )
        self.ax_img.axis("off")

        # ── Spectral profile panel ─────────────────────────────────────
        self._setup_spec_panel()

        # ── RectangleSelector on image axis ────────────────────────────
        # interactive=False → fires callback on mouse-release only; the
        # temporary selection rectangle is drawn while dragging.
        self.rect_selector = RectangleSelector(
            self.ax_img,
            self._on_rect_select,
            useblit=True,
            button=[1],            # left mouse button only
            minspanx=2,
            minspany=2,
            spancoords="data",
            interactive=False,
            props=dict(
                facecolor="none", edgecolor="white",
                alpha=0.9, linewidth=1.5, linestyle="--"
            ),
        )

        # ── Widget axes ────────────────────────────────────────────────
        # View selector — 3 options
        ax_rview = fig.add_axes([0.02, 0.04, 0.14, 0.15])
        ax_rview.set_title("View", fontsize=8, color="#ccccdd", pad=2)
        ax_rview.set_facecolor("#16213e")
        self.radio_view = RadioButtons(
            ax_rview,
            ["RGB", "NIR False Colour", "Fe-Oxide Ratio"],
            active=0,
            activecolor="#17BECF",
        )
        self.radio_view.on_clicked(self._on_view_change)
        _style_radio(self.radio_view)

        # Class selector — 2 classes (AMD_FeOx / Background)
        ax_rclass = fig.add_axes([0.18, 0.10, 0.14, 0.10])
        ax_rclass.set_title("Class", fontsize=8, color="#ccccdd", pad=2)
        ax_rclass.set_facecolor("#16213e")
        self.radio_class = RadioButtons(
            ax_rclass,
            AMD_CLASSES,
            active=0,
            activecolor=CLASS_COLORS[AMD_CLASSES[0]],
        )
        self.radio_class.on_clicked(self._on_class_change)
        _style_radio(self.radio_class)

        # Buttons
        btn_kw = dict(color="#16213e", hovercolor="#1e3060")

        ax_btoggle = fig.add_axes([0.34, 0.14, 0.13, 0.06])
        self.btn_toggle = Button(ax_btoggle, "Toggle Soil Mask", **btn_kw)
        self.btn_toggle.label.set_color("#ccccdd")
        self.btn_toggle.label.set_fontsize(8.5)
        self.btn_toggle.on_clicked(self._toggle_overlay)

        ax_bclear_cls = fig.add_axes([0.34, 0.07, 0.13, 0.06])
        self.btn_clear_cls = Button(ax_bclear_cls, "Clear Class", **btn_kw)
        self.btn_clear_cls.label.set_color("#ccccdd")
        self.btn_clear_cls.label.set_fontsize(8.5)
        self.btn_clear_cls.on_clicked(self._clear_class)

        ax_bclear_all = fig.add_axes([0.49, 0.14, 0.13, 0.06])
        self.btn_clear_all = Button(
            ax_bclear_all, "Clear All",
            color="#2a1010", hovercolor="#3a2020"
        )
        self.btn_clear_all.label.set_color("#ffaaaa")
        self.btn_clear_all.label.set_fontsize(8.5)
        self.btn_clear_all.on_clicked(self._clear_all)

        ax_bsave = fig.add_axes([0.49, 0.02, 0.21, 0.11])
        self.btn_save = Button(
            ax_bsave, "Save & Continue",
            color="#0a3d0a", hovercolor="#155215"
        )
        self.btn_save.label.set_color("#80ff80")
        self.btn_save.label.set_fontsize(11)
        self.btn_save.label.set_fontweight("bold")
        self.btn_save.on_clicked(self._save_and_continue)

        # Status / info text (bottom-right of widget strip)
        self.status_text = fig.text(
            0.755, 0.10,
            "Draw rectangles on the image.\n"
            "Fe-Oxide Ratio view is most diagnostic\n"
            "for AMD_FeOx selection.",
            ha="center", va="center",
            fontsize=8, color="#888899",
            linespacing=1.5,
        )

        # ── Connect events ─────────────────────────────────────────────
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # ------------------------------------------------------------------
    # Spectral panel initialisation
    # ------------------------------------------------------------------

    def _setup_spec_panel(self):
        """Initialise the spectral profile panel.

        Creates:
        * Permanent diagnostic reference lines at 430, 660, 875 nm.
        * One faint background line per AMD class (updated as ROIs are drawn).
        * A solid 'main' line for the selected ROI mean spectrum.
        * A thin dotted 'hover' line for the pixel under the cursor.
        """
        ax = self.ax_spec

        # Diagnostic reference lines (created once; never removed)
        for wl, label in DIAG_WAVELENGTHS.items():
            ax.axvline(
                wl, color="#4a4a6a", linestyle="--",
                linewidth=1.0, alpha=0.8, zorder=1
            )
            ax.text(
                wl + 7, 0.97, label,
                color="#7777aa", fontsize=6.5,
                ha="left", va="top",
                transform=ax.get_xaxis_transform(),
                zorder=1,
            )

        # Background class mean spectra (faint, colour-coded by class)
        # These update whenever a new ROI is drawn.
        self.bg_lines = {}
        for cls in AMD_CLASSES:
            (line,) = ax.plot(
                [], [],
                color=CLASS_COLORS[cls],
                alpha=0.30, linewidth=1.1,
                linestyle="-", zorder=2,
                label=f"_{cls}",   # leading underscore → hidden in legend
            )
            self.bg_lines[cls] = line

        # Main line: mean spectrum of the most recent ROI (solid, class-coloured)
        (self.main_line,) = ax.plot(
            [], [],
            color="white", linewidth=2.2,
            zorder=5, label="Selected ROI",
        )

        # Hover line: single-pixel spectrum (thin dotted, neutral)
        (self.hover_line,) = ax.plot(
            [], [],
            color="#888899", linewidth=0.9,
            linestyle=":", zorder=4,
            label="Hover pixel",
        )

        ax.set_xlim(self.wavelengths.min(), self.wavelengths.max())
        ax.set_ylim(0, 0.65)
        ax.set_xlabel("Wavelength (nm)", fontsize=9)
        ax.set_ylabel("Reflectance", fontsize=9)
        ax.set_title("Spectral Profile", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(
            fontsize=7.5, loc="upper right",
            framealpha=0.25, edgecolor="#444466",
        )

    # ------------------------------------------------------------------
    # Event callbacks
    # ------------------------------------------------------------------

    def _on_view_change(self, label):
        """Switch the displayed composite when a View radio button is clicked."""
        view_map = {
            "RGB":            "RGB",
            "NIR False Colour": "NIR",
            "Fe-Oxide Ratio": "Iron_Oxide",
        }
        self.current_view = view_map[label]
        self._refresh_image()

    def _on_class_change(self, label):
        """Set the active mineral class and update the radio button colour."""
        self.current_class = label
        # Update active colour to match the class colour
        try:
            self.radio_class.activecolor = CLASS_COLORS[label]
        except Exception:
            pass
        self._update_status(f"Active class: {label}")

    def _on_rect_select(self, eclick, erelease):
        """Handle completion of a drawn ROI rectangle.

        Converts axes data coordinates to integer pixel (row, col) indices,
        clamps to image bounds, stores the rectangle, and draws a styled
        patch with a class label.  Updates the spectral profile for the
        selected region.

        Parameters
        ----------
        eclick : MouseEvent   — position at mouse-press
        erelease : MouseEvent — position at mouse-release
        """
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        if None in (x1, y1, x2, y2):
            return

        # imshow maps: x-axis → columns, y-axis → rows
        c1, c2 = int(min(x1, x2)), int(max(x1, x2))
        r1, r2 = int(min(y1, y2)), int(max(y1, y2))

        # Clamp to valid image bounds
        r1 = max(0, min(r1, self.rows - 1))
        r2 = max(0, min(r2, self.rows - 1))
        c1 = max(0, min(c1, self.cols - 1))
        c2 = max(0, min(c2, self.cols - 1))

        if r1 >= r2 or c1 >= c2:
            return   # degenerate selection; ignore

        cls = self.current_class
        self.rectangles[cls].append([r1, c1, r2, c2])

        # Draw a semi-transparent rectangle with a class label
        color = CLASS_COLORS[cls]
        patch = mpatches.Rectangle(
            (c1, r1), c2 - c1, r2 - r1,
            linewidth=1.8, edgecolor=color,
            facecolor=color, alpha=0.22,
            zorder=10,
        )
        self.ax_img.add_patch(patch)

        lbl = self.ax_img.text(
            (c1 + c2) / 2.0, (r1 + r2) / 2.0,
            cls.replace("_", " "),
            color="white", fontsize=6.5,
            ha="center", va="center", fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=color, alpha=0.55, edgecolor="none",
            ),
            zorder=11,
        )

        self.rect_patches[cls].append(patch)
        self.rect_labels[cls].append(lbl)

        # Update spectral profile and background class lines
        n_soil = self._update_spec_from_roi(r1, c1, r2, c2, cls)
        self._update_status(
            f"{cls}: {n_soil} soil px in last ROI\n"
            f"Total ROIs: {len(self.rectangles[cls])}"
        )

        self.fig.canvas.draw_idle()

    def _on_mouse_move(self, event):
        """Update the hover spectrum line when the cursor moves over the image.

        Only redraws if the pixel position has changed to avoid lag.
        """
        if event.inaxes is not self.ax_img:
            return
        if event.xdata is None or event.ydata is None:
            return

        col = int(round(event.xdata))
        row = int(round(event.ydata))

        if (row, col) == self._last_hover_pos:
            return
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return

        self._last_hover_pos = (row, col)
        spec = self.cube[row, col, :].astype(np.float64)
        self.hover_line.set_data(self.wavelengths, spec)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Button callbacks
    # ------------------------------------------------------------------

    def _toggle_overlay(self, event):
        """Show or hide the semi-transparent soil mask overlay."""
        self.overlay_visible = not self.overlay_visible
        self.overlay_artist.set_visible(self.overlay_visible)
        state = "ON" if self.overlay_visible else "OFF"
        self._update_status(f"Soil mask overlay: {state}")
        self.fig.canvas.draw_idle()

    def _clear_class(self, event):
        """Remove all ROI rectangles for the currently selected class."""
        cls = self.current_class
        for patch in self.rect_patches[cls]:
            patch.remove()
        for txt in self.rect_labels[cls]:
            txt.remove()
        self.rect_patches[cls].clear()
        self.rect_labels[cls].clear()
        self.rectangles[cls].clear()
        self.bg_lines[cls].set_data([], [])
        self._update_status(f"Cleared all ROIs for {cls}")
        self.fig.canvas.draw_idle()

    def _clear_all(self, event):
        """Remove every drawn ROI for all classes."""
        for cls in AMD_CLASSES:
            for patch in self.rect_patches[cls]:
                patch.remove()
            for txt in self.rect_labels[cls]:
                txt.remove()
            self.rect_patches[cls].clear()
            self.rect_labels[cls].clear()
            self.rectangles[cls].clear()
            self.bg_lines[cls].set_data([], [])

        if self.std_fill is not None:
            self.std_fill.remove()
            self.std_fill = None
        self.main_line.set_data([], [])
        self.main_line.set_label("Selected ROI")
        self.hover_line.set_data([], [])
        self._update_status("All ROIs cleared")
        self.fig.canvas.draw_idle()

    def _save_and_continue(self, event):
        """Serialise training pixels to .npz and close the window.

        For each AMD class, all pixels within drawn rectangles that also fall
        inside the soil mask are collected.  Duplicate (row, col) pairs (from
        overlapping rectangles) are de-duplicated.

        The resulting .npz file has one key per class; the value is an int32
        array of shape (n_pixels, 2) with columns [row, col].

        A summary table is printed to the terminal showing pixel counts.
        """
        training_data = {}
        summary_rows  = []

        for cls in AMD_CLASSES:
            pixel_set = set()   # use a set to avoid duplicates from overlapping ROIs

            for r1, c1, r2, c2 in self.rectangles[cls]:
                rr = np.arange(r1, r2 + 1)
                cc = np.arange(c1, c2 + 1)
                rr_g, cc_g = np.meshgrid(rr, cc, indexing="ij")
                rr_f = rr_g.ravel()
                cc_f = cc_g.ravel()
                valid = self.soil_mask[rr_f, cc_f]
                for r, c in zip(rr_f[valid].tolist(), cc_f[valid].tolist()):
                    pixel_set.add((int(r), int(c)))

            if pixel_set:
                arr = np.array(sorted(pixel_set), dtype=np.int32)
            else:
                arr = np.empty((0, 2), dtype=np.int32)

            training_data[cls] = arr
            summary_rows.append({"Class": cls, "Pixels": len(pixel_set)})

        # Save
        out_path = Path(self.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), **training_data)

        # Print terminal summary
        print("\n" + "=" * 55)
        print("TRAINING PIXEL SELECTION SUMMARY")
        print("=" * 55)
        print(f"  {'Class':<22}  {'Pixels':>8}  {'ROIs':>5}")
        print("  " + "-" * 40)
        total = 0
        for row_d in summary_rows:
            cls = row_d["Class"]
            n_px = row_d["Pixels"]
            n_rois = len(self.rectangles[cls])
            flag = ""
            if n_px == 0 and n_rois > 0:
                flag = "  ← no soil pixels in ROI(s)"
            elif n_px == 0:
                flag = "  *"
            print(f"  {cls:<22}  {n_px:>8,}  {n_rois:>5}{flag}")
            total += n_px
        print("  " + "-" * 40)
        print(f"  {'Total':<22}  {total:>8,}")
        print("=" * 55)
        print(f"\nSaved: {out_path.resolve()}")
        if any(r["Pixels"] == 0 for r in summary_rows):
            print("\n  * Classes marked with * have 0 pixels and will not "
                  "appear in the training set.")

        plt.close(self.fig)

    # ------------------------------------------------------------------
    # Display update helpers
    # ------------------------------------------------------------------

    def _refresh_image(self):
        """Redraw the image artist and overlay visibility for the current view."""
        if self.current_view == "Iron_Oxide":
            gray = self.composites["Iron_Oxide"]
            display = np.stack([gray, gray, gray], axis=2)
        else:
            display = self.composites[self.current_view]

        self.img_artist.set_data(display)

        # Soil mask boundary contour: visible only in Iron Oxide view
        self.boundary_artist.set_visible(self.current_view == "Iron_Oxide")
        # Soil mask overlay: always respects the user toggle
        self.overlay_artist.set_visible(self.overlay_visible)

        _TITLES = {
            "RGB":        "RGB Composite  (660 / 550 / 480 nm)",
            "NIR":        "NIR False Colour  (850 / 660 / 550 nm)",
            "Iron_Oxide": (
                "Fe\u00b3\u207a Oxide Ratio  (900 / 660 nm)\n"
                "High value \u2192 strong Fe absorption \u2014 "
                "white contour = soil mask boundary"
            ),
        }
        self.ax_img.set_title(_TITLES[self.current_view], fontsize=9.5)
        self.fig.canvas.draw_idle()

    def _update_spec_from_roi(self, r1, c1, r2, c2, cls):
        """Compute mean ± 1-σ spectrum for an ROI and update the spectral panel.

        Only pixels within the soil mask are included.  Updates:
        * ``self.main_line``  — mean spectrum (solid, class-coloured)
        * ``self.std_fill``   — ±1 σ shaded band
        * ``self.bg_lines``   — all class mean spectra (faint background)

        Parameters
        ----------
        r1, c1, r2, c2 : int   — rectangle bounds (inclusive, in pixel coords)
        cls : str               — AMD class name

        Returns
        -------
        int — number of soil pixels inside the ROI
        """
        rr = np.arange(r1, r2 + 1)
        cc = np.arange(c1, c2 + 1)
        rr_g, cc_g = np.meshgrid(rr, cc, indexing="ij")
        rr_f = rr_g.ravel()
        cc_f = cc_g.ravel()
        valid = self.soil_mask[rr_f, cc_f]
        n_valid = int(valid.sum())

        if n_valid == 0:
            self._update_status(
                f"Warning: no soil pixels in ROI for {cls}.\n"
                "Try a different region."
            )
            return 0

        spectra  = self.cube[rr_f[valid], cc_f[valid], :].astype(np.float64)
        mean_spec = spectra.mean(axis=0)
        std_spec  = spectra.std(axis=0)

        color = CLASS_COLORS[cls]
        self.main_line.set_data(self.wavelengths, mean_spec)
        self.main_line.set_color(color)
        self.main_line.set_linewidth(2.2)
        self.main_line.set_label(f"{cls.replace('_', ' ')} ROI  (n={n_valid})")

        # Replace ±1 σ fill
        if self.std_fill is not None:
            self.std_fill.remove()
        self.std_fill = self.ax_spec.fill_between(
            self.wavelengths,
            mean_spec - std_spec,
            mean_spec + std_spec,
            color=color, alpha=0.18, zorder=3,
        )

        # Refresh background class lines (includes the new ROI)
        self._update_bg_lines()

        # Auto-scale y-axis to the data
        ymax = max(float(np.nanmax(mean_spec + std_spec)) * 1.25, 0.35)
        self.ax_spec.set_ylim(0, ymax)
        self.ax_spec.legend(
            fontsize=7, loc="upper right",
            framealpha=0.25, edgecolor="#444466",
        )

        return n_valid

    def _update_bg_lines(self):
        """Recompute per-class mean spectra from all drawn ROIs.

        This is called after every new ROI is added so that the spectral panel
        always shows the current class membership at a glance.
        """
        for cls in AMD_CLASSES:
            if not self.rectangles[cls]:
                self.bg_lines[cls].set_data([], [])
                continue

            spectra_list = []
            for r1, c1, r2, c2 in self.rectangles[cls]:
                rr = np.arange(r1, r2 + 1)
                cc = np.arange(c1, c2 + 1)
                rr_g, cc_g = np.meshgrid(rr, cc, indexing="ij")
                rr_f = rr_g.ravel()
                cc_f = cc_g.ravel()
                valid = self.soil_mask[rr_f, cc_f]
                if valid.any():
                    spectra_list.append(
                        self.cube[rr_f[valid], cc_f[valid], :]
                    )

            if spectra_list:
                mean_spec = np.vstack(spectra_list).mean(axis=0)
                self.bg_lines[cls].set_data(self.wavelengths, mean_spec)

    def _update_status(self, message):
        """Update the status text widget."""
        self.status_text.set_text(message)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def show(self):
        """Display the interactive window (blocking until the window is closed)."""
        plt.show()


# =============================================================================
# ENTRY POINT
# =============================================================================


def main():
    """Launch the interactive training pixel selection GUI."""
    print("=" * 60)
    print("TRAINING PIXEL SELECTOR")
    print("=" * 60)

    hdr_path = HDR_FILE
    if not Path(hdr_path).exists():
        print(f"\nERROR: Image file not found:\n  {hdr_path}")
        print(
            "\nPlease update the HDR_FILE constant at the top of this script "
            "to point to your ENVI .hdr file."
        )
        return

    selector = TrainingPixelSelector(hdr_path)

    print(
        "\nReady.  Two-class binary classification: AMD_FeOx / Background.\n"
        "  1. Switch to Fe-Oxide Ratio view for best contrast.\n"
        "  2. Select 'AMD_FeOx' class and draw ROIs over iron-oxide areas\n"
        "     (rust-orange/yellow outcrops with high Fe-ratio values).\n"
        "  3. Switch to 'Background' class and draw ROIs over unaltered areas.\n"
        "  4. Click 'Save & Continue' when done.\n"
    )

    selector.show()


if __name__ == "__main__":
    main()
