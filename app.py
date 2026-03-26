"""
Microtiter Plate Analyzer
=========================
Mobile-friendly web application for colorimetric analysis of 96-well microtiter plates.

The user photographs a plate, marks two reference wells (A1 and H12), and receives:
  1. A table of Euclidean RGB distances relative to row A (per column).
  2. A grayscale intensity table (0 = black, 255 = white) for each well.
     The grayscale table can be downloaded as a formatted Excel file.

Installation:
    pip install streamlit pillow numpy pandas openpyxl

Usage:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import math
import io
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

# ── Constants ──────────────────────────────────────────────────────────────────
ROWS          = list("ABCDEFGH")
COLS          = list(range(1, 13))
N_ROWS        = len(ROWS)
N_COLS        = len(COLS)
SAMPLE_RADIUS = 5     # half-width of the square sampling region (px); full region = 11×11 px
BASE_W        = 700   # base display width (px)

# ── Helper functions ────────────────────────────────────────────────────────────

def resize_to_width(img: Image.Image, width: int) -> Image.Image:
    """Resize image to the given width while preserving aspect ratio."""
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.LANCZOS)


def compute_grid(pt_a1, pt_h12) -> np.ndarray:
    """
    Compute center coordinates of all 96 wells by linear interpolation
    between the two user-defined reference points (A1 and H12).
    Returns an array of shape (8, 12, 2) — [row, col, (x, y)].
    """
    x1, y1 = pt_a1
    x2, y2 = pt_h12
    grid = np.zeros((N_ROWS, N_COLS, 2), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            grid[r, c, 0] = x1 + (c / (N_COLS - 1)) * (x2 - x1)
            grid[r, c, 1] = y1 + (r / (N_ROWS - 1)) * (y2 - y1)
    return grid


def sample_color(img_array, x, y, radius=SAMPLE_RADIUS):
    """
    Return the mean RGB value of a square region of size (2*radius+1)²
    centered on pixel (x, y). Clamps to image boundaries.
    """
    h, w = img_array.shape[:2]
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    patch = img_array[y0:y1, x0:x1, :3]
    return patch.reshape(-1, 3).mean(axis=0)


def euclidean(c1, c2):
    """Euclidean distance between two RGB color vectors."""
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(c1, c2)))


def build_results(img_array, grid):
    """
    Sample colors at all 96 well positions and compute Euclidean RGB distances
    relative to a single reference vector derived from row A.

    The reference is the mean RGB vector of all 12 wells in row A, where each
    well is weighted by its perceptual luminance (ITU-R BT.601):
        L = 0.299·R + 0.587·G + 0.114·B
    Wells with higher luminance contribute more to the reference, reflecting
    the fact that brighter, more saturated wells carry more colorimetric signal.

    Returns:
        colors   : ndarray of shape (8, 12, 3) with mean RGB per well
        dist_df  : DataFrame (8×12) of Euclidean distances from the row-A reference
        ref_rgb  : 1-D array (3,) — the reference RGB vector used
    """
    colors = np.zeros((N_ROWS, N_COLS, 3), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            colors[r, c] = sample_color(img_array, *grid[r, c])

    # Compute perceptual luminance weights for row A wells
    row_a   = colors[0, :, :]                                      # shape (12, 3)
    lum_a   = 0.299*row_a[:, 0] + 0.587*row_a[:, 1] + 0.114*row_a[:, 2]  # shape (12,)
    weights = lum_a / lum_a.sum() if lum_a.sum() > 0 else np.ones(N_COLS) / N_COLS
    ref_rgb = (row_a * weights[:, np.newaxis]).sum(axis=0)         # weighted mean RGB

    # Euclidean distance of every well from the single row-A reference
    distances = np.zeros((N_ROWS, N_COLS), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            distances[r, c] = euclidean(ref_rgb, colors[r, c])

    dist_df = pd.DataFrame(distances, index=ROWS, columns=[str(c) for c in COLS])
    return colors, dist_df, ref_rgb


def draw_crosshair(img: Image.Image, x: int, y: int,
                   color: tuple, size: int = 30, thickness: int = 2) -> Image.Image:
    """Draw a crosshair (cross + centre dot) at position (x, y) on a copy of the image."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    # Horizontal line
    draw.line([(x - size, y), (x + size, y)], fill=color, width=thickness)
    # Vertical line
    draw.line([(x, y - size), (x, y + size)], fill=color, width=thickness)
    # Centre dot
    r = 5
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(255,255,255), width=1)
    return out


def draw_grid_on_image(img: Image.Image, grid: np.ndarray,
                        pt_a1=None, pt_h12=None) -> Image.Image:
    """
    Overlay detected well positions on the image.
    Row A wells are drawn in yellow; all other rows in cyan.
    Reference points A1 and H12 are additionally highlighted with larger circles
    in red and green respectively.
    """
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    r_draw = max(4, int(min(img.width, img.height) / 120))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            color = (255, 220, 0) if r == 0 else (0, 200, 255)
            draw.ellipse([x-r_draw, y-r_draw, x+r_draw, y+r_draw],
                         outline=color, width=2)
    if pt_a1:
        x, y = pt_a1
        s = r_draw * 2
        draw.ellipse([x-s, y-s, x+s, y+s], outline=(255, 60, 60), width=3)
    if pt_h12:
        x, y = pt_h12
        s = r_draw * 2
        draw.ellipse([x-s, y-s, x+s, y+s], outline=(60, 255, 60), width=3)
    return out


def zoom_crop(img: Image.Image, cx: int, cy: int, zoom: int) -> tuple:
    """
    Return a zoomed view of the image centred on (cx, cy).

    At zoom=100 the original image is returned unchanged (no crop).
    At zoom>100 a region of size (width * 100/zoom) is cropped around (cx, cy)
    and upscaled back to the original dimensions, simulating optical zoom.

    Returns:
        (zoomed_image, offset_x, offset_y)
        offset_x/y: top-left corner of the crop in original image coordinates
    """
    if zoom <= 100:
        return img, 0, 0
    # Visible region size = original size * (100 / zoom)
    frac   = 100 / zoom
    crop_w = int(img.width  * frac)
    crop_h = int(img.height * frac)
    # Centre crop on crosshair position, clamped to image boundaries
    ox = max(0, min(cx - crop_w // 2, img.width  - crop_w))
    oy = max(0, min(cy - crop_h // 2, img.height - crop_h))
    cropped = img.crop((ox, oy, ox + crop_w, oy + crop_h))
    zoomed  = cropped.resize((img.width, img.height), Image.LANCZOS)
    return zoomed, ox, oy


def color_table_html(colors, dist_df, ref_rgb) -> str:
    """
    Build an HTML table where each cell background reflects the actual well color.
    Cell value = Euclidean RGB distance from the single row-A reference vector.
    Row A is highlighted with a red border to mark it as the reference row.
    """
    # Reference colour swatch for display
    r_ref, g_ref, b_ref = ref_rgb.astype(int)
    lum_ref = 0.299*r_ref + 0.587*g_ref + 0.114*b_ref
    fg_ref  = "#000" if lum_ref > 128 else "#fff"
    ref_swatch = (
        f"<div style='display:inline-block;width:18px;height:18px;border-radius:3px;"
        f"background:rgb({r_ref},{g_ref},{b_ref});border:1px solid #999;"
        f"vertical-align:middle;margin-right:5px;'></div>"
        f"<span style='font-size:11px;'>Reference (row A mean): "
        f"R={r_ref} G={g_ref} B={b_ref}</span>"
    )

    html  = f"<div style='margin-bottom:6px;'>{ref_swatch}</div>"
    html += '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            rgb    = colors[r_idx, c_idx].astype(int)
            dist   = dist_df.iloc[r_idx, c_idx]
            bg     = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            lum    = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            fg     = "#000" if lum > 128 else "#fff"
            border = "2px solid #e00" if r_idx == 0 else "1px solid #ccc"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:{border};min-width:36px;'>"
                     f"{dist:.1f}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def build_grayscale(img_orig: Image.Image, grid: np.ndarray) -> pd.DataFrame:
    """
    Convert the image to grayscale and sample the mean intensity (0–255)
    at each of the 96 well positions.
    Returns a DataFrame (8×12) of integer grayscale values.
    """
    img_gray  = img_orig.convert("L")          # standard luminance grayscale
    gray_array = np.array(img_gray, dtype=float)
    values = np.zeros((N_ROWS, N_COLS), dtype=float)
    h, w   = gray_array.shape
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            x0, x1 = max(0, int(x) - SAMPLE_RADIUS), min(w, int(x) + SAMPLE_RADIUS + 1)
            y0, y1 = max(0, int(y) - SAMPLE_RADIUS), min(h, int(y) + SAMPLE_RADIUS + 1)
            values[r, c] = gray_array[y0:y1, x0:x1].mean()
    return pd.DataFrame(
        np.round(values).astype(int),
        index=ROWS,
        columns=[str(c) for c in COLS]
    )


def grayscale_table_html(gray_df: pd.DataFrame) -> str:
    """
    Build an HTML table showing grayscale intensity per well.
    Cell background matches the actual gray shade; text colour adapts for contrast.
    """
    html = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            g      = int(gray_df.iloc[r_idx, c_idx])
            bg     = f"rgb({g},{g},{g})"
            fg     = "#000" if g > 128 else "#fff"
            border = "1px solid #ccc"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:{border};min-width:36px;'>"
                     f"{g}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def export_grayscale_excel(gray_df: pd.DataFrame, img_orig: Image.Image,
                           grid: np.ndarray) -> bytes:
    """
    Create a formatted Excel workbook with the grayscale intensity table.
    Each cell:
      - displays the integer grayscale value (0–255)
      - has a background fill matching the actual gray shade of that well
      - uses black or white text for legibility
    Column headers (1–12) and row headers (A–H) are included.
    Returns the workbook as bytes suitable for st.download_button.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Grayscale intensities"

    thin = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # ── Header row (column numbers) ────────────────────────────────────────────
    ws.cell(row=1, column=1, value="")           # top-left corner cell (empty)
    for c_idx, c_label in enumerate(COLS):
        cell = ws.cell(row=1, column=c_idx + 2, value=c_label)
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border    = border

    # ── Data rows ──────────────────────────────────────────────────────────────
    for r_idx, row_label in enumerate(ROWS):
        # Row letter header
        hdr = ws.cell(row=r_idx + 2, column=1, value=row_label)
        hdr.font      = Font(bold=True)
        hdr.alignment = Alignment(horizontal="center")
        hdr.border    = border

        for c_idx in range(N_COLS):
            g    = int(gray_df.iloc[r_idx, c_idx])
            cell = ws.cell(row=r_idx + 2, column=c_idx + 2, value=g)

            # Gray fill — openpyxl expects a 6-digit hex colour string
            hex_g  = f"{g:02X}"
            hex_col = hex_g * 3          # e.g. "7F7F7F"
            cell.fill      = PatternFill("solid", fgColor="FF" + hex_col)
            cell.font      = Font(color="000000" if g > 128 else "FFFFFF")
            cell.alignment = Alignment(horizontal="center")
            cell.border    = border

    # ── Column widths ──────────────────────────────────────────────────────────
    ws.column_dimensions["A"].width = 5   # row-label column
    for c_idx in range(N_COLS):
        ws.column_dimensions[get_column_letter(c_idx + 2)].width = 6

    # ── Metadata sheet ─────────────────────────────────────────────────────────
    ws_meta = wb.create_sheet("Info")
    ws_meta.append(["Microtiter Plate Analyzer — grayscale export"])
    ws_meta.append(["Values represent mean grayscale intensity (0=black, 255=white)"])
    ws_meta.append(["Sampling region: 11×11 px centred on each well"])
    ws_meta.append(["Grayscale conversion: ITU-R BT.601 (PIL Image.convert('L'))"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def inverted_table_html(gray_df: pd.DataFrame) -> str:
    """
    Build an HTML table showing pseudo-absorbance (255 - grayscale) per well.
    Higher value = darker well = stronger reaction.
    Cell background is the inverted gray shade for visual consistency.
    """
    html = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            g_inv  = 255 - int(gray_df.iloc[r_idx, c_idx])
            # Background uses the inverted shade so darker = higher value visually
            bg     = f"rgb({g_inv},{g_inv},{g_inv})"
            fg     = "#000" if g_inv > 128 else "#fff"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:1px solid #ccc;min-width:36px;'>"
                     f"{g_inv}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def export_inverted_excel(gray_df: pd.DataFrame) -> bytes:
    """
    Create a formatted Excel workbook with the pseudo-absorbance table (255 - grayscale).
    Each cell background reflects the inverted gray shade (darker = higher value).
    Returns the workbook as bytes.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Pseudo-absorbance"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Header row
    ws.cell(row=1, column=1, value="")
    for c_idx, c_label in enumerate(COLS):
        cell = ws.cell(row=1, column=c_idx + 2, value=c_label)
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border    = border

    # Data rows
    for r_idx, row_label in enumerate(ROWS):
        hdr = ws.cell(row=r_idx + 2, column=1, value=row_label)
        hdr.font      = Font(bold=True)
        hdr.alignment = Alignment(horizontal="center")
        hdr.border    = border

        for c_idx in range(N_COLS):
            g_inv  = 255 - int(gray_df.iloc[r_idx, c_idx])
            cell   = ws.cell(row=r_idx + 2, column=c_idx + 2, value=g_inv)
            hex_g  = f"{g_inv:02X}"
            hex_col = hex_g * 3
            cell.fill      = PatternFill("solid", fgColor="FF" + hex_col)
            cell.font      = Font(color="000000" if g_inv > 128 else "FFFFFF")
            cell.alignment = Alignment(horizontal="center")
            cell.border    = border

    # Column widths
    ws.column_dimensions["A"].width = 5
    for c_idx in range(N_COLS):
        ws.column_dimensions[get_column_letter(c_idx + 2)].width = 6

    # Metadata sheet
    ws_meta = wb.create_sheet("Info")
    ws_meta.append(["Microtiter Plate Analyzer — pseudo-absorbance export"])
    ws_meta.append(["Values = 255 - grayscale intensity (0=white/no reaction, 255=black/full reaction)"])
    ws_meta.append(["Higher value = darker well = stronger colorimetric reaction"])
    ws_meta.append(["Grayscale conversion: ITU-R BT.601 (PIL Image.convert('L'))"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def build_absorbance(img_orig: Image.Image, grid: np.ndarray,
                     blank_row_idx: int = 0,
                     k_override: float = 3.0) -> pd.DataFrame:
    """
    Compute approximate absorbance for each well using a global blank reference.

    The blank is estimated as the mean grayscale intensity of all 12 wells in the
    selected blank row. This represents unabsorbed light, analogous to the blank
    cuvette in a spectrophotometer.

        gray_blank = mean of all blank-row well intensities
        gray_mean  = mean of all pixels in the 11×11 px region of each well
        A          = -log10(gray_mean / gray_blank) × k_override

    k_override is a user-supplied correction factor that compensates for smartphone
    camera gamma and JPEG compression. It can be calibrated as:
        k = A_expected (from plate reader) / A_raw (from this function with k=1)

    Returns a DataFrame (8×12) of absorbance values rounded to 4 decimal places.
    Wells lighter than the blank are set to 0.
    """
    img_gray   = img_orig.convert("L")
    gray_array = np.array(img_gray, dtype=float)
    h, w       = gray_array.shape

    # First pass: collect mean intensity for all wells
    means = np.zeros((N_ROWS, N_COLS), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            x0 = max(0, int(x) - SAMPLE_RADIUS)
            x1 = min(w, int(x) + SAMPLE_RADIUS + 1)
            y0 = max(0, int(y) - SAMPLE_RADIUS)
            y1 = min(h, int(y) + SAMPLE_RADIUS + 1)
            means[r, c] = gray_array[y0:y1, x0:x1].mean()

    # Global blank = mean intensity of the selected blank row
    gray_blank = means[blank_row_idx, :].mean()

    # Compute absorbance using user-supplied correction factor
    values = np.zeros((N_ROWS, N_COLS), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            gm = means[r, c]
            if gray_blank > 0 and gm < gray_blank:
                values[r, c] = -math.log10(gm / gray_blank) * k_override
            else:
                values[r, c] = 0.0

    return pd.DataFrame(
        np.round(values, 4),
        index=ROWS,
        columns=[str(c) for c in COLS]
    )


def absorbance_table_html(abs_df: pd.DataFrame) -> str:
    """
    Build an HTML table displaying estimated absorbance values (grayscale-based).
    Higher absorbance = darker background (maps A to a gray shade for visual cue).
    Values are shown to 4 decimal places.
    """
    # Map absorbance to a gray shade: A=0 → white (255), A≥2 → black (0)
    def a_to_gray(a):
        return max(0, int(255 * (1 - min(a, 2.0) / 2.0)))

    html  = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            a  = float(abs_df.iloc[r_idx, c_idx])
            g  = a_to_gray(a)
            bg = f"rgb({g},{g},{g})"
            fg = "#000" if g > 128 else "#fff"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:1px solid #ccc;min-width:42px;'>"
                     f"{a:.4f}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def build_absorbance_channel(img_orig: Image.Image, grid: np.ndarray,
                             blank_row_idx: int = 0,
                             channel: int = 2,
                             k_override: float = 1.0) -> pd.DataFrame:
    """
    Compute absorbance using a single RGB channel instead of grayscale.
    This improves sensitivity for dyes with a narrow absorption peak, because
    only the channel most affected by the dye is used (e.g. Blue for eosin).

        channel: 0 = Red, 1 = Green, 2 = Blue
        blank_row_idx: row used as blank reference
        k_override: correction factor (same meaning as in build_absorbance)

    Formula: A = -log10(ch_mean / ch_blank) × k_override
    """
    img_array = np.array(img_orig.convert("RGB"), dtype=float)
    ch_array  = img_array[:, :, channel]
    h, w      = ch_array.shape

    means = np.zeros((N_ROWS, N_COLS), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            x0 = max(0, int(x) - SAMPLE_RADIUS)
            x1 = min(w, int(x) + SAMPLE_RADIUS + 1)
            y0 = max(0, int(y) - SAMPLE_RADIUS)
            y1 = min(h, int(y) + SAMPLE_RADIUS + 1)
            means[r, c] = ch_array[y0:y1, x0:x1].mean()

    ch_blank = means[blank_row_idx, :].mean()

    values = np.zeros((N_ROWS, N_COLS), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cm = means[r, c]
            if ch_blank > 0 and cm < ch_blank:
                values[r, c] = -math.log10(cm / ch_blank) * k_override
            else:
                values[r, c] = 0.0

    return pd.DataFrame(
        np.round(values, 4),
        index=ROWS,
        columns=[str(c) for c in COLS]
    )


# Channel display colours for background tint in table and Excel
CHANNEL_TINT = {
    0: (255, 200, 200),   # Red channel → light red tint
    1: (200, 240, 200),   # Green channel → light green tint
    2: (200, 210, 255),   # Blue channel → light blue tint
}
CHANNEL_NAME = {0: "Red", 1: "Green", 2: "Blue"}


def channel_absorbance_table_html(abs_df: pd.DataFrame, channel: int) -> str:
    """
    HTML table for single-channel absorbance.
    Background uses a tinted shade matching the selected channel,
    darkening with higher absorbance values.
    """
    tr, tg, tb = CHANNEL_TINT[channel]

    def cell_bg(a):
        # Darken the channel tint linearly with absorbance (max at A=2)
        factor = max(0.0, 1.0 - min(a, 2.0) / 2.0)
        r = int(tr + (0 - tr) * (1 - factor))
        g = int(tg + (0 - tg) * (1 - factor))
        b = int(tb + (0 - tb) * (1 - factor))
        return f"rgb({r},{g},{b})", 0.299*r + 0.587*g + 0.114*b

    html  = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            a        = float(abs_df.iloc[r_idx, c_idx])
            bg, lum  = cell_bg(a)
            fg       = "#000" if lum > 128 else "#fff"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:1px solid #ccc;min-width:42px;'>"
                     f"{a:.4f}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def export_channel_absorbance_excel(abs_df: pd.DataFrame, channel: int,
                                    blank_row_label: str, k: float) -> bytes:
    """
    Excel export for single-channel absorbance.
    Cell background uses a tinted shade matching the channel.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = f"{CHANNEL_NAME[channel]} channel absorbance"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    tr, tg, tb = CHANNEL_TINT[channel]

    def a_to_hex(a):
        factor = max(0.0, 1.0 - min(a, 2.0) / 2.0)
        r = int(tr + (0 - tr) * (1 - factor))
        g = int(tg + (0 - tg) * (1 - factor))
        b = int(tb + (0 - tb) * (1 - factor))
        return f"{r:02X}{g:02X}{b:02X}"

    # Column headers
    ws.cell(row=1, column=1, value="")
    for c_idx, c_label in enumerate(COLS):
        cell           = ws.cell(row=1, column=c_idx + 2, value=c_label)
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border    = border

    # Data rows
    for r_idx, row_label in enumerate(ROWS):
        hdr            = ws.cell(row=r_idx + 2, column=1, value=row_label)
        hdr.font       = Font(bold=True)
        hdr.alignment  = Alignment(horizontal="center")
        hdr.border     = border

        for c_idx in range(N_COLS):
            a    = float(abs_df.iloc[r_idx, c_idx])
            cell = ws.cell(row=r_idx + 2, column=c_idx + 2, value=round(a, 4))
            hex_col       = a_to_hex(a)
            cell.fill     = PatternFill("solid", fgColor="FF" + hex_col)
            r2, g2, b2    = int(hex_col[0:2],16), int(hex_col[2:4],16), int(hex_col[4:6],16)
            lum           = 0.299*r2 + 0.587*g2 + 0.114*b2
            cell.font     = Font(color="000000" if lum > 128 else "FFFFFF")
            cell.alignment = Alignment(horizontal="center")
            cell.border   = border

    ws.column_dimensions["A"].width = 5
    for c_idx in range(N_COLS):
        ws.column_dimensions[get_column_letter(c_idx + 2)].width = 9

    ws_meta = wb.create_sheet("Info")
    ws_meta.append([f"Microtiter Plate Analyzer — {CHANNEL_NAME[channel]} channel absorbance"])
    ws_meta.append([f"Channel used: {CHANNEL_NAME[channel]} (0=R, 1=G, 2=B)"])
    ws_meta.append([f"Blank reference row: {blank_row_label}"])
    ws_meta.append([f"Correction factor k: {k}"])
    ws_meta.append(["Formula: A = -log10(ch_mean / ch_blank) × k"])
    ws_meta.append(["Using a single channel improves sensitivity for dyes with narrow absorption peaks."])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()



    """
    Build an HTML table displaying estimated absorbance values.
    Higher absorbance = darker background (maps A to a gray shade for visual cue).
    Values are shown to 4 decimal places.
    """
    # Map absorbance to a gray shade: A=0 → white (255), A≥2 → black (0)
    def a_to_gray(a):
        return max(0, int(255 * (1 - min(a, 2.0) / 2.0)))

    html  = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            a  = float(abs_df.iloc[r_idx, c_idx])
            g  = a_to_gray(a)
            bg = f"rgb({g},{g},{g})"
            fg = "#000" if g > 128 else "#fff"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:1px solid #ccc;min-width:42px;'>"
                     f"{a:.4f}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def export_absorbance_excel(abs_df: pd.DataFrame) -> bytes:
    """
    Create a formatted Excel workbook with the estimated absorbance table.
    Cell background maps absorbance to a gray shade (A=0 → white, A≥2 → black).
    Returns the workbook as bytes.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Estimated absorbance"

    thin   = Side(style="thin", color="CCCCCC")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    def a_to_hex(a):
        g   = max(0, int(255 * (1 - min(a, 2.0) / 2.0)))
        hx  = f"{g:02X}"
        return hx * 3

    # Column headers
    ws.cell(row=1, column=1, value="")
    for c_idx, c_label in enumerate(COLS):
        cell = ws.cell(row=1, column=c_idx + 2, value=c_label)
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border    = border

    # Data rows
    for r_idx, row_label in enumerate(ROWS):
        hdr = ws.cell(row=r_idx + 2, column=1, value=row_label)
        hdr.font      = Font(bold=True)
        hdr.alignment = Alignment(horizontal="center")
        hdr.border    = border

        for c_idx in range(N_COLS):
            a    = float(abs_df.iloc[r_idx, c_idx])
            cell = ws.cell(row=r_idx + 2, column=c_idx + 2, value=round(a, 4))

            hex_col       = a_to_hex(a)
            cell.fill     = PatternFill("solid", fgColor="FF" + hex_col)
            g_int         = int(hex_col[:2], 16)
            cell.font     = Font(color="000000" if g_int > 128 else "FFFFFF")
            cell.alignment = Alignment(horizontal="center")
            cell.border   = border

    # Column widths
    ws.column_dimensions["A"].width = 5
    for c_idx in range(N_COLS):
        ws.column_dimensions[get_column_letter(c_idx + 2)].width = 9

    # Metadata sheet
    ws_meta = wb.create_sheet("Info")
    ws_meta.append(["Microtiter Plate Analyzer — estimated absorbance export"])
    ws_meta.append(["Blank reference = mean grayscale intensity of all 12 row-A wells"])
    ws_meta.append(["gray_mean = mean of all pixels in the 11×11 px well region"])
    ws_meta.append(["k = 1.0 + 0.5 × (1 - gray_blank/255)²  (smartphone gamma correction)"])
    ws_meta.append(["A = -log10(gray_mean / gray_blank) × k"])
    ws_meta.append(["Wells lighter than blank → A = 0"])
    ws_meta.append(["Background colour: white = A≈0, black = A≥2"])

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def export_distances_excel(colors, dist_df, ref_rgb) -> bytes:
    """
    Create a formatted Excel workbook with the Euclidean distance table.
    Structure mirrors the grayscale export: column headers in row 1, data from row 2.
    Each cell background reflects the actual well colour.
    Row A cells have a red border; the minimum per column (B–H) is bold and underlined.
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Euclidean distances"

    thin      = Side(style="thin",   color="CCCCCC")
    thick_red = Side(style="medium", color="EE0000")
    border_std = Border(left=thin,      right=thin,      top=thin,      bottom=thin)
    border_ref = Border(left=thick_red, right=thick_red, top=thick_red, bottom=thick_red)

    # ── Column headers (row 1) ─────────────────────────────────────────────────
    ws.cell(row=1, column=1, value="")
    for c_idx, c_label in enumerate(COLS):
        cell = ws.cell(row=1, column=c_idx + 2, value=c_label)
        cell.font      = Font(bold=True)
        cell.alignment = Alignment(horizontal="center")
        cell.border    = border_std

    # ── Find minimum per column (rows B–H only) ────────────────────────────────
    # (kept for metadata info only, no longer used for cell formatting)

    # ── Data rows (rows 2–9) ───────────────────────────────────────────────────
    for r_idx, row_label in enumerate(ROWS):
        hdr = ws.cell(row=r_idx + 2, column=1, value=row_label)
        hdr.font      = Font(bold=True)
        hdr.alignment = Alignment(horizontal="center")
        hdr.border    = border_std

        for c_idx in range(N_COLS):
            rgb  = colors[r_idx, c_idx].astype(int)
            dist = round(float(dist_df.iloc[r_idx, c_idx]), 1)
            cell = ws.cell(row=r_idx + 2, column=c_idx + 2, value=dist)

            # Background = actual well colour
            hex_col = f"{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
            cell.fill = PatternFill("solid", fgColor="FF" + hex_col)

            # Text colour based on luminance
            lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            cell.font      = Font(color="000000" if lum > 128 else "FFFFFF")
            cell.alignment = Alignment(horizontal="center")
            cell.border    = border_ref if r_idx == 0 else border_std

    # ── Column widths ──────────────────────────────────────────────────────────
    ws.column_dimensions["A"].width = 5
    for c_idx in range(N_COLS):
        ws.column_dimensions[get_column_letter(c_idx + 2)].width = 7

    # ── Metadata sheet ─────────────────────────────────────────────────────────
    r_ref, g_ref, b_ref = ref_rgb.astype(int)
    ws_meta = wb.create_sheet("Info")
    ws_meta.append(["Microtiter Plate Analyzer — Euclidean distance export"])
    ws_meta.append(["Reference = luminance-weighted mean RGB of all 12 row-A wells"])
    ws_meta.append([f"Reference RGB: R={r_ref}  G={g_ref}  B={b_ref}"])
    ws_meta.append(["Distance = sqrt((R1-R2)² + (G1-G2)² + (B1-B2)²)"])
    ws_meta.append(["Red border = row A (reference row)"])
    ws_meta.append(["Bold + underline = minimum distance per column (most similar to reference)"])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# Known correction factors per camera brand (empirical values)
CAMERA_K = {
    "apple":   2.8,   # iPhone 13–16 / Air
    "samsung": 2.4,
    "google":  3.5,   # Pixel
    "xiaomi":  2.2,
    "redmi":   2.2,
    "huawei":  2.3,
    "honor":   2.3,
    "oneplus": 2.5,
    "oppo":    2.5,
}
K_DEFAULT = 3.0   # fallback when brand is unknown or EXIF is missing


def k_from_exif(uploaded_file) -> tuple[float, str]:
    """
    Read EXIF Make/Model from an uploaded image file and return
    (k_value, info_string). Falls back to K_DEFAULT if EXIF is absent
    or brand is unrecognised.

    Reads the file into a BytesIO buffer first to avoid stream position issues
    with Streamlit's UploadedFile object.
    """
    try:
        # Read entire file into memory so seek/reopen works reliably
        uploaded_file.seek(0)
        raw_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        buf = io.BytesIO(raw_bytes)

        img_raw = Image.open(buf)
        img_raw.load()   # force full load including EXIF

        # PIL ≥ 9.1: use getexif(); fallback to _getexif() for older versions
        try:
            exif_data = img_raw.getexif()          # returns dict-like ExifData
            make  = str(exif_data.get(271, "")).strip().lower()
            model = str(exif_data.get(272, "")).strip()
        except AttributeError:
            raw_exif = img_raw._getexif() or {}
            make  = str(raw_exif.get(271, "")).strip().lower()
            model = str(raw_exif.get(272, "")).strip()

        if not make:
            return K_DEFAULT, "No EXIF camera data found — using default k."

        for brand, k_val in CAMERA_K.items():
            if brand in make:
                return k_val, f"Detected: **{make.title()} {model}** → k = {k_val}"

        return K_DEFAULT, (f"Detected: **{make.title()} {model}** "
                           f"(unknown brand) → using default k = {K_DEFAULT}")

    except Exception as e:
        return K_DEFAULT, f"Could not read EXIF data ({e}) — using default k."


# ── Main UI ────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Microtiter Analyzer",
        page_icon="🧪",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

    # Mobile-friendly CSS overrides
    st.markdown("""
    <style>
      .block-container { padding:1rem 0.6rem 2rem !important; max-width:100% !important; }
      h1  { font-size:1.35rem !important; }
      h2, h3 { font-size:1.05rem !important; }
      .stButton>button, .stDownloadButton>button {
        width:100%; padding:0.8rem; font-size:1rem; border-radius:10px; margin-top:4px;
      }
      div[data-testid="stSlider"] { padding: 0 4px; }
      iframe { border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧪 Microtiter Analyzer")

    # ── Session state initialisation ───────────────────────────────────────────
    defaults = {"step": 1, "pt_a1": None, "pt_h12": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Step 0: Image upload ───────────────────────────────────────────────────
    st.markdown("### 📷 Upload a plate photograph")
    uploaded = st.file_uploader("JPG or PNG", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if not uploaded:
        st.info("Select or photograph the plate using the button above.")
        return

    # Read entire file into memory once — avoids stream position issues
    uploaded.seek(0)
    file_bytes = uploaded.read()
    uploaded.seek(0)

    # Read EXIF camera brand and set default k (only on first load of this image)
    if "exif_k" not in st.session_state:
        exif_k, exif_msg = k_from_exif(io.BytesIO(file_bytes))
        st.session_state.exif_k   = exif_k
        st.session_state.exif_msg = exif_msg

    img_orig  = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img_array = np.array(img_orig)
    OW, OH    = img_orig.width, img_orig.height

    st.success(f"Image loaded: {OW}×{OH} px")

    # ── Step 1: Select well A1 ─────────────────────────────────────────────────
    if st.session_state.step == 1:
        st.markdown("### 1️⃣ Position the crosshair on well **A1** (top-left corner)")

        # Working image — always BASE_W px wide; crosshair coordinates are in display px
        img_d  = resize_to_width(img_orig, BASE_W)
        DW, DH = img_d.width, img_d.height
        sx, sy = OW / DW, OH / DH   # scale factors: display px → original px

        # Default crosshair position (~8 % from the top-left corner)
        if "a1_x" not in st.session_state:
            st.session_state.a1_x = max(1, int(DW * 0.08))
        if "a1_y" not in st.session_state:
            st.session_state.a1_y = max(1, int(DH * 0.08))

        # X / Y sliders (values in display px)
        st.caption("1. Move sliders to approximate position  2. Zoom in  3. Fine-tune sliders")
        ax = st.slider("← X (horizontal) →", 0, DW,
                       min(st.session_state.a1_x, DW), key="sl_a1x")
        ay = st.slider("↑ Y (vertical) ↓",   0, DH,
                       min(st.session_state.a1_y, DH), key="sl_a1y")
        st.session_state.a1_x = ax
        st.session_state.a1_y = ay

        zoom = st.slider("🔍 Zoom", 100, 600,
                         value=st.session_state.get("zoom_a1", 100),
                         step=25, format="%d%%", key="zoom_a1")

        # Crop around the crosshair and upscale to simulate zoom
        zoomed, off_x, off_y = zoom_crop(img_d, ax, ay, zoom)
        # Crosshair position in the zoomed image coordinate system
        ch_x = int((ax - off_x) * zoom / 100)
        ch_y = int((ay - off_y) * zoom / 100)
        preview = draw_crosshair(zoomed, ch_x, ch_y, color=(255, 60, 60), size=20)
        st.image(preview, use_container_width=True)
        st.caption(f"Position in original image: X={int(ax*sx)}, Y={int(ay*sy)}")

        if st.button("✅ Confirm A1 and continue", type="primary"):
            st.session_state.pt_a1 = (int(ax * sx), int(ay * sy))
            st.session_state.step  = 2
            st.rerun()

    # ── Step 2: Select well H12 ────────────────────────────────────────────────
    elif st.session_state.step == 2:
        st.success(f"✅ A1 saved: {st.session_state.pt_a1}")
        st.markdown("### 2️⃣ Position the crosshair on well **H12** (bottom-right corner)")

        img_d  = resize_to_width(img_orig, BASE_W)
        DW, DH = img_d.width, img_d.height
        sx, sy = OW / DW, OH / DH

        # Default crosshair position (~92 % from the top-left corner)
        if "h12_x" not in st.session_state:
            st.session_state.h12_x = max(1, int(DW * 0.92))
        if "h12_y" not in st.session_state:
            st.session_state.h12_y = max(1, int(DH * 0.92))

        st.caption("1. Move sliders to approximate position  2. Zoom in  3. Fine-tune sliders")
        hx = st.slider("← X (horizontal) →", 0, DW,
                       min(st.session_state.h12_x, DW), key="sl_h12x")
        hy = st.slider("↑ Y (vertical) ↓",   0, DH,
                       min(st.session_state.h12_y, DH), key="sl_h12y")
        st.session_state.h12_x = hx
        st.session_state.h12_y = hy

        zoom = st.slider("🔍 Zoom", 100, 600,
                         value=st.session_state.get("zoom_h12", 100),
                         step=25, format="%d%%", key="zoom_h12")

        zoomed, off_x, off_y = zoom_crop(img_d, hx, hy, zoom)
        ch_x = int((hx - off_x) * zoom / 100)
        ch_y = int((hy - off_y) * zoom / 100)
        preview = draw_crosshair(zoomed, ch_x, ch_y, color=(60, 220, 60), size=20)
        st.image(preview, use_container_width=True)
        st.caption(f"Position in original image: X={int(hx*sx)}, Y={int(hy*sy)}")

        if st.button("✅ Confirm H12 and run analysis", type="primary"):
            st.session_state.pt_h12 = (int(hx * sx), int(hy * sy))
            st.session_state.step   = 3
            st.rerun()

        if st.button("↩️ Back — reselect A1"):
            st.session_state.step  = 1
            st.session_state.pt_a1 = None
            for k in ["a1_x","a1_y","h12_x","h12_y"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Step 3: Results ────────────────────────────────────────────────────────
    elif st.session_state.step == 3:
        pt_a1  = st.session_state.pt_a1
        pt_h12 = st.session_state.pt_h12
        st.success(f"✅ A1: {pt_a1}  |  H12: {pt_h12}")

        grid = compute_grid(pt_a1, pt_h12)

        st.markdown("### 3️⃣ Detected well grid")
        annotated_orig = draw_grid_on_image(img_orig, grid, pt_a1, pt_h12)
        st.image(resize_to_width(annotated_orig, BASE_W),
                 caption="Yellow = row A  |  Cyan = other rows  |  Red = A1  |  Green = H12",
                 use_container_width=True)

        colors, dist_df, ref_rgb = build_results(img_array, grid)

        st.markdown("### 4️⃣ Euclidean distances from row A reference")
        st.markdown(color_table_html(colors, dist_df, ref_rgb), unsafe_allow_html=True)
        st.caption("Each cell = Euclidean RGB distance from the luminance-weighted mean RGB of row A (all 12 wells). "
                   "Row A border = reference row.")

        dist_xlsx = export_distances_excel(colors, dist_df, ref_rgb)
        st.download_button(
            label="⬇️ Download distances table (Excel)",
            data=dist_xlsx,
            file_name="microtiter_distances.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ── Grayscale intensity table ──────────────────────────────────────────
        st.markdown("### 5️⃣ Grayscale intensity per well")
        st.caption("The image is converted to grayscale (ITU-R BT.601). "
                   "Each value is the mean pixel intensity (0 = black, 255 = white) "
                   "sampled from an 11×11 px region centred on the well.")

        gray_df = build_grayscale(img_orig, grid)
        st.markdown(grayscale_table_html(gray_df), unsafe_allow_html=True)

        xlsx_bytes = export_grayscale_excel(gray_df, img_orig, grid)
        st.download_button(
            label="⬇️ Download grayscale table (Excel)",
            data=xlsx_bytes,
            file_name="microtiter_grayscale.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ── Pseudo-absorbance table (255 − grayscale) ─────────────────────────
        st.markdown("### 6️⃣ Pseudo-absorbance per well (255 − grayscale)")
        st.caption("Inverted grayscale: higher value = darker well = stronger colorimetric reaction. "
                   "Approximates the concept of absorbance measured by a plate reader.")

        st.markdown(inverted_table_html(gray_df), unsafe_allow_html=True)

        inv_xlsx_bytes = export_inverted_excel(gray_df)
        st.download_button(
            label="⬇️ Download pseudo-absorbance table (Excel)",
            data=inv_xlsx_bytes,
            file_name="microtiter_pseudoabsorbance.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ── Estimated absorbance table ─────────────────────────────────────────
        st.markdown("### 7️⃣ Estimated absorbance per well")

        # Blank row selector
        blank_row_label = st.selectbox(
            "Blank row (lightest row = unabsorbed light reference)",
            options=ROWS,
            index=0,
            key="blank_row"
        )
        blank_row_idx = ROWS.index(blank_row_label)

        # Correction factor k — default from EXIF camera brand
        st.info(st.session_state.get("exif_msg", ""))
        k_user = st.slider(
            "Correction factor k",
            min_value=1.0, max_value=5.0,
            value=float(st.session_state.get("exif_k", K_DEFAULT)),
            step=0.1,
            key="k_correction"
        )
        st.caption(
            f"**k = {k_user:.1f}** — multiplies the raw log value to compensate for smartphone "
            "camera gamma and JPEG compression. "
            "The default value is an **empirical estimate** based on camera brand — "
            "it is not a scientifically validated constant and may vary between devices, "
            "firmware versions, and lighting conditions. "
            "For accurate results, calibrate using at least one well with a known absorbance from a plate reader: "
            "**k = A_plate_reader / A_shown_here**. "
            "Once calibrated for your specific device and conditions, the same k can be reused."
        )

        st.caption(
            f"Blank reference = mean grayscale intensity of all 12 wells in row **{blank_row_label}**. "
            f"Formula: **A = −log₁₀(gray_mean / gray_blank) × {k_user:.1f}**. "
            "Wells lighter than the blank are set to 0. "
            "Background shade: white ≈ 0, black ≥ 2."
        )

        abs_df = build_absorbance(img_orig, grid, blank_row_idx, k_user)
        st.markdown(absorbance_table_html(abs_df), unsafe_allow_html=True)

        abs_xlsx = export_absorbance_excel(abs_df)
        st.download_button(
            label="⬇️ Download absorbance table (Excel)",
            data=abs_xlsx,
            file_name="microtiter_absorbance.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        # ── Single-channel absorbance table ───────────────────────────────────
        st.markdown("### 8️⃣ Single-channel absorbance per well")
        st.caption(
            "Uses only one RGB channel instead of grayscale. "
            "This improves accuracy for dyes with a narrow absorption peak — "
            "choose the channel most sensitive to your dye: "
            "**Blue** for red/pink dyes (e.g. eosin, phenol red), "
            "**Red** for blue/purple dyes (e.g. crystal violet, coomassie), "
            "**Green** for red or violet dyes. "
            "Same blank row and k as above apply."
        )

        ch_label = st.selectbox(
            "RGB channel for absorbance calculation",
            options=["Red (0)", "Green (1)", "Blue (2)"],
            index=2,    # default: Blue
            key="abs_channel"
        )
        ch_idx = ["Red (0)", "Green (1)", "Blue (2)"].index(ch_label)

        ch_abs_df = build_absorbance_channel(
            img_orig, grid, blank_row_idx, ch_idx, k_user
        )
        st.markdown(channel_absorbance_table_html(ch_abs_df, ch_idx),
                    unsafe_allow_html=True)
        st.caption(
            f"Formula: **A = −log₁₀(ch_mean / ch_blank) × {k_user:.1f}** "
            f"using the **{CHANNEL_NAME[ch_idx]}** channel. "
            "Blank = mean of selected blank row in this channel."
        )

        ch_xlsx = export_channel_absorbance_excel(
            ch_abs_df, ch_idx, blank_row_label, k_user
        )
        st.download_button(
            label=f"⬇️ Download {CHANNEL_NAME[ch_idx]}-channel absorbance (Excel)",
            data=ch_xlsx,
            file_name=f"microtiter_absorbance_{CHANNEL_NAME[ch_idx].lower()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        with st.expander("🔬 Mean RGB values per well"):
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                ch_df = pd.DataFrame(
                    colors[:, :, ch_idx].astype(int),
                    index=ROWS, columns=[str(c) for c in COLS]
                )
                st.write(f"**Channel {ch_name}**")
                st.dataframe(ch_df, use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Start over (new image)"):
            for k in ["step","pt_a1","pt_h12","a1_x","a1_y","h12_x","h12_y",
                      "prev_zoom_a1","prev_zoom_h12","exif_k","exif_msg"]:
                st.session_state.pop(k, None)
            st.session_state.step = 1
            st.rerun()


if __name__ == "__main__":
    main()
