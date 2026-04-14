"""
MTT Assay Analyzer - Optimized Version
=========================
Upravená verze s automatickou optimalizací korekčního faktoru k 
na základě EXIF dat a linearity RGB kanálů.
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
SAMPLE_RADIUS = 5     
BASE_W        = 700   

# ── New Physics-Based Functions ────────────────────────────────────────────────

def get_camera_k_hint(uploaded_file):
    """Extrahuje EXIF a určí rozsah k podle modelu mobilu."""
    try:
        uploaded_file.seek(0)
        img = Image.open(uploaded_file)
        exif = img.getexif()
        make = str(exif.get(271, "")).lower()
        model = str(exif.get(272, ""))
        
        if "apple" in make:
            return (2.2, 3.3), f"📸 iPhone ({model}) - optimalizace k v rozsahu 2.2-3.3"
        elif "samsung" in make:
            return (1.9, 2.8), f"📸 Samsung ({model}) - optimalizace k v rozsahu 1.9-2.8"
        else:
            return (1.8, 4.0), "📸 Model neznámý - použit široký rozsah optimalizace"
    except:
        return (1.8, 4.0), "⚠️ EXIF data nedostupná - použit standardní rozsah"

def estimate_k_multichannel(img_orig, grid, blank_indices, k_range):
    """
    Analyzuje R, G, B kanály a vybere ten s nejlepší linearitou pro MTT.
    """
    img_array = np.array(img_orig, dtype=float)
    h, w = img_array.shape[:2]
    
    ch_data = {"R": [], "G": [], "B": []}
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            x0, x1 = max(0, int(x)-5), min(w, int(x)+6)
            y0, y1 = max(0, int(y)-5), min(h, int(y)+6)
            patch = img_array[y0:y1, x0:x1]
            for i, name in enumerate(["R", "G", "B"]):
                ch_data[name].append(np.mean(patch[:, :, i]))

    best_overall_k = 2.2
    best_overall_score = -np.inf
    best_ch_name = "G"

    for name in ["R", "G", "B"]:
        vals = np.array(ch_data[name])
        v_blank = np.median(vals[blank_indices])
        if v_blank == 0: continue
        r_vals = vals / v_blank
        
        # Maska pro validní data (střední rozsah absorbance)
        mask = (r_vals > 0.05) & (r_vals < 0.96)
        r_fit = np.sort(r_vals[mask])
        
        if len(r_fit) < 5: continue
        
        x = np.arange(len(r_fit))
        k_vals = np.linspace(k_range[0], k_range[1], 100)
        
        for k in k_vals:
            abs_v = -k * np.log10(r_fit)
            coeffs = np.polyfit(x, abs_v, 1)
            p = np.poly1d(coeffs)
            y_hat = p(x)
            
            # R2 skóre linearity
            ss_res = np.sum((abs_v - y_hat)**2)
            ss_tot = np.sum((abs_v - np.mean(abs_v))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Penalizace za nelineární výkyvy
            smoothness = np.sum(np.diff(abs_v, n=2)**2)
            score = r2 - (0.02 * smoothness)
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_k = k
                best_ch_name = name
                
    return round(float(best_overall_k), 2), best_ch_name

# ── Original Helper functions ──────────────────────────────────────────────────

def resize_to_width(img: Image.Image, width: int) -> Image.Image:
    ratio = width / img.width
    return img.resize((width, int(img.height * ratio)), Image.LANCZOS)

def compute_grid(pt_a1, pt_h12) -> np.ndarray:
    x1, y1 = pt_a1
    x2, y2 = pt_h12
    grid = np.zeros((N_ROWS, N_COLS, 2), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            grid[r, c, 0] = x1 + (c / (N_COLS - 1)) * (x2 - x1)
            grid[r, c, 1] = y1 + (r / (N_ROWS - 1)) * (y2 - y1)
    return grid

def sample_color(img_array, x, y, radius=SAMPLE_RADIUS):
    h, w = img_array.shape[:2]
    x0, x1 = max(0, int(x) - radius), min(w, int(x) + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(h, int(y) + radius + 1)
    patch = img_array[y0:y1, x0:x1, :3]
    if patch.size == 0: return np.array([0, 0, 0])
    return patch.reshape(-1, 3).mean(axis=0)

def build_results(img_array, grid, ref_row_idx: int = 0):
    colors = np.zeros((N_ROWS, N_COLS, 3), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            colors[r, c] = sample_color(img_array, *grid[r, c])
    
    row_ref = colors[ref_row_idx, :, :]
    lum_ref = 0.299*row_ref[:, 0] + 0.587*row_ref[:, 1] + 0.114*row_ref[:, 2]
    weights = lum_ref / lum_ref.sum() if lum_ref.sum() > 0 else np.ones(N_COLS)/N_COLS
    ref_rgb = (row_ref * weights[:, np.newaxis]).sum(axis=0)
    
    distances = np.zeros((N_ROWS, N_COLS), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            distances[r, c] = math.sqrt(sum((float(a)-float(b))**2 for a,b in zip(ref_rgb, colors[r,c])))
            
    dist_df = pd.DataFrame(distances, index=ROWS, columns=[str(c) for c in COLS])
    return colors, dist_df, ref_rgb

def build_grayscale(img_orig: Image.Image, grid: np.ndarray) -> pd.DataFrame:
    img_gray = img_orig.convert("L")
    gray_array = np.array(img_gray, dtype=float)
    values = np.zeros((N_ROWS, N_COLS), dtype=float)
    h, w = gray_array.shape
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            x0, x1 = max(0, int(x)-SAMPLE_RADIUS), min(w, int(x)+SAMPLE_RADIUS+1)
            y0, y1 = max(0, int(y)-SAMPLE_RADIUS), min(h, int(y)+SAMPLE_RADIUS+1)
            values[r, c] = gray_array[y0:y1, x0:x1].mean()
    return pd.DataFrame(np.round(values).astype(int), index=ROWS, columns=[str(c) for c in COLS])

def build_absorbance(gray_df: pd.DataFrame, blank_row_idx: int, k_val: float) -> pd.DataFrame:
    gray_values = gray_df.values.astype(float)
    gray_blank = np.median(gray_values[blank_row_idx, :])
    abs_values = np.zeros_like(gray_values)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            val = gray_values[r, c]
            if gray_blank > 0 and val < gray_blank and val > 0:
                abs_values[r, c] = -math.log10(val / gray_blank) * k_val
            else:
                abs_values[r, c] = 0.0
    return pd.DataFrame(np.round(abs_values, 4), index=ROWS, columns=[str(c) for c in COLS])

# ── UI Helpers ────────────────────────────────────────────────────────────────

def draw_crosshair(img: Image.Image, x: int, y: int, color: tuple, size: int = 30) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.line([(x - size, y), (x + size, y)], fill=color, width=2)
    draw.line([(x, y - size), (x, y + size)], fill=color, width=2)
    r = 5
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(255,255,255))
    return out

def zoom_crop(img: Image.Image, cx: int, cy: int, zoom: int) -> tuple:
    if zoom <= 100: return img, 0, 0
    frac = 100 / zoom
    cw, ch = int(img.width * frac), int(img.height * frac)
    ox = max(0, min(cx - cw // 2, img.width - cw))
    oy = max(0, min(cy - ch // 2, img.height - ch))
    return img.crop((ox, oy, ox + cw, oy + ch)).resize((img.width, img.height), Image.LANCZOS), ox, oy

# ── Excel Export (Zůstává beze změn pro kompatibilitu) ────────────────────────

def create_formatted_xlsx(df, title, subtitle=""):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Data', startrow=4)
        wb = writer.book
        ws = writer.sheets['Data']
        ws.merge_cells('A1:M1')
        ws['A1'] = title
        ws['A1'].font = Font(size=16, bold=True)
        ws['A1'].alignment = Alignment(horizontal='center')
    return output.getvalue()

# ── Main Application ──────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="MTT Assay Analyzer Pro", layout="centered")
    st.title("🧪 MTT Assay Analyzer Pro")

    if "step" not in st.session_state: st.session_state.step = 1

    uploaded_file = st.file_uploader("Upload plate image (JPG/PNG)", type=["jpg","jpeg","png"])
    
    if uploaded_file:
        if "exif_k_range" not in st.session_state:
            rng, msg = get_camera_k_hint(uploaded_file)
            st.session_state.exif_k_range = rng
            st.session_state.exif_msg = msg

        img_orig = Image.open(uploaded_file).convert("RGB")
        OW, OH = img_orig.width, img_orig.height

        if st.session_state.step == 1:
            st.subheader("1️⃣ Mark well A1 (Top-Left)")
            img_d = resize_to_width(img_orig, BASE_W)
            ax = st.slider("X position", 0, img_d.width, int(img_d.width*0.1))
            ay = st.slider("Y position", 0, img_d.height, int(img_d.height*0.1))
            z = st.slider("Zoom", 100, 600, 100)
            zi, ox, oy = zoom_crop(img_d, ax, ay, z)
            st.image(draw_crosshair(zi, int((ax-ox)*z/100), int((ay-oy)*z/100), (255,0,0)), use_container_width=True)
            if st.button("Confirm A1"):
                st.session_state.pt_a1 = (int(ax * OW/img_d.width), int(ay * OH/img_d.height))
                st.session_state.step = 2
                st.rerun()

        elif st.session_state.step == 2:
            st.subheader("2️⃣ Mark well H12 (Bottom-Right)")
            img_d = resize_to_width(img_orig, BASE_W)
            hx = st.slider("X position", 0, img_d.width, int(img_d.width*0.9))
            hy = st.slider("Y position", 0, img_d.height, int(img_d.height*0.9))
            z = st.slider("Zoom", 100, 600, 100)
            zi, ox, oy = zoom_crop(img_d, hx, hy, z)
            st.image(draw_crosshair(zi, int((hx-ox)*z/100), int((hy-oy)*z/100), (0,255,0)), use_container_width=True)
            if st.button("Run Analysis"):
                st.session_state.pt_h12 = (int(hx * OW/img_d.width), int(hy * OH/img_d.height))
                st.session_state.step = 3
                st.rerun()

        elif st.session_state.step == 3:
            grid = compute_grid(st.session_state.pt_a1, st.session_state.pt_h12)
            gray_df = build_grayscale(img_orig, grid)
            
            st.info(st.session_state.exif_msg)
            blank_row_label = st.selectbox("Select Blank Row", ROWS, index=0)
            
            if "auto_k" not in st.session_state:
                with st.spinner("Analyzing optical linearity..."):
                    b_idx = ROWS.index(blank_row_label)
                    b_indices = np.arange(b_idx * 12, (b_idx + 1) * 12)
                    ak, ach = estimate_k_multichannel(img_orig, grid, b_indices, st.session_state.exif_k_range)
                    st.session_state.auto_k = ak
                    st.session_state.best_ch = ach

            st.success(f"Best data source: **{st.session_state.best_ch}** channel.")
            
            k_final = st.slider("Correction factor k (gamma)", 1.0, 5.0, float(st.session_state.auto_k), 0.01)
            
            st.subheader("📊 Estimated Absorbance")
            abs_df = build_absorbance(gray_df, ROWS.index(blank_row_label), k_final)
            st.dataframe(abs_df.style.format("{:.4f}").background_gradient(cmap="Greys"), use_container_width=True)

            # Export
            xlsx_data = create_formatted_xlsx(abs_df, "MTT Absorbance Data", f"k={k_final}, Channel={st.session_state.best_ch}")
            st.download_button("⬇️ Download Excel", data=xlsx_data, file_name="mtt_results.xlsx")

            if st.button("🔄 Reset"):
                for key in ["step","pt_a1","pt_h12","auto_k","exif_k_range"]:
                    if key in st.session_state: del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()