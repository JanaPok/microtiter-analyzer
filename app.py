"""
Microtiter Plate Analyzer – mobilní verze v2
=============================================
Instalace:
    pip install streamlit pillow numpy pandas

Spuštění:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import math
import io

# ── Konstanty ──────────────────────────────────────────────────────────────────
ROWS          = list("ABCDEFGH")
COLS          = list(range(1, 13))
N_ROWS        = len(ROWS)
N_COLS        = len(COLS)
SAMPLE_RADIUS = 5
BASE_W        = 700   # základní šířka pro zobrazení

# ── Pomocné funkce ─────────────────────────────────────────────────────────────

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
    return patch.reshape(-1, 3).mean(axis=0)


def euclidean(c1, c2):
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(c1, c2)))


def build_results(img_array, grid):
    colors = np.zeros((N_ROWS, N_COLS, 3), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            colors[r, c] = sample_color(img_array, *grid[r, c])
    distances = np.zeros((N_ROWS, N_COLS), dtype=float)
    for c in range(N_COLS):
        ref = colors[0, c]
        for r in range(N_ROWS):
            distances[r, c] = euclidean(ref, colors[r, c])
    return colors, pd.DataFrame(distances, index=ROWS, columns=[str(c) for c in COLS])


def draw_crosshair(img: Image.Image, x: int, y: int,
                   color: tuple, size: int = 30, thickness: int = 2) -> Image.Image:
    """Nakreslí průsečík (křížek) na danou pozici."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    # Vodorovná čára
    draw.line([(x - size, y), (x + size, y)], fill=color, width=thickness)
    # Svislá čára
    draw.line([(x, y - size), (x, y + size)], fill=color, width=thickness)
    # Střední tečka
    r = 5
    draw.ellipse([x-r, y-r, x+r, y+r], fill=color, outline=(255,255,255), width=1)
    return out


def draw_grid_on_image(img: Image.Image, grid: np.ndarray,
                        pt_a1=None, pt_h12=None) -> Image.Image:
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


def color_table_html(colors, dist_df) -> str:
    html = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            rgb = colors[r_idx, c_idx].astype(int)
            dist = dist_df.iloc[r_idx, c_idx]
            bg   = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            lum  = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            fg   = "#000" if lum > 128 else "#fff"
            border = "2px solid #e00" if r_idx == 0 else "1px solid #ccc"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:{border};min-width:36px;'>"
                     f"{dist:.1f}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


# ── Hlavní UI ──────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Microtiter Analyzer",
        page_icon="🧪",
        layout="centered",
        initial_sidebar_state="collapsed"
    )

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

    # ── Session state ──────────────────────────────────────────────────────────
    defaults = {"step": 1, "pt_a1": None, "pt_h12": None}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.markdown("### 📷 Nahraj fotografii destičky")
    uploaded = st.file_uploader("JPG nebo PNG", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if not uploaded:
        st.info("Vyber nebo vyfoť destičku pomocí tlačítka výše.")
        return

    img_orig  = Image.open(uploaded).convert("RGB")
    img_array = np.array(img_orig)
    OW, OH    = img_orig.width, img_orig.height
    st.success(f"Načteno: {OW}×{OH} px")

    # ── Výběr A1 ───────────────────────────────────────────────────────────────
    if st.session_state.step == 1:
        st.markdown("### 1️⃣ Nastav průsečík na jamku **A1** (levý horní roh)")

        # Zoom slider
        zoom = st.slider("🔍 Přiblížení", min_value=50, max_value=200,
                         value=100, step=10, format="%d%%", key="zoom_a1")
        disp_w = int(BASE_W * zoom / 100)
        img_d  = resize_to_width(img_orig, disp_w)
        DW, DH = img_d.width, img_d.height

        # Výchozí pozice průsečíku – ~10 % od okraje
        if "a1_x" not in st.session_state:
            st.session_state.a1_x = max(1, int(DW * 0.08))
        if "a1_y" not in st.session_state:
            st.session_state.a1_y = max(1, int(DH * 0.08))

        # Při změně zoomu přepočítej pozici průsečíku (zachovej relativní polohu)
        prev_zoom_key = "prev_zoom_a1"
        prev_zoom = st.session_state.get(prev_zoom_key, zoom)
        if prev_zoom != zoom:
            scale_change = zoom / prev_zoom
            st.session_state.a1_x = int(st.session_state.a1_x * scale_change)
            st.session_state.a1_y = int(st.session_state.a1_y * scale_change)
            st.session_state[prev_zoom_key] = zoom

        # Slidery pro X a Y
        st.caption("Pohybuj slidery pro přesné umístění průsečíku:")
        ax = st.slider("← X (vodorovně) →", min_value=0, max_value=DW,
                       value=min(st.session_state.a1_x, DW), key="sl_a1x")
        ay = st.slider("↑ Y (svisle) ↓",    min_value=0, max_value=DH,
                       value=min(st.session_state.a1_y, DH), key="sl_a1y")
        st.session_state.a1_x = ax
        st.session_state.a1_y = ay

        # Živý náhled s průsečíkem
        preview = draw_crosshair(img_d, ax, ay, color=(255, 60, 60), size=25)
        st.image(preview, use_container_width=True)
        st.caption(f"Průsečík: X={ax}, Y={ay}  (v orig: X={int(ax*OW/DW)}, Y={int(ay*OH/DH)})")

        if st.button("✅ Potvrdit A1 a pokračovat", type="primary"):
            real_x = int(ax * OW / DW)
            real_y = int(ay * OH / DH)
            st.session_state.pt_a1 = (real_x, real_y)
            st.session_state.step  = 2
            st.rerun()

    # ── Výběr H12 ──────────────────────────────────────────────────────────────
    elif st.session_state.step == 2:
        st.success(f"✅ A1 uložena: {st.session_state.pt_a1}")
        st.markdown("### 2️⃣ Nastav průsečík na jamku **H12** (pravý dolní roh)")

        zoom = st.slider("🔍 Přiblížení", min_value=50, max_value=200,
                         value=100, step=10, format="%d%%", key="zoom_h12")
        disp_w = int(BASE_W * zoom / 100)
        img_d  = resize_to_width(img_orig, disp_w)
        DW, DH = img_d.width, img_d.height

        if "h12_x" not in st.session_state:
            st.session_state.h12_x = max(1, int(DW * 0.92))
        if "h12_y" not in st.session_state:
            st.session_state.h12_y = max(1, int(DH * 0.92))

        prev_zoom_key = "prev_zoom_h12"
        prev_zoom = st.session_state.get(prev_zoom_key, zoom)
        if prev_zoom != zoom:
            scale_change = zoom / prev_zoom
            st.session_state.h12_x = int(st.session_state.h12_x * scale_change)
            st.session_state.h12_y = int(st.session_state.h12_y * scale_change)
            st.session_state[prev_zoom_key] = zoom

        st.caption("Pohybuj slidery pro přesné umístění průsečíku:")
        hx = st.slider("← X (vodorovně) →", min_value=0, max_value=DW,
                       value=min(st.session_state.h12_x, DW), key="sl_h12x")
        hy = st.slider("↑ Y (svisle) ↓",    min_value=0, max_value=DH,
                       value=min(st.session_state.h12_y, DH), key="sl_h12y")
        st.session_state.h12_x = hx
        st.session_state.h12_y = hy

        preview = draw_crosshair(img_d, hx, hy, color=(60, 220, 60), size=25)
        st.image(preview, use_container_width=True)
        st.caption(f"Průsečík: X={hx}, Y={hy}  (v orig: X={int(hx*OW/DW)}, Y={int(hy*OH/DH)})")

        if st.button("✅ Potvrdit H12 a spustit analýzu", type="primary"):
            real_x = int(hx * OW / DW)
            real_y = int(hy * OH / DH)
            st.session_state.pt_h12 = (real_x, real_y)
            st.session_state.step   = 3
            st.rerun()

        if st.button("↩️ Zpět – znovu vybrat A1"):
            st.session_state.step  = 1
            st.session_state.pt_a1 = None
            for k in ["a1_x","a1_y","h12_x","h12_y"]:
                st.session_state.pop(k, None)
            st.rerun()

    # ── Výsledky ───────────────────────────────────────────────────────────────
    elif st.session_state.step == 3:
        pt_a1  = st.session_state.pt_a1
        pt_h12 = st.session_state.pt_h12
        st.success(f"✅ A1: {pt_a1}  |  H12: {pt_h12}")

        grid = compute_grid(pt_a1, pt_h12)

        st.markdown("### 3️⃣ Detekovaná mřížka")
        annotated = draw_grid_on_image(
            resize_to_width(img_orig, BASE_W),
            compute_grid(
                (int(pt_a1[0]*BASE_W/OW),  int(pt_a1[1]*BASE_W/OW * (img_orig.height/img_orig.width * BASE_W/BASE_W))),
                (int(pt_h12[0]*BASE_W/OW), int(pt_h12[1]*BASE_W/OW * (img_orig.height/img_orig.width * BASE_W/BASE_W)))
            )
        )
        # Přesnější vykreslení mřížky v orig rozměrech → zmenšit pro zobrazení
        annotated_orig = draw_grid_on_image(img_orig, grid, pt_a1, pt_h12)
        st.image(resize_to_width(annotated_orig, BASE_W),
                 caption="Žlutá = řádek A  |  Modrá = ostatní  |  Červená = A1  |  Zelená = H12",
                 use_container_width=True)

        colors, dist_df = build_results(img_array, grid)

        st.markdown("### 4️⃣ Euklidovské vzdálenosti od řádku A")
        st.markdown(color_table_html(colors, dist_df), unsafe_allow_html=True)
        st.caption("Každá buňka = RGB vzdálenost od jamky řádku A ve stejném sloupci. Řádek A = 0.")

        st.markdown("#### Numerická tabulka")
        st.dataframe(dist_df.style.background_gradient(cmap="YlOrRd"),
                     use_container_width=True)

        csv = dist_df.to_csv(index=True).encode("utf-8")
        st.download_button("⬇️ Stáhnout CSV", data=csv,
                           file_name="microtiter_distances.csv", mime="text/csv")

        with st.expander("🔬 Průměrné RGB hodnoty jamek"):
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                ch_df = pd.DataFrame(
                    colors[:, :, ch_idx].astype(int),
                    index=ROWS, columns=[str(c) for c in COLS]
                )
                st.write(f"**Kanál {ch_name}**")
                st.dataframe(ch_df, use_container_width=True)

        st.markdown("---")
        if st.button("🔄 Začít znovu (nový snímek)"):
            for k in ["step","pt_a1","pt_h12","a1_x","a1_y","h12_x","h12_y",
                      "prev_zoom_a1","prev_zoom_h12"]:
                st.session_state.pop(k, None)
            st.session_state.step = 1
            st.rerun()


if __name__ == "__main__":
    main()
