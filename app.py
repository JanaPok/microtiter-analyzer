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


def zoom_crop(img: Image.Image, cx: int, cy: int, zoom: int) -> tuple:
    """
    Vrátí (oříznutý+zvětšený obrázek, offset_x, offset_y).
    Při zoom=100 vrátí původní obrázek (bez ořezu).
    Při zoom>100 ořízne oblast kolem (cx,cy) a zvětší ji na BASE_W.
    offset_x/y jsou souřadnice levého horního rohu ořezu v orig. px.
    """
    if zoom <= 100:
        return img, 0, 0
    # Viditelná oblast = BASE_W * (100/zoom) px z originálu
    frac   = 100 / zoom
    crop_w = int(img.width  * frac)
    crop_h = int(img.height * frac)
    # Vycentruj kolem průsečíku, ale nepřekračuj okraje
    ox = max(0, min(cx - crop_w // 2, img.width  - crop_w))
    oy = max(0, min(cy - crop_h // 2, img.height - crop_h))
    cropped = img.crop((ox, oy, ox + crop_w, oy + crop_h))
    zoomed  = cropped.resize((img.width, img.height), Image.LANCZOS)
    return zoomed, ox, oy


def color_table_html(colors, dist_df) -> str:
    # Pro každý sloupec najdi řádek s minimální vzdáleností (ignoruj řádek A = index 0,
    # kde je vzdálenost vždy 0; hledáme minimum mezi B–H)
    col_min_row = {}
    for c_idx in range(N_COLS):
        col_vals = dist_df.iloc[1:, c_idx]   # řádky B–H
        col_min_row[c_idx] = col_vals.values.argmin() + 1  # +1 kvůli offsetu (přeskočili jsme A)

    html = '<div style="overflow-x:auto;-webkit-overflow-scrolling:touch;">'
    html += '<table style="border-collapse:collapse;font-size:11px;min-width:480px;">'
    html += "<tr><th style='padding:3px 4px;'></th>"
    for c in COLS:
        html += f"<th style='padding:3px 4px;text-align:center;'>{c}</th>"
    html += "</tr>"
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:3px 4px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            rgb  = colors[r_idx, c_idx].astype(int)
            dist = dist_df.iloc[r_idx, c_idx]
            bg   = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            lum  = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            fg   = "#000" if lum > 128 else "#fff"
            border = "2px solid #e00" if r_idx == 0 else "1px solid #ccc"
            is_min = (r_idx == col_min_row[c_idx])
            val_style = "font-weight:900;text-decoration:underline;" if is_min else ""
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:{border};min-width:36px;{val_style}'>"
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

        # Pracovní obrázek – vždy BASE_W px wide, souřadnice jsou v orig px
        img_d  = resize_to_width(img_orig, BASE_W)
        DW, DH = img_d.width, img_d.height
        sx, sy = OW / DW, OH / DH   # scale orig → display

        # Výchozí pozice průsečíku v display px (~8 % od okraje)
        if "a1_x" not in st.session_state:
            st.session_state.a1_x = max(1, int(DW * 0.08))
        if "a1_y" not in st.session_state:
            st.session_state.a1_y = max(1, int(DH * 0.08))

        # Slidery pro X a Y (v display px)
        st.caption("1. Pohybuj slidery na přibližnou polohu  2. Přibliž  3. Dolaď slidery")
        ax = st.slider("← X (vodorovně) →", 0, DW,
                       min(st.session_state.a1_x, DW), key="sl_a1x")
        ay = st.slider("↑ Y (svisle) ↓",    0, DH,
                       min(st.session_state.a1_y, DH), key="sl_a1y")
        st.session_state.a1_x = ax
        st.session_state.a1_y = ay

        zoom = st.slider("🔍 Přiblížení obrázku", 100, 600,
                         value=st.session_state.get("zoom_a1", 100),
                         step=25, format="%d%%", key="zoom_a1")

        # Zoom = crop kolem průsečíku → resize zpět na DW×DH
        zoomed, off_x, off_y = zoom_crop(img_d, ax, ay, zoom)
        # Průsečík v souřadnicích zoomedého obrázku
        ch_x = int((ax - off_x) * zoom / 100)
        ch_y = int((ay - off_y) * zoom / 100)
        preview = draw_crosshair(zoomed, ch_x, ch_y, color=(255, 60, 60), size=20)
        st.image(preview, use_container_width=True)
        st.caption(f"Pozice v orig. obrázku: X={int(ax*sx)}, Y={int(ay*sy)}")

        if st.button("✅ Potvrdit A1 a pokračovat", type="primary"):
            st.session_state.pt_a1 = (int(ax * sx), int(ay * sy))
            st.session_state.step  = 2
            st.rerun()

    # ── Výběr H12 ──────────────────────────────────────────────────────────────
    elif st.session_state.step == 2:
        st.success(f"✅ A1 uložena: {st.session_state.pt_a1}")
        st.markdown("### 2️⃣ Nastav průsečík na jamku **H12** (pravý dolní roh)")

        img_d  = resize_to_width(img_orig, BASE_W)
        DW, DH = img_d.width, img_d.height
        sx, sy = OW / DW, OH / DH

        if "h12_x" not in st.session_state:
            st.session_state.h12_x = max(1, int(DW * 0.92))
        if "h12_y" not in st.session_state:
            st.session_state.h12_y = max(1, int(DH * 0.92))

        st.caption("1. Pohybuj slidery na přibližnou polohu  2. Přibliž  3. Dolaď slidery")
        hx = st.slider("← X (vodorovně) →", 0, DW,
                       min(st.session_state.h12_x, DW), key="sl_h12x")
        hy = st.slider("↑ Y (svisle) ↓",    0, DH,
                       min(st.session_state.h12_y, DH), key="sl_h12y")
        st.session_state.h12_x = hx
        st.session_state.h12_y = hy

        zoom = st.slider("🔍 Přiblížení obrázku", 100, 600,
                         value=st.session_state.get("zoom_h12", 100),
                         step=25, format="%d%%", key="zoom_h12")

        zoomed, off_x, off_y = zoom_crop(img_d, hx, hy, zoom)
        ch_x = int((hx - off_x) * zoom / 100)
        ch_y = int((hy - off_y) * zoom / 100)
        preview = draw_crosshair(zoomed, ch_x, ch_y, color=(60, 220, 60), size=20)
        st.image(preview, use_container_width=True)
        st.caption(f"Pozice v orig. obrázku: X={int(hx*sx)}, Y={int(hy*sy)}")

        if st.button("✅ Potvrdit H12 a spustit analýzu", type="primary"):
            st.session_state.pt_h12 = (int(hx * sx), int(hy * sy))
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
        annotated_orig = draw_grid_on_image(img_orig, grid, pt_a1, pt_h12)
        st.image(resize_to_width(annotated_orig, BASE_W),
                 caption="Žlutá = řádek A  |  Modrá = ostatní  |  Červená = A1  |  Zelená = H12",
                 use_container_width=True)

        colors, dist_df = build_results(img_array, grid)

        st.markdown("### 4️⃣ Euklidovské vzdálenosti od řádku A")
        st.markdown(color_table_html(colors, dist_df), unsafe_allow_html=True)
        st.caption("Každá buňka = RGB vzdálenost od jamky řádku A ve stejném sloupci. Řádek A = 0. Tučně = nejpodobnější řádku A v daném sloupci.")

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
