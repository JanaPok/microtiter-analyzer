"""
Microtiter Plate Analyzer
=========================
Streamlit webová aplikace pro analýzu barev mikrotitrační destičky.

Instalace závislostí:
    pip install streamlit streamlit-image-coordinates pillow numpy pandas

Spuštění:
    streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import math

# Pokus o import streamlit-image-coordinates
try:
    from streamlit_image_coordinates import streamlit_image_coordinates
    HAS_COORDS = True
except ImportError:
    HAS_COORDS = False

# ── Konstanty ──────────────────────────────────────────────────────────────────
ROWS = list("ABCDEFGH")       # 8 řádků
COLS = list(range(1, 13))     # 12 sloupců
N_ROWS = len(ROWS)            # 8
N_COLS = len(COLS)            # 12
SAMPLE_RADIUS = 5             # poloměr oblasti pro průměr barvy (px)
MAX_DISPLAY_WIDTH = 900       # max šířka zobrazení obrázku (px)

# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def compute_grid(pt_a1: tuple, pt_h12: tuple) -> np.ndarray:
    """
    Vypočítá středové souřadnice všech 96 jamek jako rovnoměrnou mřížku.
    Vrátí pole tvaru (8, 12, 2) – [row, col, (x, y)].
    """
    x1, y1 = pt_a1
    x2, y2 = pt_h12
    grid = np.zeros((N_ROWS, N_COLS, 2), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            frac_r = r / (N_ROWS - 1)
            frac_c = c / (N_COLS - 1)
            grid[r, c, 0] = x1 + frac_c * (x2 - x1)
            grid[r, c, 1] = y1 + frac_r * (y2 - y1)
    return grid


def sample_color(img_array: np.ndarray, x: float, y: float, radius: int = SAMPLE_RADIUS) -> np.ndarray:
    """
    Vrátí průměrnou RGB hodnotu kruhové oblasti o daném poloměru kolem (x, y).
    """
    h, w = img_array.shape[:2]
    x0 = max(0, int(x) - radius)
    x1 = min(w, int(x) + radius + 1)
    y0 = max(0, int(y) - radius)
    y1 = min(h, int(y) + radius + 1)
    patch = img_array[y0:y1, x0:x1, :3]
    return patch.reshape(-1, 3).mean(axis=0)


def euclidean_distance(c1: np.ndarray, c2: np.ndarray) -> float:
    """Euklidovská vzdálenost dvou RGB barev."""
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(c1, c2)))


def build_results(img_array: np.ndarray, grid: np.ndarray):
    """
    Načte barvy ze všech jamek a vrátí:
      - colors  : (8, 12, 3) pole průměrných RGB hodnot
      - distances: DataFrame (8×12) euklidovských vzdáleností od řádku A
    """
    colors = np.zeros((N_ROWS, N_COLS, 3), dtype=float)
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            colors[r, c] = sample_color(img_array, x, y)

    # Reference = řádek A (index 0)
    distances = np.zeros((N_ROWS, N_COLS), dtype=float)
    for c in range(N_COLS):
        ref = colors[0, c]
        for r in range(N_ROWS):
            distances[r, c] = euclidean_distance(ref, colors[r, c])

    dist_df = pd.DataFrame(
        distances,
        index=ROWS,
        columns=[str(c) for c in COLS]
    )
    return colors, dist_df


def draw_grid_on_image(img: Image.Image, grid: np.ndarray) -> Image.Image:
    """
    Nakreslí detekovanou mřížku jamek přímo do obrázku (PIL).
    Vrátí nový obrázek s vizualizací.
    """
    from PIL import ImageDraw
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    r_draw = max(4, int(min(img.width, img.height) / 120))

    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            color = (255, 255, 0) if r == 0 else (0, 200, 255)
            draw.ellipse(
                [x - r_draw, y - r_draw, x + r_draw, y + r_draw],
                outline=color,
                width=2
            )
    # Zvýrazni A1 a H12
    x1, y1 = grid[0, 0]
    x2, y2 = grid[7, 11]
    draw.ellipse([x1 - r_draw*2, y1 - r_draw*2, x1 + r_draw*2, y1 + r_draw*2],
                 outline=(255, 0, 0), width=3)
    draw.ellipse([x2 - r_draw*2, y2 - r_draw*2, x2 + r_draw*2, y2 + r_draw*2],
                 outline=(255, 0, 0), width=3)
    return out


def color_table_html(colors: np.ndarray, dist_df: pd.DataFrame) -> str:
    """Vrátí HTML tabulku s barevným pozadím každé buňky."""
    html = "<table style='border-collapse:collapse;font-size:12px;'>"
    # Záhlaví
    html += "<tr><th></th>"
    for c in COLS:
        html += f"<th style='padding:4px 8px;text-align:center;'>{c}</th>"
    html += "</tr>"
    # Řádky
    for r_idx, row_label in enumerate(ROWS):
        html += f"<tr><td style='padding:4px 8px;font-weight:bold;'>{row_label}</td>"
        for c_idx in range(N_COLS):
            rgb = colors[r_idx, c_idx].astype(int)
            dist = dist_df.iloc[r_idx, c_idx]
            bg = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            # Světlost pro barvu textu
            lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
            fg = "#000" if lum > 128 else "#fff"
            border = "2px solid #f00" if r_idx == 0 else "1px solid #ccc"
            html += (
                f"<td style='background:{bg};color:{fg};padding:6px 8px;"
                f"text-align:center;border:{border};min-width:52px;'>"
                f"{dist:.1f}</td>"
            )
        html += "</tr>"
    html += "</table>"
    return html


def resize_for_display(img: Image.Image, max_width: int = MAX_DISPLAY_WIDTH) -> Image.Image:
    """Zmenší obrázek pro zobrazení, zachová poměr stran."""
    if img.width <= max_width:
        return img
    ratio = max_width / img.width
    return img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)


# ── Streamlit UI ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Microtiter Analyzer", page_icon="🧪", layout="wide")
    st.title("🧪 Microtiter Plate Analyzer")
    st.caption("Analyzuje barvy jamek mikrotitrační destičky a počítá euklidovské vzdálenosti od řádku A.")

    # ── Krok 1: Nahrání obrázku ────────────────────────────────────────────────
    st.header("1. Nahraj fotografii destičky")
    uploaded = st.file_uploader("Vyber obrázek (JPG, PNG)", type=["jpg", "jpeg", "png"])

    if not uploaded:
        st.info("📷 Na iPhonu: otevři tuto stránku v Safari, nahraj nebo vyfoť destičku.")
        return

    img = Image.open(uploaded).convert("RGB")
    img_array = np.array(img)

    st.success(f"Obrázek načten: {img.width} × {img.height} px")

    # ── Krok 2: Výběr bodů ────────────────────────────────────────────────────
    st.header("2. Klikni na jamky A1 a H12")
    st.write("Klikni nejprve na střed jamky **A1** (levý horní roh), poté na **H12** (pravý dolní roh).")

    # Uložení kliknutých bodů do session state
    if "points" not in st.session_state:
        st.session_state.points = []

    if not HAS_COORDS:
        st.warning(
            "⚠️ Knihovna `streamlit-image-coordinates` není nainstalována.\n\n"
            "Spusť: `pip install streamlit-image-coordinates` a restartuj aplikaci.\n\n"
            "Prozatím zadej souřadnice ručně níže."
        )
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Jamka A1** (levý horní roh)")
            a1x = st.number_input("A1 – X souřadnice (px)", value=0, min_value=0, max_value=img.width)
            a1y = st.number_input("A1 – Y souřadnice (px)", value=0, min_value=0, max_value=img.height)
        with col2:
            st.write("**Jamka H12** (pravý dolní roh)")
            h12x = st.number_input("H12 – X souřadnice (px)", value=img.width - 1, min_value=0, max_value=img.width)
            h12y = st.number_input("H12 – Y souřadnice (px)", value=img.height - 1, min_value=0, max_value=img.height)

        if st.button("✅ Použít tyto souřadnice"):
            st.session_state.points = [(a1x, a1y), (h12x, h12y)]

    else:
        # Interaktivní klikání do obrázku
        img_display = resize_for_display(img)
        scale_x = img.width / img_display.width
        scale_y = img.height / img_display.height

        st.write(f"Kliknutých bodů: **{len(st.session_state.points)} / 2**")
        if len(st.session_state.points) == 0:
            st.info("👆 Klikni na střed jamky **A1**")
        elif len(st.session_state.points) == 1:
            st.info("👆 Klikni na střed jamky **H12**")

        coords = streamlit_image_coordinates(img_display, key="plate_click")

        if coords is not None:
            real_x = int(coords["x"] * scale_x)
            real_y = int(coords["y"] * scale_y)
            new_pt = (real_x, real_y)
            # Přidej bod jen pokud je nový (odlišný od posledního)
            if len(st.session_state.points) == 0 or st.session_state.points[-1] != new_pt:
                if len(st.session_state.points) < 2:
                    st.session_state.points.append(new_pt)
                    st.rerun()

        if st.button("🔄 Resetovat body"):
            st.session_state.points = []
            st.rerun()

        if len(st.session_state.points) >= 1:
            st.write(f"✅ **A1**: {st.session_state.points[0]}")
        if len(st.session_state.points) >= 2:
            st.write(f"✅ **H12**: {st.session_state.points[1]}")

    # ── Krok 3: Analýza ────────────────────────────────────────────────────────
    if len(st.session_state.points) == 2:
        pt_a1 = st.session_state.points[0]
        pt_h12 = st.session_state.points[1]

        grid = compute_grid(pt_a1, pt_h12)

        # Zobraz obrázek s mřížkou
        st.header("3. Ověř detekovanou mřížku")
        annotated = draw_grid_on_image(img, grid)
        st.image(resize_for_display(annotated), caption="Žlutá = řádek A | Modrá = ostatní | Červená = A1 a H12")

        # ── Krok 4: Výsledky ──────────────────────────────────────────────────
        st.header("4. Výsledky – euklidovské vzdálenosti od řádku A")

        colors, dist_df = build_results(img_array, grid)

        # Barevná tabulka
        st.subheader("Tabulka s barevným pozadím jamek")
        st.markdown(color_table_html(colors, dist_df), unsafe_allow_html=True)
        st.caption("Číslo v každé buňce = euklidovská vzdálenost RGB od jamky v řádku A (stejný sloupec). Řádek A = reference (hodnota 0).")

        # Čistá numerická tabulka
        st.subheader("Numerická tabulka vzdáleností")
        st.dataframe(dist_df.style.background_gradient(cmap="YlOrRd"), use_container_width=True)

        # Export CSV
        st.subheader("Export")
        csv = dist_df.to_csv(index=True).encode("utf-8")
        st.download_button(
            label="⬇️ Stáhnout jako CSV",
            data=csv,
            file_name="microtiter_distances.csv",
            mime="text/csv"
        )

        # Volitelně: raw RGB hodnoty
        with st.expander("🔬 Zobrazit průměrné RGB hodnoty každé jamky"):
            for ch_idx, ch_name in enumerate(["R", "G", "B"]):
                channel_df = pd.DataFrame(
                    colors[:, :, ch_idx].astype(int),
                    index=ROWS,
                    columns=[str(c) for c in COLS]
                )
                st.write(f"**Kanál {ch_name}**")
                st.dataframe(channel_df, use_container_width=True)


if __name__ == "__main__":
    main()
