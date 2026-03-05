"""
Microtiter Plate Analyzer – mobilní verze
==========================================
Instalace:
    pip install streamlit pillow numpy pandas

Spuštění:
    streamlit run app.py
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import math
import base64
import io

# ── Konstanty ──────────────────────────────────────────────────────────────────
ROWS          = list("ABCDEFGH")
COLS          = list(range(1, 13))
N_ROWS        = len(ROWS)
N_COLS        = len(COLS)
SAMPLE_RADIUS = 5
MAX_W         = 800

# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def resize_for_display(img: Image.Image, max_w: int = MAX_W) -> Image.Image:
    if img.width <= max_w:
        return img
    ratio = max_w / img.width
    return img.resize((max_w, int(img.height * ratio)), Image.LANCZOS)


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


def draw_grid_on_image(img: Image.Image, grid: np.ndarray) -> Image.Image:
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    r_draw = max(4, int(min(img.width, img.height) / 120))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            x, y = grid[r, c]
            color = (255, 220, 0) if r == 0 else (0, 200, 255)
            draw.ellipse([x-r_draw, y-r_draw, x+r_draw, y+r_draw], outline=color, width=2)
    x1, y1 = grid[0, 0]
    x2, y2 = grid[7, 11]
    s = r_draw * 2
    draw.ellipse([x1-s, y1-s, x1+s, y1+s], outline=(255, 60, 60), width=3)
    draw.ellipse([x2-s, y2-s, x2+s, y2+s], outline=(60, 255, 60), width=3)
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
            bg = f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"
            lum = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            fg = "#000" if lum > 128 else "#fff"
            border = "2px solid #e00" if r_idx == 0 else "1px solid #ccc"
            html += (f"<td style='background:{bg};color:{fg};padding:4px 3px;"
                     f"text-align:center;border:{border};min-width:36px;'>"
                     f"{dist:.1f}</td>")
        html += "</tr>"
    html += "</table></div>"
    return html


def crosshair_component(img_display: Image.Image, label: str,
                         color: str, def_x: int, def_y: int) -> dict | None:
    """
    Vykreslí interaktivní canvas s posuvným průsečíkem.
    Vrátí {'x': px, 'y': py} v souřadnicích orig. obrázku po stisknutí tlačítka.
    scale_x a scale_y jsou zakódovány přímo do HTML.
    """
    b64 = pil_to_b64(img_display)
    dw, dh = img_display.width, img_display.height

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ background:#0e1117; font-family:sans-serif; }}
#wrap {{
  position:relative; display:block;
  width:100%; max-width:{dw}px;
  touch-action:none; user-select:none; -webkit-user-select:none;
}}
#img {{ display:block; width:100%; height:auto; border-radius:8px; }}
#ch {{
  position:absolute; pointer-events:none;
  transform:translate(-50%,-50%);
  width:0; height:0;
}}
.h {{ position:absolute; height:2px; width:48px; background:{color};
      top:0; left:0; transform:translate(-50%,-50%) translateY(0); margin-left:-24px; margin-top:-1px; }}
.v {{ position:absolute; width:2px; height:48px; background:{color};
      top:0; left:0; transform:translate(-50%,-50%); margin-left:-1px; margin-top:-24px; }}
.dot {{ position:absolute; width:10px; height:10px; border-radius:50%;
        background:{color}; box-shadow:0 0 8px 3px {color}99;
        margin-left:-5px; margin-top:-5px; }}
#lbl {{
  position:absolute; top:6px; left:8px;
  background:{color}dd; color:#000; font-weight:bold;
  font-size:13px; padding:2px 8px; border-radius:6px;
}}
#info {{ margin:6px 0; font-size:12px; color:#888; text-align:center; }}
#btn {{
  display:block; width:92%; max-width:320px;
  margin:8px auto 4px; padding:14px;
  background:{color}; color:#000; font-weight:bold;
  font-size:16px; border:none; border-radius:10px;
  cursor:pointer; box-shadow:0 4px 14px {color}55;
}}
</style>
</head>
<body>
<div id="wrap">
  <img id="img" src="data:image/jpeg;base64,{b64}" draggable="false">
  <div id="ch" style="left:{def_x}px;top:{def_y}px;">
    <div class="h"></div>
    <div class="v"></div>
    <div class="dot"></div>
  </div>
  <div id="lbl">{label}</div>
</div>
<div id="info">X: <span id="cx">?</span> &nbsp;|&nbsp; Y: <span id="cy">?</span></div>
<button id="btn">✅ Potvrdit &nbsp;{label}</button>

<script>
const img = document.getElementById('img');
const ch  = document.getElementById('ch');
const cxEl = document.getElementById('cx');
const cyEl = document.getElementById('cy');
const btn  = document.getElementById('btn');

// Pozice průsečíku v "procentech" šířky/výšky obrázku
// (takto je to nezávislé na skutečné velikosti renderu)
let pctX = {def_x} / {dw};
let pctY = {def_y} / {dh};

function clamp(v,lo,hi){{ return Math.max(lo,Math.min(hi,v)); }}

function applyPos() {{
  const r = img.getBoundingClientRect();
  const px = pctX * r.width;
  const py = pctY * r.height;
  ch.style.left = px + 'px';
  ch.style.top  = py + 'px';
  cxEl.textContent = Math.round(pctX * {dw});
  cyEl.textContent = Math.round(pctY * {dh});
}}

function onDown(e) {{
  e.preventDefault();
  move(e);
}}

function move(e) {{
  e.preventDefault();
  const r = img.getBoundingClientRect();
  let cx, cy;
  if (e.touches) {{
    cx = e.touches[0].clientX - r.left;
    cy = e.touches[0].clientY - r.top;
  }} else {{
    cx = e.clientX - r.left;
    cy = e.clientY - r.top;
  }}
  pctX = clamp(cx / r.width,  0, 1);
  pctY = clamp(cy / r.height, 0, 1);
  applyPos();
}}

img.addEventListener('touchstart', onDown, {{passive:false}});
document.addEventListener('touchmove',  move,  {{passive:false}});

img.addEventListener('mousedown', onDown);
document.addEventListener('mousemove', e => {{ if (e.buttons) move(e); }});

// Inicializace po načtení obrázku
img.addEventListener('load', applyPos);
if (img.complete) applyPos();

btn.addEventListener('click', () => {{
  window.parent.postMessage({{
    type: 'streamlit:setComponentValue',
    value: {{ x: Math.round(pctX * {dw}), y: Math.round(pctY * {dh}) }}
  }}, '*');
}});
</script>
</body>
</html>"""

    return components.html(html, height=dh + 120, scrolling=False)


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
      .block-container { padding: 1rem 0.6rem 2rem !important; max-width:100% !important; }
      h1 { font-size:1.4rem !important; }
      h2, h3 { font-size:1.05rem !important; }
      .stButton>button, .stDownloadButton>button {
        width:100%; padding:0.75rem; font-size:1rem; border-radius:10px;
      }
      iframe { border-radius:10px; }
    </style>
    """, unsafe_allow_html=True)

    st.title("🧪 Microtiter Analyzer")

    # Session state
    for key, val in [("step", 1), ("pt_a1", None), ("pt_h12", None)]:
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.markdown("### 📷 Nahraj fotografii destičky")
    uploaded = st.file_uploader("JPG nebo PNG", type=["jpg","jpeg","png"],
                                label_visibility="collapsed")
    if not uploaded:
        st.info("Vyber nebo vyfoť destičku pomocí tlačítka výše.")
        return

    img_orig    = Image.open(uploaded).convert("RGB")
    img_display = resize_for_display(img_orig)
    img_array   = np.array(img_orig)
    scale_x     = img_orig.width  / img_display.width
    scale_y     = img_orig.height / img_display.height

    st.success(f"Načteno: {img_orig.width}×{img_orig.height} px")

    # Výchozí souřadnice průsečíků v display px
    dw, dh = img_display.width, img_display.height
    DEF_A1_X  = min(70,  dw - 10)
    DEF_A1_Y  = min(70,  dh - 10)
    DEF_H12_X = max(dw - 80, 10)
    DEF_H12_Y = max(dh - 80, 10)

    # ── Krok 1: A1 ────────────────────────────────────────────────────────────
    if st.session_state.step == 1:
        st.markdown("### 1️⃣ Nastav červený průsečík na jamku **A1**")
        st.caption("Táhni prstem nebo myší. Pak potvrď tlačítkem.")
        val = crosshair_component(img_display, "A1", "#ff4444", DEF_A1_X, DEF_A1_Y)
        if val and isinstance(val, dict) and "x" in val:
            st.session_state.pt_a1 = (int(val["x"] * scale_x), int(val["y"] * scale_y))
            st.session_state.step = 2
            st.rerun()

    # ── Krok 2: H12 ───────────────────────────────────────────────────────────
    elif st.session_state.step == 2:
        st.success(f"✅ A1 uložena: {st.session_state.pt_a1}")
        st.markdown("### 2️⃣ Nastav zelený průsečík na jamku **H12**")
        st.caption("Táhni prstem nebo myší. Pak potvrď tlačítkem.")
        val = crosshair_component(img_display, "H12", "#44ff88", DEF_H12_X, DEF_H12_Y)
        if val and isinstance(val, dict) and "x" in val:
            st.session_state.pt_h12 = (int(val["x"] * scale_x), int(val["y"] * scale_y))
            st.session_state.step = 3
            st.rerun()
        if st.button("↩️ Znovu vybrat A1"):
            st.session_state.step = 1
            st.session_state.pt_a1 = None
            st.rerun()

    # ── Krok 3: Výsledky ──────────────────────────────────────────────────────
    elif st.session_state.step == 3:
        pt_a1  = st.session_state.pt_a1
        pt_h12 = st.session_state.pt_h12
        st.success(f"✅ A1: {pt_a1}  |  H12: {pt_h12}")

        grid = compute_grid(pt_a1, pt_h12)

        st.markdown("### 3️⃣ Detekovaná mřížka")
        annotated = draw_grid_on_image(img_orig, grid)
        st.image(resize_for_display(annotated),
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
        if st.button("🔄 Začít znovu"):
            st.session_state.step = 1
            st.session_state.pt_a1 = None
            st.session_state.pt_h12 = None
            st.rerun()


if __name__ == "__main__":
    main()
