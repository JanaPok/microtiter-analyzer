# MTT Assay Analyzer

A mobile-friendly, hardware-free web application for quantitative colorimetric analysis of MTT assay 96-well plates using an ordinary smartphone photograph.

> **No plate reader required.** Upload a photo, align two wells, and receive a full suite of colorimetric outputs — directly in your browser on any device.

---

## Overview

Quantitative analysis of MTT assay 96-well plates conventionally requires a dedicated plate reader spectrophotometer — an instrument costing USD 10,000–50,000. MTT Assay Analyzer replaces this hardware with a smartphone camera and a web browser. The user photographs the plate, marks two reference wells (A1 and H12), and the application automatically detects all 96 well positions and computes multiple colorimetric metrics, including an estimated absorbance value.

The application is built with [Streamlit](https://streamlit.io), runs fully in-browser with no installation on the user's device, and works on iOS, Android, and desktop.

---

## Features at a glance

| Feature | Description |
|---|---|
| **Automatic well detection** | Mark wells A1 and H12; remaining 94 positions interpolated automatically |
| **Interactive zoom** | Up to 600 % zoom for precise crosshair placement on small screens |
| **EXIF camera detection** | Reads camera brand from photo metadata and suggests an appropriate correction factor |
| **7 analysis outputs** | From RGB distances to gamma-corrected calibrated absorbance |
| **Excel export** | Every table downloadable as a formatted `.xlsx` file with coloured cell backgrounds |
| **Optional calibration** | Enter 2–4 plate-reader reference values; app fits a linear model and calibrates all wells |

---

## How it works

### Step 1 — Upload a photograph

Take a photo of the plate with any smartphone (JPG or PNG). Uniform, diffuse overhead lighting gives the best results. The app reads the EXIF metadata to detect the camera brand and suggest a correction factor *k*.

### Step 2 — Align the well grid

Use the X/Y sliders to position a red crosshair on well **A1** (top-left) and a green crosshair on well **H12** (bottom-right). A zoom slider (100–600 %) lets you magnify the image for precise placement. The remaining 94 well centres are computed by linear interpolation, which holds for standard SBS-format 96-well plates.

### Step 3 — Review the detected grid

The app overlays all 96 detected positions on the image for visual verification before analysis.

### Step 4 — Read the results

Seven tables are computed automatically:

#### 4️⃣ Euclidean RGB distances from reference row
The user selects which row serves as the reference (typically the blank/control row). Each cell contains the Euclidean distance in RGB space between that well and the luminance-weighted mean RGB of all 12 wells in the selected reference row:

```
d = sqrt( (R1−R2)² + (G1−G2)² + (B1−B2)² )
```

The reference row is highlighted with a red border in the table.

#### 5️⃣ Pseudo-absorbance
Simple inversion of grayscale intensity: darker well = higher value = stronger reaction.

```
PA = 255 − L   where L = 0.299·R + 0.587·G + 0.114·B
```

#### 6️⃣ Estimated absorbance (grayscale-based)
Lambert–Beer approximation using a user-selected blank row and a camera correction factor *k*:

```
A = −log₁₀(L_well / L_blank) × k
```

The blank row (the lightest row, typically the negative control) is selected by the user. The factor *k* is pre-set from the camera brand detected in the photo EXIF data, and can be adjusted with a slider.

> **Note:** The default *k* values per brand are empirical estimates, not scientifically validated constants. For accurate results, calibrate: *k = A_plate_reader / A_shown_here*.

#### 7️⃣ Gamma-corrected absorbance with optional calibration

The most accurate method. Applies the full sRGB gamma correction before computing transmittance, using equal weights across R, G, B channels:

```
I_linear = ((pixel/255 + 0.055) / 1.055)^2.2   ← invert sRGB gamma
T_eff    = mean(T_R, T_G, T_B)                  ← mean transmittance
A        = −log₁₀(T_eff)
```

**Optional linear calibration:** Enter 2–4 wells whose absorbance you measured on a plate reader (non-blank wells only). The app fits `A_cal = slope × A_raw + intercept` and applies it to all wells. The blank row is automatically excluded from calibration point selection.

> Based on experimental validation across four independent plates, this approach achieves Pearson r > 0.99 and R² > 0.99 compared to a commercial spectrophotometer under standardised imaging conditions.

---

## Installation (local)

**Requirements:** Python 3.9 or higher

```bash
# 1. Clone the repository
git clone https://github.com/Pokornz/mtt_assay_analyzer.git
cd mtt_assay_analyzer

# 2. Install dependencies
pip install streamlit pillow numpy pandas openpyxl

# 3. Run
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

---

## Online access

The app is deployed on Streamlit Cloud — no installation needed:

> 🔗 **[your-app-url.streamlit.app](https://your-app-url.streamlit.app)**
> *(update this link after deploying)*

**iPhone:** Safari → Share → "Add to Home Screen"  
**Android:** Chrome → Menu → "Add to Home Screen"

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| [Streamlit](https://streamlit.io) | ≥ 1.32 | Web application framework |
| [Pillow](https://python-pillow.org) | ≥ 10.0 | Image loading, processing, EXIF reading |
| [NumPy](https://numpy.org) | ≥ 1.24 | Array operations, gamma correction |
| [Pandas](https://pandas.pydata.org) | ≥ 2.0 | Tabular output |
| [openpyxl](https://openpyxl.readthedocs.io) | ≥ 3.1 | Excel export with coloured cells |

---

## Limitations

- Results depend on **illumination quality**. Uniform, diffuse overhead lighting is strongly recommended. Avoid direct sunlight, shadows cast by the camera, and strong reflections from the plate lid.
- The method captures **broadband RGB light**, not monochromatic light at the dye's absorption peak. This introduces a systematic underestimation of absorbance that is partially corrected by gamma correction and channel weighting, but cannot be fully eliminated without a bandpass filter or calibration.
- **Gamma and EXIF correction factors** are empirical estimates based on camera brand. They may vary between firmware versions and shooting conditions. For quantitative results, always calibrate against at least two plate-reader reference values.
- **Well detection** relies on manual alignment of two corner wells. Plates that are significantly rotated or perspective-distorted may produce less accurate grids. Photograph the plate as flat-on as possible.
- The sampling region is fixed at **11 × 11 pixels** around each well centre. For very small or irregularly spaced wells, or very low-resolution photographs, this may include pixels outside the well boundary.

---

## Contributing

Contributions are welcome. Please open an issue to discuss proposed changes before submitting a pull request.

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## Citation

If you use MTT Assay Analyzer in your research, please cite:

```
Pokornz. (2025). MTT Assay Analyzer (Version 1.0) [Software].
GitHub. https://github.com/Pokornz/mtt_assay_analyzer
```

*A formal citation with DOI will be added upon publication.*

---

## Author

Developed by **[your name]**, [your institution], [year].  
Contact: [your email or GitHub profile]
