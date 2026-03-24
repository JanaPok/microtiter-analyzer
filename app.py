import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# === NOVÁ FUNKCE ABSORBANCE ===
def calculate_absorbance(pseudoabs_matrix):
    """Vypočítá absorbance z pseudoabsorbance matice"""
    absorbance_matrix = np.zeros_like(pseudoabs_matrix)
    empirical_matrix = np.zeros_like(pseudoabs_matrix)
    
    for i in range(pseudoabs_matrix.shape[0]):
        for j in range(pseudoabs_matrix.shape[1]):
            PA = pseudoabs_matrix[i,j]
            
            # Logaritmická (s auto-blank korekci)
            gray = 255 - PA
            T = gray / 255
            A_log = -np.log10(np.clip(T, 0.01, 1))  # Ořez proti nule
            
            # Tvůj empirický vzorec  
            A_emp = (PA / 140.46) ** 1.4996
            
            absorbance_matrix[i,j] = A_log * 1.3  # Mobil korekce
            empirical_matrix[i,j] = A_emp
    
    return absorbance_matrix, empirical_matrix

# === TABULKA + HEATMAP ===
def display_absorbance_table(absorbance_matrix, empirical_matrix):
    # Vytvoř DataFrame pro 96-well (8x12)
    df_abs = pd.DataFrame(absorbance_matrix, 
                         index=[f'{chr(65+i)}' for i in range(8)],
                         columns=range(1,13))
    df_emp = pd.DataFrame(empirical_matrix,
                         index=[f'{chr(65+i)}' for i in range(8)],
                         columns=range(1,13))
    
    # Streamlit tabulky
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Absorbance (logaritmická + korekce)")
        st.dataframe(df_abs.round(3), use_container_width=True)
    
    with col2:
        st.subheader("📊 Tvůj empirický vzorec")
        st.dataframe(df_emp.round(3), use_container_width=True)
    
    # Heatmap
    fig = make_subplots(rows=1, cols=1, subplot_titles='Absorbance Heatmap')
    fig.add_trace(go.Heatmap(z=absorbance_matrix, 
                            colorscale='RdYlBu_r',
                            text=absorbance_matrix.round(2),
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            colorbar=dict(title="A")),
                 row=1, col=1)
    fig.update_layout(height=500, title_text="96-well Absorbance Map")
    st.plotly_chart(fig, use_container_width=True)

# === INTEGRACE DO TVÉHO KÓDU ===
# Po tvém výpočtu pseudoabs_matrix:
if 'pseudoabs_matrix' in locals():
    abs_matrix, emp_matrix = calculate_absorbance(pseudoabs_matrix)
    display_absorbance_table(abs_matrix, emp_matrix)
    
    # Statistiky
    st.metric("Průměr A", f"{abs_matrix.mean():.3f}")
    st.metric("Max A", f"{abs_matrix.max():.3f}")
    st.metric("CV (%)", f"{abs_matrix.std()/abs_matrix.mean()*100:.1f}%")
