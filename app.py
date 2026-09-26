import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# ---------------------------------------------------------------------------
# Load your trained model (place model.pkl next to this file).
# This assumes model.pkl is a full sklearn Pipeline that takes RAW columns
# (company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os)
# and that the target was trained as np.log(price) — the very common setup
# for this dataset. Adjust column names / order / the np.exp() below to match
# YOUR notebook if it differs.
# ---------------------------------------------------------------------------
MODEL_PATH = "model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

# ---------------------------------------------------------------------------
# Themes (mirrors the HTML version: Volt / Blueprint / Terminal / Nebula)
# ---------------------------------------------------------------------------
THEMES = {
    "Volt":      {"bg": "#14171C", "panel": "#1B1F26", "line": "#2B313B", "text": "#E8E6E0", "muted": "#8B93A1", "accent": "#F5C242", "ink": "#14171C"},
    "Blueprint": {"bg": "#EEF1F6", "panel": "#FFFFFF", "line": "#C7D0DC", "text": "#1B2430", "muted": "#5B6472", "accent": "#1E4FD8", "ink": "#FFFFFF"},
    "Terminal":  {"bg": "#0B120E", "panel": "#101B14", "line": "#20362A", "text": "#D8EFE0", "muted": "#75917F", "accent": "#3FE08C", "ink": "#0B120E"},
    "Nebula":    {"bg": "#161029", "panel": "#1E1836", "line": "#332B54", "text": "#EAE4FB", "muted": "#9187B5", "accent": "#B98CFF", "ink": "#161029"},
}

theme_name = st.sidebar.radio("Theme", list(THEMES.keys()), horizontal=False)
t = THEMES[theme_name]

st.markdown(f"""
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  html, body, [class*="css"] {{ font-family: 'IBM Plex Mono', monospace; }}
  h1, h2, h3, .price-num {{ font-family: 'Space Grotesk', sans-serif !important; }}
  .stApp {{ background-color: {t['bg']}; color: {t['text']}; transition: background-color .3s ease; }}
  section[data-testid="stSidebar"] {{ background-color: {t['panel']}; border-right:1px solid {t['line']}; }}
  .price-panel {{
      background:{t['panel']}; border:1px solid {t['line']}; padding:22px 20px;
  }}
  .price-num {{
      font-size:52px; font-weight:700; color:{t['accent']}; margin:8px 0 2px; line-height:1;
  }}
  .eyebrow {{ font-size:11px; color:{t['muted']}; letter-spacing:1px; }}
  .range-est {{ color:{t['muted']}; font-size:12.5px; margin-bottom:16px; }}
  .bar-lbl {{ display:flex; justify-content:space-between; font-size:11.5px; color:{t['muted']}; margin-bottom:4px; }}
  .bar-track {{ height:6px; background:{t['line']}; margin-bottom:12px; }}
  .bar-fill {{ height:100%; background:{t['accent']}; transition:width .4s ease; }}
  .note {{ margin-top:18px; font-size:11.5px; color:{t['muted']}; line-height:1.6; border-top:1px solid {t['line']}; padding-top:12px; }}
  .stButton>button {{ background:{t['accent']}; color:{t['ink']}; border:none; font-weight:700; }}
</style>
""", unsafe_allow_html=True)

st.title("Laptop Price Predictor")
st.caption("Configure a spec sheet — get a live price estimate")

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
form_col, price_col = st.columns([1.15, 0.85])

with form_col:
    st.subheader("Brand & Build")
    c1, c2 = st.columns(2)
    company = c1.selectbox("Company", ["Apple", "Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Huawei", "Google", "LG", "Samsung", "Toshiba", "Vero", "Xiaomi", "Razer", "Mediacom","Fujitsu", "Microsoft"], index=1)
    laptop_type = c2.selectbox("Type", ["Ultrabook", "Notebook", "Gaming", "2 in 1 Convertible", "Workstation", "Netbook"], index=1)

    st.subheader("Display")
    size = st.radio("Screen size", [13.3, 14, 15.6, 17.3], index=2, horizontal=True)
    resolution = st.selectbox("Resolution", ["1366x768", "1920x1080", "2560x1440", "3840x2160"], index=1)
    d1, d2 = st.columns(2)
    touchscreen = d1.checkbox("Touchscreen")
    ips = d2.checkbox("IPS panel", value=True)

    st.subheader("Performance")
    p1, p2 = st.columns(2)
    cpu = p1.selectbox("CPU", ["Intel Core i3", "Intel Core i5", "Intel Core i7", "AMD Ryzen 5", "AMD Ryzen 7", "Apple M-series"], index=1)
    gpu = p2.selectbox("GPU COMPANY", ["Nvidia", "Intel", "AMD","ARM"], index=1)
    ram = st.select_slider("RAM (GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)

    st.subheader("Storage & OS")
    s1, s2, s3 = st.columns(3)
    ssd = s1.selectbox("Primary (GB)", [0, 256, 512, 1024], index=1)
    storageType = s2.radio("Storage Type (Secondary)",["SSD","HDD","Hybrid","Flash"],horizontal=True,index = 2)
    secondary = 0
    if storageType != "None":
        secondary = s3.selectbox(f"Secondary {storageType} size (GB)", [128, 256, 500, 1000, 2000], index=2)
    os_choice = st.selectbox("Operating system", ["Windows", "macOS", "Linux", "No OS"], index=0)
    weight = st.slider("Weight (kg)", 0.0, 5.0, 1.5, 0.1)

# ---------------------------------------------------------------------------
# PLACEHOLDER estimator — swap for your trained model
# ---------------------------------------------------------------------------
def estimate_price(spec: dict):
    base = 400
    cpu_w = {"Intel Core i3": 0, "Intel Core i5": 180, "Intel Core i7": 380,
             "AMD Ryzen 5": 160, "AMD Ryzen 7": 320, "Apple M-series": 500}
    gpu_w = {"Integrated": 0, "Nvidia GTX/RTX": 220, "Nvidia RTX (high-end)": 520, "AMD Radeon": 180}
    brand_w = {"Apple": 450, "Dell": 40, "HP": 30, "Lenovo": 40, "Asus": 60, "Acer": 0, "MSI": 120}
    type_w = {"Ultrabook": 150, "Notebook": 0, "Gaming": 280, "2 in 1 Convertible": 120, "Workstation": 350}
    res_w = {"1366x768": 0, "1920x1080": 60, "2560x1440": 160, "3840x2160": 320}

    contrib = {
        "Processor": cpu_w.get(spec["cpu"], 0),
        "Graphics": gpu_w.get(spec["gpu"], 0),
        "Brand": brand_w.get(spec["company"], 0),
        "Category": type_w.get(spec["type"], 0),
        "RAM": spec["ram"] * 9,
        "Storage": spec["ssd"] * 0.35 + spec["hdd"] * 0.05,
        "Display": res_w.get(spec["res"], 0) + (40 if spec["ips"] else 0) + (70 if spec["touch"] else 0),
    }
    total = base + sum(contrib.values())
    price = round(total / 10) * 10
    return price, contrib
    
spec = {
    "company": company, "type": laptop_type, "size": size, "res": resolution,
    "touch": touchscreen, "ips": ips, "cpu": cpu, "gpu": gpu, "ram": ram,
    "primary": ssd, "secondary": secondary, "storageType":storageType, "os": os_choice, "weight": weight,
}


def predict_with_model(spec: dict):
    """Builds the raw feature row and calls your trained pipeline.
    Edit the column names/order to match exactly what you used in training."""
    x_res, y_res = (int(v) for v in spec["res"].split("x"))
    ppi = ((x_res**2 + y_res**2) ** 0.5) / spec["size"]

    cpu_company = spec["cpu"].split(" ")[0]


    query = pd.DataFrame([{
        "Company": spec["company"],
        "TypeName": spec["type"],
        "Ram_(GB)": spec["ram"],
        "Inches": spec["size"],
        "Weight_(kg)": spec["weight"],
        "Screen_Height": x_res,
        "Screen_Width": y_res,
        "Touchscreen": 1 if spec["touch"] else 0,
        "IPS": 1 if spec["ips"] else 0,
        "PPI": ppi,
        "Cpu_Company": cpu_company,
        "SSD_GB": spec["primary"],
        "Gpu_Grouped": spec["gpu"],
        "OpSys_Grouped": spec["os"],
        "Total_Storage_GB": spec['primary'] + spec['secondary'],
        spec["storageType"]+"_GB": spec['secondary'],
        "Flash_GB": spec["secondary"] if spec['storageType'] == "Flash" else 0,
        "Hybrid_GB": spec["secondary"] if spec['storageType'] == "Hybrid" else 0,
        "HDD_GB": spec["secondary"] if spec['storageType'] == "HDD" else 0,
        "SSD_GB": spec["secondary"] if spec['storageType'] == "SSD" else 0
        
    }])
    log_price = model.predict(query)[0]
    return int(np.exp(log_price))

if model is not None:
    price = predict_with_model(spec)
    contrib = None  # no per-feature breakdown for a real model unless you add one
else:
    price, contrib = estimate_price(spec)

low, high = round(price * 0.9 / 10) * 10, round(price * 1.1 / 10) * 10
max_contrib = max(contrib.values()) if contrib else 1

with price_col:
    if contrib:
        bars_html = "".join(
            f"""<div class="bar-lbl"><span>{k}</span><span>+${round(v)}</span></div>
                <div class="bar-track"><div class="bar-fill" style="width:{max(4, (v/max_contrib)*100)}%"></div></div>"""
            for k, v in contrib.items()
        )
    else:
        bars_html = f'<div class="eyebrow" style="margin-top:6px;">Predicted by your trained model ({MODEL_PATH})</div>'

    st.markdown(f"""
    <div class="price-panel">
      <div class="eyebrow">ESTIMATED PRICE</div>
      <div class="price-num">Rs.{price:,}</div>
      <div class="range-est">range ${low:,} – ${high:,}</div>
      {bars_html}
      <div class="note">
        {"This is a <b>placeholder formula</b>, not your trained model. Add <code>model.pkl</code> next to this file to switch to real predictions." if not model else "Prediction is coming from your loaded <code>model.pkl</code>. Check <code>predict_with_model()</code> if the columns don't match your training notebook."}
      </div>
    </div>
    """, unsafe_allow_html=True)
