# 💻 Laptop Price Predictor

A Streamlit web app that estimates the price of a laptop from its specifications. Pick a brand, screen, CPU, GPU, RAM, storage and OS, and get a live price prediction from a trained machine-learning model.

---

## ✨ Features

- **Live predictions**: the estimate updates as you change any spec.
- **Rich spec inputs**: brand, laptop type, screen size, resolution, touchscreen, IPS panel, CPU, GPU, RAM, primary and secondary storage, OS and weight.
- **Automatic feature engineering**: computes **PPI** (pixels per inch) from resolution and screen size, and derives the CPU brand and total storage for you.
- **Price range**: shows the predicted price with a ±10% band.
- **4 built-in themes**: Volt, Blueprint, Terminal and Nebula, switchable from the sidebar.
- **Graceful fallback**: if `model.pkl` is missing, the app uses a simple placeholder formula so the UI still works.

---

## 🗂️ Project Structure

```
.
├── app.py          # Streamlit app (UI + prediction logic)
├── model.pkl       # Trained scikit-learn pipeline (log-price target)
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd <your-repo-folder>
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit numpy pandas scikit-learn
```

> ⚠️ Use the **same scikit-learn version** you used to train and pickle the model. Pickles are not guaranteed to load across different versions. Pin it in a `requirements.txt`, for example `scikit-learn==1.x.x`.

### 4. Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**.

---

## 🧠 How It Works

1. The user's selections are collected into a `spec` dictionary.
2. `predict_with_model()` converts them into a single-row `pandas.DataFrame` with the same columns the model was trained on.
3. The pipeline predicts the **log of the price**, and the app converts it back with `np.exp()`.
4. The result is displayed as a price with a ±10% range.

### Model input features

| Feature | Description |
|---|---|
| `Company` | Laptop brand |
| `TypeName` | Ultrabook, Notebook, Gaming, etc. |
| `Ram_(GB)` | RAM size |
| `Inches` | Screen size |
| `Weight_(kg)` | Weight |
| `Screen_Height`, `Screen_Width` | Resolution |
| `Touchscreen`, `IPS` | Binary display flags |
| `PPI` | Pixels per inch, computed from resolution and size |
| `Cpu_Company` | Intel / AMD / Apple |
| `SSD_GB`, `HDD_GB`, `Hybrid_GB`, `Flash_GB` | Storage by type |
| `Total_Storage_GB` | Primary + secondary storage |
| `Gpu_Grouped` | GPU manufacturer |
| `OpSys_Grouped` | Operating system group |

> The column names and order in `predict_with_model()` must match your training notebook exactly.

---

## 🎨 Themes

Choose a theme from the sidebar:

| Theme | Style |
|---|---|
| **Volt** | Dark charcoal with yellow accent |
| **Blueprint** | Light with blue accent |
| **Terminal** | Dark green, hacker style |
| **Nebula** | Deep purple |

To add your own, add an entry to the `THEMES` dictionary in `app.py`.

---

## 🛠️ Tech Stack

- [Streamlit](https://streamlit.io/) for the frontend
- [scikit-learn](https://scikit-learn.org/) for the model pipeline
- [pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data handling

---

## 📄 License
