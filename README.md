# 🏎 PitWall Analytics Dashboard

**B2C F1 Performance Data Platform — Subscriber Retention Intelligence**

---

## 📁 Required Folder Structure (IMPORTANT)

Your GitHub repo **must** look exactly like this:

```
your-repo/
├── app.py
├── data_generator.py
├── model_utils.py
├── theme.py
├── tab1_descriptive.py
├── tab2_diagnostic.py
├── tab3_predictive.py
├── tab4_prescriptive.py
├── requirements.txt
├── README.md
├── .streamlit/
│   └── config.toml          ← MUST be inside .streamlit folder
└── data/
    └── PitWall_Analytics_Cleaned.xlsx   ← MUST be inside data folder
```

---

## 🚀 Deploy on Streamlit Cloud (Step-by-Step)

1. **Upload everything to GitHub** keeping the folder structure exactly as above.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Connect your repo · set **Main file path** → `app.py` · Branch → `main`
4. Click **Deploy** — done. No code changes needed.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
