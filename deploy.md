# Deployment Guide — FORESIGHT on Streamlit Community Cloud

## Prerequisites

Before deploying, confirm all three pipeline steps have been run and their
output files exist locally:

```
data/processed/panel.parquet          ← python -m src.pipeline   (Teammate 1)
data/processed/forecasts.parquet      ← python -m src.forecast   (Teammate 2)
data/processed/risk_scores.parquet    ← python -m src.risk       (Teammate 2)
```

The repository must be pushed to GitHub — either public, or private with the
Zidio mentor added as a collaborator.

---

## Step 1 — Commit the Processed Parquet Files

The `data/processed/` folder is listed in `.gitignore` by default to avoid
committing large files. For deployment, the three Parquet files must be force-
added so Streamlit Cloud can load them without running the pipeline on the server.

```bash
git add -f data/processed/panel.parquet
git add -f data/processed/forecasts.parquet
git add -f data/processed/risk_scores.parquet
git commit -m "feat: add processed parquet files for Streamlit Cloud deployment"
git push origin main
```

> **Why not run the pipeline on Streamlit Cloud?**
> Streamlit Cloud's free tier has limited CPU and memory. LightGBM training
> on 130,000+ rows during cold start is unreliable. Committing the pre-built
> outputs is simpler and faster.

---

## Step 2 — Deploy on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with your GitHub account.
2. Click **"New app"**.
3. Select your **repository** from the dropdown.
4. Set **Branch** to `main`.
5. Set **Main file path** to: `app/main.py`
6. Click **"Deploy!"**
7. Wait approximately 2 minutes for the first build to complete.
8. Your app URL will be in the format:
   `https://[your-github-username]-foresight-app-main-[hash].streamlit.app`

> Copy this URL — you need it for the README.md placeholders, the executive
> readout, and the Zidio submission form.

---

## Step 3 — Smoke-Test Checklist

Run through every item below **before** recording the demo video and before
the final M4 checkpoint. Do this with a **cold browser** (private/incognito
window) to simulate a new user loading the app for the first time.

### App loads
- [ ] Navigate to the public URL in a private browser tab
- [ ] App loads without a spinner error or `ModuleNotFoundError`
- [ ] Sidebar shows all four navigation items

### Overview page
- [ ] Decisioning scatter grid renders with all four coloured quadrant groups
- [ ] KPI strip shows non-zero values for at least two metrics
- [ ] Category filter updates both the grid and the table
- [ ] Priority action table is sorted by ₹ value at stake (highest first)
- [ ] "Download Full Risk Report" button downloads a valid CSV file

### Forecast page
- [ ] SKU selector dropdown loads (most important SKU pre-selected)
- [ ] Select `SKU0001` — forecast chart renders with 4 traces:
      actual history, confidence band, baseline dashed line, LightGBM line
- [ ] Vertical "Forecast Start" line is visible
- [ ] Forecast data table shows 8 rows
- [ ] Download button downloads a CSV for the selected SKU

### Risk & Actions page
- [ ] All four tabs are present
- [ ] 🔴 Reorder Now tab shows SKUs and a download button
- [ ] 🔵 Markdown / Clear tab shows SKUs and a download button
- [ ] Category filter updates all four tabs
- [ ] Empty tabs show a success message (not an error)

### Score a SKU page
- [ ] Text input is visible and empty on first load
- [ ] Type `SKU0042` → results appear: 6 metric cards, forecast table, JSON block
- [ ] Recommended action callout is visible
- [ ] JSON block shows `forecast_8_weeks` array with 8 entries
- [ ] Type `INVALID999` → red error message appears + JSON error response
      (app does NOT crash or show a stack trace)

### URL-based scoring (D6 criterion)
- [ ] Navigate to: `[your-app-url]?sku_id=SKU0100`
- [ ] Score a SKU page loads with `SKU0100` pre-filled in the input
- [ ] Results appear immediately without manual input

### Downloads
- [ ] Download CSV from Overview — opens in a spreadsheet with correct columns
- [ ] Download Reorder Plan CSV from Risk page — correct columns, correct SKUs only

---

## Troubleshooting Common Deployment Failures

### Error: `FileNotFoundError: data/processed/forecasts.parquet`
**Cause:** Parquet files were not committed to git.
**Fix:** Run the `git add -f` commands in Step 1 and push again.

### Error: `ModuleNotFoundError: No module named 'app'`
**Cause:** Import path issue — Streamlit Cloud may not recognise `app/` as a package.
**Fix:** Add the following lines to the top of any page file that throws this error:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

### Error: `ModuleNotFoundError: No module named 'lightgbm'` (or any other package)
**Cause:** Package missing from `requirements.txt`.
**Fix:** Add the missing package to `requirements.txt` and push:
```
lightgbm>=4.0
```

### App loads but charts are blank / empty DataFrames
**Cause:** Parquet files have no data, or column names don't match what the page expects.
**Fix:** Check the exact column names against the spec in the Teammate 3 guide.
Run `pd.read_parquet("data/processed/risk_scores.parquet").columns.tolist()` locally
to verify.

### App runs locally but fails on Cloud with a different error
**Cause:** Version mismatch between local environment and Streamlit Cloud.
**Fix:** Pin your package versions in `requirements.txt`:
```
pandas==2.1.4
numpy==1.26.2
lightgbm==4.1.0
streamlit==1.32.0
plotly==5.18.0
pyarrow==14.0.2
```

---

## Updating the App After a Data Refresh

Whenever Teammate 1 or 2 re-runs the pipeline with new data:

```bash
# Re-run the pipeline
python -m src.pipeline
python -m src.forecast
python -m src.risk

# Force-add the updated Parquet files and push
git add -f data/processed/panel.parquet
git add -f data/processed/forecasts.parquet
git add -f data/processed/risk_scores.parquet
git commit -m "chore: refresh processed parquet files with updated data"
git push origin main
```

Streamlit Community Cloud detects the push to `main` and automatically
redeploys within approximately 2 minutes. No manual action on the Cloud
dashboard is needed.