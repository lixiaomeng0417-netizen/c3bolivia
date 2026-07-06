# C3 Bolivia Data Explorer

Streamlit data exploration app built from the local `c3databolivia` CSV files and modeled after the ExPdPy exploration template.

## Privacy

Run this app locally. Do not deploy it to Streamlit Cloud, GitHub Pages, or another public hosting service unless the CSV files in `c3databolivia` are intentionally public.

The app does not upload data to an external service. It reads local CSV files from `../c3databolivia`, and the file-upload control has been removed to avoid accidental data sharing.

## Run

```bash
cd /Users/lixiaomeng/Documents/GitHub/c3bolivia/explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app includes:

- local dataset selection
- department, province, year, range, and outlier filters
- local data catalog for all CSV and GeoJSON files in `c3databolivia`
- overview, description, within/between, trends, group, composition, relationship, dynamics, and GDP per capita deep-dive views
- panel exploration for GDP per capita, population, and night-time lights
- downloadable filtered data, config JSON, and reproducible notebook starter
