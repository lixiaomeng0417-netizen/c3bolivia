# C3 Bolivia Data Explorer

Streamlit data exploration app built from the local `c3databolivia` CSV files and modeled after the ExPdPy exploration template.

## Run

```bash
cd /Users/lixiaomeng/Documents/GitHub/c3bolivia/explorer
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app includes:

- dataset selection and CSV upload
- department, province, year, range, and outlier filters
- overview, description, within/between, trends, group, composition, relationship, and dynamics views
- downloadable filtered data, config JSON, and reproducible notebook starter
