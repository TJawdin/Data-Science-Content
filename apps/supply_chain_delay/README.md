# Supply Chain Delay Risk Predictor

Predict late-delivery risk for Brazilian e-commerce orders with a LightGBM model trained on the Olist dataset. The app supports single predictions, batch scoring, time trends, and a geographic view—plus PDF reports.

---

## Quickstart

```bash
# 1) Create & activate a virtual environment (Python 3.11+ recommended)
python -m venv .venv                 # create venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies (pinned)
pip install -r requirements.txt      # installs Streamlit, LightGBM, SHAP, etc.

# 3) Run the app from the project root
streamlit run apps/supply_chain_delay/app.py
