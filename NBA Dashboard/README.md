# NBA Game Performance Prediction Dashboard 🏀

An interactive Streamlit dashboard that visualizes NBA player stats and predicts individual game points using a trained XGBoost model.

## 📊 Features

- Select any NBA player from the 2024–25 season
- View their game-by-game points and core stats
- Predict points based on custom input stats
- Built with Streamlit, Pandas, Plotly, and XGBoost

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/your-username/nba-dashboard-app.git
cd nba-dashboard-app

# Set up a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/main.py
