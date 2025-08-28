# Best Available Rate Checker Web App

## 📦 Setup Guide

### Step 1 – Unpack and open folder in your code editor (VS Code recommended)

### Step 2 – Set up Python (only once)
- Install Python 3.10+
- Open terminal:
  ```
  python -m venv venv
  venv\Scripts\activate  (Windows)
  source venv/bin/activate (macOS/Linux)
  ```

### Step 3 – Install dependencies
  ```
  pip install -r requirements.txt
  playwright install
  ```

### Step 4 – Run the app locally
  ```
  streamlit run app.py
  ```

### Step 5 – Deploy to Render
- Create a GitHub repo and push this folder
- Go to https://render.com and connect the repo
- Use `web service`, build command: `pip install -r requirements.txt && playwright install`, start command: `streamlit run app.py`

### Step 6 – Password protection
- Open `.streamlit/secrets.toml` and update usernames & passwords

Enjoy!"# ratechecker" 
