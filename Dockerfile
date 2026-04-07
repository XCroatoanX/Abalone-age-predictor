FROM python:3.14-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app and trained model
COPY app.py .
COPY abalone_age_gbr.pkl .

# Render sets $PORT for web services
CMD ["sh", "-c", "streamlit run app.py --server.headless=true --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
