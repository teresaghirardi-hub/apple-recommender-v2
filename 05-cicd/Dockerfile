FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Train both models at build time so they are ready when the API starts
RUN python 02-experiment-tracking/train.py
RUN python 02-experiment-tracking/train_revenue.py

EXPOSE 8000

# Serve the FastAPI app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "03-deployment"]
