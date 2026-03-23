# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy only the files you need
COPY app.py .
COPY live_testing/ ./live_testing/          
# ← important: trailing slash = copy folder
COPY data/processed ./data/processed
COPY data/models/model.pkl ./data/models/model.pkl

# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install pandas
RUN pip install scikit-learn

# Run the application
CMD ["python", "app.py"]
