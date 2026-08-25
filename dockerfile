FROM python:3.11-slim

WORKDIR /app

COPY app.py .
COPY live_testing/ ./live_testing/          
COPY data/processed/test.csv ./data/processed/
COPY data/models/model.pkl ./data/models/model.pkl

RUN pip install pandas scikit-learn

CMD ["python", "app.py"]
