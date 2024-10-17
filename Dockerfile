FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python train_model.py && \
    ls -l regression.joblib && \
    python -c "import joblib; model = joblib.load('regression.joblib'); print('Model loaded successfully')"

EXPOSE 8000

CMD ["python", "main.py"]