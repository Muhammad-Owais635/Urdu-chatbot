FROM python:3.12-slim

WORKDIR /app

# System deps kept minimal — no compiler needed for the base (non-BERT) model
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the classifier at build time so the image is ready to run immediately
RUN python train_classifier.py

EXPOSE 5000

ENV FLASK_DEBUG=false
ENV HOST=0.0.0.0
ENV PORT=5000

CMD ["python", "app.py"]
