FROM python:3.9

WORKDIR /app

COPY src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/main.py .

# Create models directory for storing trained models and scalers
RUN mkdir -p models

# Create output directory for results
RUN mkdir -p output

ENV N_ESTIMATORS=100
ENV MAX_DEPTH=10
ENV TEST_SIZE=0.2
ENV RANDOM_STATE=42

CMD ["python", "main.py"]

