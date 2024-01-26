FROM python:3.9.5-slim

RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir numpy==1.26.1 \
    pandas[performance]==2.1.2 \
    dask[complete]==2023.10.1 \
    dask-ml \
    xgboost==2.0.1 \
    PyYAML \
    types-PyYAML \
    pyarrow \
    fastparquet \
    scikit-learn \
    matplotlib \
    bayesian-optimization

ENV PIPELINE_PATH="src/risknet/run/pipeline.py"

CMD ["python"]