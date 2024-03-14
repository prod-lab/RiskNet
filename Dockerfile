FROM python:3.9.5-slim

RUN apt-get update && \
    apt-get install -y gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

COPY . /app

# Install any needed packages specified in setup.py or setup.cfg
RUN pip install --no-cache-dir -e .