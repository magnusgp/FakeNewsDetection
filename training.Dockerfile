# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc wget curl python3.9 && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install dvc 'dvc[gs]'

COPY requirements.txt requirements.txt
COPY setup.py setup.py
COPY src/ src/
COPY .dvc/config .dvc/config
COPY data/processed.dvc data/processed.dvc
COPY run_training.sh run_training.sh

WORKDIR /
RUN python3.9 -m pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["./run_training.sh"]