# Use Python37
FROM python:3.7
# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt requirements.txt
COPY setup.py setup.py
RUN pip install -r requirements.txt --no-cache-dir
# Set the WORKDIR to be the folder
COPY . /app
# Expose port 5000
EXPOSE 5000
ENV PORT 5000
WORKDIR /app
# Use gunicorn as the entrypoint
# CMD exec gunicorn --bind :$PORT main:app --workers 1 --threads 1 --timeout 0

CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1

