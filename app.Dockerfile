# Use Python37
FROM python:3.9-slim
# Copy requirements.txt to the docker image and install packages
COPY requirements.txt requirements.txt
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install transformers
# Set the WORKDIR to be the folder
COPY main.py main.py
COPY /models /models
# Expose port 5000
EXPOSE 5000
ENV PORT 5000
WORKDIR /
# Use uvicorn as the entrypoint
CMD exec uvicorn main:app --port $PORT --host 0.0.0.0 --workers 1

