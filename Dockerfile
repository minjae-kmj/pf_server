# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.7-slim

RUN apt-get update
RUN apt-get install gcc -y

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True
ENV CVXOPT_BUILD_GLPK 1

# Install requirements
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy local code to the container image
COPY . /app

# Run the web service on container startup. Here we use the gunicorn
CMD exec gunicorn --bind :$PORT app:app