FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy necessary files
COPY requirements.txt /app
COPY src /app/src
COPY conf/train_conf.json /app/conf/train_conf.json
COPY datasets /app/datasets

# Install any dependencies needed by our Python script
RUN pip install -U pip
RUN pip install -r requirements.txt
