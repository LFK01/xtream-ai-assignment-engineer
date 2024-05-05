# Parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy necessary files
COPY requirements.txt /app

# Install any dependencies needed by our Python script
RUN pip install -U pip
RUN pip install -r requirements.txt
