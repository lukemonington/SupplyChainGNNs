# Use the official Python 3.11 image as the base
FROM python:3.11.11-slim

# Set the working directory
WORKDIR /app

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch 2.6.0 CPU-only version and other dependencies
RUN pip install --no-cache-dir \
    torch==2.6.0 \
    torch-geometric==2.6.1 \
    flask==3.1.0 \
    matplotlib==3.10.0 \
    pandas==2.2.2 \
    scikit-learn==1.6.1

# Copy the application files
COPY saved_models/ /app/saved_models/
COPY deployment_assets/ /app/deployment_assets/
COPY inference.py .
COPY test_api.py .
COPY test_api_with_synthetic_data.py .


# Expose the port
EXPOSE 5000

# Run the API
CMD ["python", "inference.py"]
