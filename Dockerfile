# Use Python image
FROM python:3.10-slim

# Install dependencies required for image processing
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy all your code and model files into the container
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit (7860) and FastAPI (8000) ports
EXPOSE 7860
EXPOSE 8000

# Start both FastAPI and Streamlit
CMD ["bash", "-c", "uvicorn main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 7860 --server.enableCORS false"]
