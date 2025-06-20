# Gunakan base image yang sudah support TensorFlow
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Set workdir
WORKDIR /app

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy all source code
COPY . .

# Expose port for Railway
EXPOSE 8080

# Jalankan streamlit, gunakan $PORT dari Railway
CMD streamlit run app.py --server.port $PORT --server.address 0.0.0.0