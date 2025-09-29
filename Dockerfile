# Multi-stage build for MLB DFS Tools
# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./
RUN npm ci --production=false

# Copy frontend source
COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 2: Setup Python backend with frontend
FROM python:3.12-slim

# Install Node.js for any runtime needs
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy Python requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY src/ src/
COPY pipeline/ pipeline/

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/dist frontend/dist

# Create necessary directories
RUN mkdir -p output dk_data fd_data ikb_data

# Set Python path
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start command
CMD ["uvicorn", "src.dashboard_api:app", "--host", "0.0.0.0", "--port", "8000"]