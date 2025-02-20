# Use multi-stage build for smaller final image
FROM python:3.9-slim as builder

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Copy application code
COPY . .

# Final stage
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /root/.local /root/.local
COPY --from=builder /app .

# Install production dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "src.web.api:app", "--host", "0.0.0.0", "--port", "8000"]
