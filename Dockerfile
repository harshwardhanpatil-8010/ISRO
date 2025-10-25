# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies required for geospatial libraries like GDAL for geopandas and rasterio
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    gdal-bin \
    libspatialindex-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Create and switch to a new non-root user for security
RUN useradd -ms /bin/bash appuser

# Install Python dependencies
# We copy the requirements file first to leverage Docker's layer caching.
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY --chown=appuser:appuser . .

# Create directories for logs, cache, and outputs and set permissions
RUN mkdir -p logs cache outputs && chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose the port that Streamlit runs on
EXPOSE 8501

# Add a healthcheck to ensure the streamlit app is running
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# The command to run the application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]