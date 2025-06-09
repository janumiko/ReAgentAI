# Use an official Python runtime as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Create a non-root user and group
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . .
RUN mv data/docker_config.yml data/config.yml

# Copy the uv binary into the container
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create cache and local directories for uv and set permissions
RUN mkdir -p /home/appuser/.cache/uv && \
    mkdir -p /home/appuser/.local/share/uv && \
    chown -R appuser:appuser /home/appuser

# Set ownership of the working directory to the non-root user
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Install Python dependencies
RUN uv sync --locked

# Expose the port for the Gradio app
EXPOSE 7860

# Command to run the application
CMD ["uv", "run", "run.py"]