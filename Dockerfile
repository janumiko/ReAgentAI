# Use an official Python runtime as a base image
FROM python:3.11-slim
# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set the working directory in the container
WORKDIR /app

## Install system dependencies
RUN apt-get update \
    && apt-get install -y \
    libgl1-mesa-glx \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the project files into the container
COPY . .
RUN mv data/docker_config.yml data/config.yml

# Install Python dependencies
RUN uv sync --locked

# Expose the port for the Gradio app
EXPOSE 7860

# Command to run the application
CMD ["uv", "run", "run.py"]
