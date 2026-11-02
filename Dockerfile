
# Use a lightweight Python 3.12 image
FROM python:3.12-slim

# Set Python environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the sandbox working directory
WORKDIR /src

# Install pinned testing dependencies
RUN pip install --no-cache-dir \
    pytest==9.2.0 \
    pytest-json-report==1.5.0

# Create a non-root user
RUN groupadd --gid 10001 sandbox && \
    useradd --uid 10001 --gid 10001 --no-create-home sandbox

# Run containers as a non-root user
USER 10001:10001

# Default command
CMD ["python3"]
