# Start from a lightweight Python 3.11 base image.
FROM python:3.11-slim

# Prevent Python from creating .pyc bytecode files inside the container.
ENV PYTHONDONTWRITEBYTECODE=1
# Force Python to flush stdout and stderr immediately for clearer logs.
ENV PYTHONUNBUFFERED=1
# Run Streamlit in headless mode so it works inside a container.
ENV STREAMLIT_SERVER_HEADLESS=true

# Set the working directory for all following commands.
WORKDIR /app

# Copy the dependency file first to improve Docker layer caching.
COPY requirements.txt .
# Install Python dependencies required by the app.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container.
COPY . .
# Create the runtime data directory used by the app for writable CSV storage.
RUN mkdir -p /app/data/runtime

# Expose the default Streamlit port.
EXPOSE 8501

# Start the Streamlit app and bind it to all network interfaces.
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
