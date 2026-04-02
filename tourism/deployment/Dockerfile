# Use Python 3.10 base image (matches CI pipeline)
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy all deployment files into the container
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 7860

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
