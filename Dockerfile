# Use the official Python image from Docker Hub
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

# Expose port 5000 for the Flask API
EXPOSE 5000

# Start the server with Waitress
CMD ["waitress-serve", "--host=0.0.0.0", "--port=5000", "app:app"]
