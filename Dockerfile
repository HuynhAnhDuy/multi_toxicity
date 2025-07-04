FROM python:3.10-slim

# Install Java
RUN apt-get update && apt-get install -y default-jre

# Install essential system libraries (add more here if needed)
RUN apt-get install -y build-essential

# Set the working directory
WORKDIR /app

# Copy all source code into the container
COPY . .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=10000", "--server.address=0.0.0.0"]
