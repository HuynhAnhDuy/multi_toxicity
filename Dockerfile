# Base image: Python 3.10 on Debian slim
FROM python:3.10-slim

# Install Java and system tools
RUN apt-get update && apt-get install -y \
    default-jre \
    build-essential \
    wget \
    unzip \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose default Streamlit port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
