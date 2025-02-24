FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt ,
RUN pip install --no-cache-dir -r

# Copy source code
COPY . .

# Run the application
CMD ["python", "src/train.py"]