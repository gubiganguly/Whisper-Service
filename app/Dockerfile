FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0

WORKDIR /app

# Copy requirements and install dependencies
COPY app/requirements.txt .
RUN pip install -r requirements.txt
RUN mkdir -p /app/models

# Copy the application (without models directory)
COPY app/main.py .

# Set environment variables
ENV WHISPER_MODEL_SIZE=large-v2
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16
ENV PORT=8000

# Expose port
EXPOSE 8000

# Start the application
CMD ["python", "main.py"]