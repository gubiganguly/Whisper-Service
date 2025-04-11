# Import necessary libraries
import os
import time
from fastapi import FastAPI, WebSocket, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
from loguru import logger
import io
from pydantic import BaseModel
from typing import Optional
import faster_whisper

# Configure logging with rotation to prevent large log files
logger.add("whisper_service.log", rotation="100 MB")

# Configure model settings from environment variables with defaults
MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v2")
# Use CUDA if available, otherwise CPU
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
# Set computation precision based on device
COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

# Global model instance
model = None
# Dictionary to store audio buffers for different connections
connection_buffers = {}

# Initialize FastAPI application
app = FastAPI(title="Whisper Transcription Service")

# Configure CORS to allow all origins, methods, and headers
# This is necessary for web clients to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
@app.on_event("startup")
async def startup_event():
    global model
    logger.info(f"Loading Whisper {MODEL_SIZE} model on {DEVICE}...")
    model = faster_whisper.WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root="./models"
    )
    logger.info(f"Model loaded successfully! Using compute type: {COMPUTE_TYPE}")
    # Log GPU info if using CUDA
    if DEVICE == "cuda":
        gpu_info = torch.cuda.get_device_properties(0)
        logger.info(f"Using GPU: {gpu_info.name} with {gpu_info.total_memory/1024**3:.2f} GB")

# Pydantic model for request validation
class TranscriptionRequest(BaseModel):
    audio_data: str  # Hex-encoded audio data
    sample_rate: int = 16000  # Default sample rate
    reset_buffer: bool = False  # Flag to reset connection buffer
    connection_id: Optional[str] = None  # Optional connection identifier

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": MODEL_SIZE, "device": DEVICE}

# Main transcription endpoint
@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    global model
    try:
        # Get or create connection ID
        conn_id = request.connection_id or "default"

        # Reset or initialize buffer for this connection
        if request.reset_buffer or conn_id not in connection_buffers:
            connection_buffers[conn_id] = []

        # Convert hex audio data to numpy array and normalize
        audio_bytes = io.BytesIO(np.frombuffer(
            bytes.fromhex(request.audio_data), dtype=np.int16
        ).astype(np.float32) / 32768.0)

        audio_np = np.frombuffer(audio_bytes.getvalue(), dtype=np.float32)
        connection_buffers[conn_id].append(audio_np)

        # Combine all audio chunks in buffer
        combined_audio = np.concatenate(connection_buffers[conn_id])
        total_duration = len(combined_audio) / request.sample_rate

        # Skip processing if audio is too short
        if total_duration < 0.5:
            return {
                "text": "",
                "isFinal": False,
                "buffer_duration": total_duration
            }

        # Perform transcription with VAD (Voice Activity Detection)
        segments, info = model.transcribe(
            combined_audio,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            word_timestamps=True
        )

        segments_list = list(segments)

        # Fallback transcription with different parameters if no segments detected
        if not segments_list:
            segments, info = model.transcribe(
                combined_audio,
                beam_size=1,
                language="en",
                temperature=0.2,
                no_speech_threshold=0.3,
                vad_filter=False
            )
            segments_list = list(segments)

        # Combine all segments into final text
        text = " ".join([segment.text for segment in segments_list]).strip()
        logger.info(f"Transcribed: '{text}' from {total_duration:.2f}s audio")

        # Keep only the last 0.5 seconds of audio in buffer
        if total_duration > 0.5:
            last_samples = int(0.5 * request.sample_rate)
            if len(combined_audio) > last_samples:
                connection_buffers[conn_id] = [combined_audio[-last_samples:]]
            else:
                connection_buffers[conn_id] = [combined_audio]

        return {
            "text": text,
            "isFinal": len(text) > 0,
            "language": info.language,
            "language_probability": info.language_probability
        }

    except Exception as e:
        logger.error(f"Error in transcription: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time transcription
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    global model
    await websocket.accept()
    connection_id = str(id(websocket))
    connection_buffers[connection_id] = []

    try:
        logger.info(f"WebSocket connection established: {connection_id}")

        while True:
            data = await websocket.receive()

            # Handle metadata messages
            if "text" in data:
                metadata = data["text"]
                logger.debug(f"Received metadata: {metadata}")
                continue

            # Handle audio data
            elif "bytes" in data:
                # Convert and normalize audio data
                audio_bytes = data["bytes"]
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                connection_buffers[connection_id].append(audio_np)
                combined_audio = np.concatenate(connection_buffers[connection_id])

                # Skip processing if audio buffer is too small
                if len(combined_audio) < 8000:
                    continue

                # Perform transcription
                segments, info = model.transcribe(
                    combined_audio,
                    beam_size=5,
                    language="en",
                    vad_filter=True
                )

                segments_list = list(segments)
                text = " ".join([segment.text for segment in segments_list]).strip()

                # Send transcription result if text is not empty
                if text:
                    logger.info(f"WebSocket transcription: '{text}'")
                    await websocket.send_json({
                        "text": text,
                        "isFinal": True,
                        "type": "transcription"
                    })

                    # Keep only the last 8000 samples in buffer
                    connection_buffers[connection_id] = [combined_audio[-8000:]] if len(combined_audio) > 8000 else [combined_audio]

    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)

    finally:
        # Clean up connection buffer when WebSocket closes
        if connection_id in connection_buffers:
            del connection_buffers[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")

# Run the FastAPI application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)