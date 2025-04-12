# Import necessary libraries
import os
import time
from fastapi import FastAPI, WebSocket, HTTPException, Request, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
from loguru import logger
import io
from pydantic import BaseModel
from typing import Optional
import faster_whisper
import asyncio
import traceback
import wave
import tempfile

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

        # Perform transcription
        segments, info = model.transcribe(
            combined_audio,
            beam_size=5,
            language="en",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )

        segments_list = list(segments)
        text = " ".join([segment.text for segment in segments_list]).strip()
        logger.info(f"Transcribed: '{text}' from {total_duration:.2f}s audio")

        # Manage buffer
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

# WebSocket endpoint for binary audio data
@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    global model
    await websocket.accept()
    connection_id = str(id(websocket))
    
    try:
        logger.info(f"WebSocket connection established: {connection_id}")
        
        message = await websocket.receive()
        if isinstance(message, dict) and "bytes" in message:
            audio_bytes = message["bytes"]
        else:
            audio_bytes = message
        
        logger.info(f"Processing {len(audio_bytes)} bytes of audio data")
        
        try:
            # Convert to numpy array
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Get sample rate - typically 16000 for Whisper
            sample_rate = 16000
            
            # Process in 30-second chunks for better accuracy
            chunk_size = 30 * sample_rate
            transcription = []
            
            # If audio is short, process it directly
            if len(audio_np) <= chunk_size:
                logger.info("Processing entire audio as a single chunk")
                segments, info = model.transcribe(
                    audio_np,
                    beam_size=5,
                    language="en",
                    initial_prompt="This is a clear audio transcription."
                )
                text = " ".join([segment.text for segment in list(segments)]).strip()
                transcription.append(text)
            else:
                # Process longer audio in chunks
                logger.info(f"Processing audio in chunks of {chunk_size} samples")
                for i in range(0, len(audio_np), chunk_size):
                    chunk = audio_np[i:i + chunk_size]
                    logger.info(f"Processing chunk {i//chunk_size + 1}, length: {len(chunk)}")
                    
                    segments, info = model.transcribe(
                        chunk,
                        beam_size=5, 
                        language="en",
                        initial_prompt="This is a clear audio transcription."
                    )
                    
                    chunk_text = " ".join([segment.text for segment in list(segments)]).strip()
                    if chunk_text:
                        transcription.append(chunk_text)
                    
                    logger.info(f"Chunk {i//chunk_size + 1} transcription: {chunk_text}")
            
            # Combine all chunks
            final_text = " ".join(transcription)
            logger.info(f"Final transcription: {final_text}")
            
            # Send response
            await websocket.send_json({
                "text": final_text,
                "type": "transcription",
                "isFinal": True
            })
            
        except Exception as e:
            logger.error(f"Processing error: {e}", exc_info=True)
            await websocket.send_json({"error": str(e), "type": "error"})
        
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket connection closed: {connection_id}")

    if not is_valid_wav(audio_bytes):
        logger.info("Converting audio to valid WAV format")
        audio_bytes = convert_to_valid_format(audio_bytes)

async def connect_to_whisper(audio_data):
    """Connect to external Whisper service and get transcription"""
    try:
        logger.info(f"Connecting to Whisper service at {WHISPER_SERVICE_URL}")
        
        # Debug the audio data
        logger.info(f"Audio data type: {type(audio_data)}, size: {len(audio_data)} bytes")
        
        async with websockets.connect(WHISPER_SERVICE_URL) as whisper_ws:
            # Send the raw audio bytes
            await whisper_ws.send(audio_data)
            logger.info("Audio data sent to Whisper")
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(whisper_ws.recv(), timeout=15.0)
                logger.info(f"Got response from Whisper: {response[:100]}...")
                
                # Parse the response
                if isinstance(response, str):
                    try:
                        result = json.loads(response)
                        return result.get("text", "")
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON: {response[:100]}...")
                        return ""
                else:
                    logger.error(f"Unexpected response type: {type(response)}")
                    return ""
                    
            except asyncio.TimeoutError:
                logger.error("Timeout waiting for Whisper response")
                return ""
                
    except Exception as e:
        logger.error(f"Error connecting to Whisper service: {str(e)}")
        return ""

def is_valid_wav(audio_bytes):
    """Check if the audio bytes represent a valid WAV file"""
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav') as temp:
            temp.write(audio_bytes)
            temp.flush()
            with wave.open(temp.name, 'rb') as wf:
                return wf.getnchannels() > 0 and wf.getsampwidth() > 0 and wf.getframerate() > 0
    except Exception:
        return False

def convert_to_valid_format(audio_bytes):
    """Convert audio to a format Whisper can understand"""
    try:
        # If it's already a valid WAV, return as is
        if is_valid_wav(audio_bytes):
            return audio_bytes
            
        # Otherwise, try to convert using temporary files
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_in:
            temp_in.write(audio_bytes)
            temp_in.flush()
            
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_out:
            os.system(f"ffmpeg -y -i {temp_in.name} -ar 16000 -ac 1 -c:a pcm_s16le {temp_out.name} > /dev/null 2>&1")
            
            with open(temp_out.name, 'rb') as f:
                return f.read()
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        return audio_bytes  # Return original if conversion fails

# Run the FastAPI application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")