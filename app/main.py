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
    connection_buffers[connection_id] = []
    
    try:
        logger.info(f"WebSocket connection established: {connection_id}")
        
        # First receive the message
        try:
            # Get raw binary data directly
            message = await websocket.receive()
            logger.info(f"Received message type: {type(message)}")
            
            # Extract audio data based on message format
            audio_bytes = None
            
            # Handle binary data in "bytes" key (websockets standard)
            if isinstance(message, dict) and "bytes" in message:
                audio_bytes = message["bytes"]
                logger.info(f"Got {len(audio_bytes)} bytes from 'bytes' key")
            
            # Handle raw data
            elif not isinstance(message, dict):
                audio_bytes = message
                logger.info(f"Got raw data, type: {type(audio_bytes)}")
            
            # Handle case where message is a dict but doesn't have bytes key
            else:
                logger.info(f"Message is a dict with keys: {message.keys()}")
                if "text" in message:
                    logger.info(f"Got text message: {message['text'][:100]}")
                
                # Try to extract binary data anyway
                for k, v in message.items():
                    if isinstance(v, bytes):
                        audio_bytes = v
                        logger.info(f"Found binary data in key '{k}': {len(audio_bytes)} bytes")
                        break
            
            # Check if we have audio data
            if not audio_bytes or not isinstance(audio_bytes, bytes):
                logger.error(f"No valid audio data found. Type: {type(audio_bytes)}")
                await websocket.send_json({"error": "No valid audio data received", "type": "error"})
                return
                
            # Process the audio data with extensive error handling
            try:
                # Try to convert bytes to numpy array (WAV format)
                logger.info(f"Processing {len(audio_bytes)} bytes of audio data")
                
                # Try different audio formats in case one fails
                audio_np = None
                
                # Try int16 format (most common for WAV)
                try:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    logger.info(f"Successfully converted audio as int16. Shape: {audio_np.shape}")
                except Exception as e1:
                    logger.warning(f"Failed to convert as int16: {e1}")
                    
                    # Try float32 format
                    try:
                        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                        logger.info(f"Successfully converted audio as float32. Shape: {audio_np.shape}")
                    except Exception as e2:
                        logger.warning(f"Failed to convert as float32: {e2}")
                        
                        # Last resort - try with uint8
                        try:
                            audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 255.0
                            logger.info(f"Converted audio as uint8. Shape: {audio_np.shape}")
                        except Exception as e3:
                            logger.error(f"All audio conversion attempts failed: {e3}")
                            await websocket.send_json({"error": "Audio conversion failed", "type": "error"})
                            return
                
                # Make sure we have valid audio
                if audio_np is None or len(audio_np) < 1000:
                    logger.error(f"Audio too short or invalid: {len(audio_np) if audio_np is not None else 'None'}")
                    await websocket.send_json({"error": "Audio too short or invalid", "type": "error"})
                    return
                
                # Log audio stats
                logger.info(f"Audio array: shape={audio_np.shape}, min={audio_np.min():.4f}, max={audio_np.max():.4f}")
                
                # Perform transcription with simplified parameters
                try:
                    logger.info("Starting transcription with simplified params...")
                    
                    # Use simplified parameters to reduce chance of errors
                    segments, info = model.transcribe(
                        audio_np, 
                        beam_size=1,
                        language="en",
                        vad_filter=False  # Disable VAD to simplify processing
                    )
                    
                    # Process segments
                    segments_list = list(segments)
                    text = " ".join([segment.text for segment in segments_list]).strip()
                    logger.info(f"Transcription result: '{text}'")
                    
                    # Send success response
                    await websocket.send_json({
                        "text": text,
                        "isFinal": True,
                        "type": "transcription"
                    })
                    logger.info("Successfully sent transcription result")
                    
                except Exception as e:
                    trace = traceback.format_exc()
                    logger.error(f"Transcription failed: {e}\n{trace}")
                    await websocket.send_json({"error": f"Transcription error: {str(e)}", "type": "error"})
                
            except Exception as e:
                trace = traceback.format_exc()
                logger.error(f"Audio processing error: {e}\n{trace}")
                await websocket.send_json({"error": f"Processing error: {str(e)}", "type": "error"})
                
        except Exception as e:
            trace = traceback.format_exc()
            logger.error(f"Error receiving message: {e}\n{trace}")
            await websocket.send_json({"error": f"Message receiving error: {str(e)}", "type": "error"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {connection_id}")
    except Exception as e:
        trace = traceback.format_exc()
        logger.error(f"WebSocket error: {e}\n{trace}")
    finally:
        if connection_id in connection_buffers:
            del connection_buffers[connection_id]
        logger.info(f"WebSocket connection closed: {connection_id}")

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

# Run the FastAPI application
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")