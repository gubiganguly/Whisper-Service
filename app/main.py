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
import asyncio
import websockets
import json

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
        
        # Main processing loop
        while True:
            try:
                # Wait for data with a timeout
                data = await asyncio.wait_for(websocket.receive(), timeout=30.0)
                logger.info(f"Received data type: {type(data)}")
                
                # Process different message types
                if "text" in data:
                    # Handle text messages (metadata)
                    metadata = data["text"]
                    logger.debug(f"Received text metadata: {metadata}")
                    
                elif "bytes" in data:
                    # Handle binary audio data
                    audio_bytes = data["bytes"]
                    logger.info(f"Received {len(audio_bytes)} bytes of audio data")
                    
                    # Convert to float32 and normalize
                    # Try different data types since we don't know the input format
                    try:
                        # Try as int16 first
                        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    except:
                        try:
                            # Try as float32
                            audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                        except:
                            # Last resort: just try to make it work
                            audio_np = np.frombuffer(audio_bytes, dtype=np.uint8).astype(np.float32) / 255.0
                    
                    # Add to buffer
                    connection_buffers[connection_id].append(audio_np)
                    
                    # Combine audio chunks (up to a reasonable length)
                    combined_audio = np.concatenate(connection_buffers[connection_id])
                    
                    # Skip processing if audio buffer is too small
                    if len(combined_audio) < 4000:
                        logger.info(f"Audio too short ({len(combined_audio)} samples), waiting for more data")
                        await websocket.send_json({
                            "text": "",
                            "isFinal": False,
                            "type": "transcription"
                        })
                        continue
                    
                    # Log buffer information
                    logger.info(f"Processing {len(combined_audio)} samples")
                    
                    # Perform transcription
                    try:
                        segments, info = model.transcribe(
                            combined_audio,
                            beam_size=5,
                            language="en",
                            vad_filter=True
                        )
                        
                        segments_list = list(segments)
                        text = " ".join([segment.text for segment in segments_list]).strip()
                        
                        # Send transcription result
                        logger.info(f"WebSocket transcription: '{text}'")
                        await websocket.send_json({
                            "text": text,
                            "isFinal": True,
                            "type": "transcription"
                        })
                        
                        # Keep only the last portion of audio in buffer to maintain context
                        if len(combined_audio) > 8000:
                            connection_buffers[connection_id] = [combined_audio[-8000:]]
                        else:
                            connection_buffers[connection_id] = [combined_audio]
                            
                    except Exception as e:
                        logger.error(f"Transcription error: {str(e)}")
                        await websocket.send_json({
                            "error": "Transcription failed",
                            "type": "error"
                        })
                
                else:
                    # Unknown message format
                    logger.warning(f"Unknown message format received: {data.keys() if isinstance(data, dict) else 'not a dict'}")
                    # Try to interpret as raw audio if it's not a dict
                    if not isinstance(data, dict):
                        try:
                            # Assume it's raw binary data
                            audio_bytes = data
                            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            connection_buffers[connection_id].append(audio_np)
                            
                            # Process as above
                            combined_audio = np.concatenate(connection_buffers[connection_id])
                            if len(combined_audio) >= 4000:
                                segments, info = model.transcribe(
                                    combined_audio,
                                    beam_size=5,
                                    language="en",
                                    vad_filter=True
                                )
                                
                                segments_list = list(segments)
                                text = " ".join([segment.text for segment in segments_list]).strip()
                                
                                logger.info(f"WebSocket transcription from raw data: '{text}'")
                                await websocket.send_json({
                                    "text": text,
                                    "isFinal": True,
                                    "type": "transcription"
                                })
                                
                                if len(combined_audio) > 8000:
                                    connection_buffers[connection_id] = [combined_audio[-8000:]]
                                else:
                                    connection_buffers[connection_id] = [combined_audio]
                        except Exception as e:
                            logger.error(f"Failed to process unknown data format: {str(e)}")
                            await websocket.send_json({
                                "error": "Unknown data format",
                                "type": "error"
                            })
                
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for data, closing connection")
                await websocket.send_json({
                    "error": "Connection timeout",
                    "type": "error"
                })
                break
                
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}", exc_info=True)
        try:
            await websocket.send_json({
                "error": str(e),
                "type": "error"
            })
        except:
            pass

    finally:
        # Clean up connection buffer when WebSocket closes
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
    uvicorn.run(app, host="0.0.0.0", port=port)