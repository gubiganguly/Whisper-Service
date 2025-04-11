# whisper_client.py
import pyaudio
import numpy as np
import requests
import threading
import time
import queue
import os
from datetime import datetime

# Configuration
API_URL = "https://9s1ha2i2tz2z9t-8000.proxy.runpod.net"
CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 300  # Adjust based on your environment

class WhisperClient:
    def __init__(self):
        self.running = False
        self.audio_queue = queue.Queue()
        self.transcription = ""
        self.connection_id = datetime.now().strftime("%Y%m%d%H%M%S")
        
    def start_recording(self):
        self.running = True
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        print("Recording started. Speak into your microphone...")
        print("Press Ctrl+C to stop")
        
        # Main recording loop
        try:
            while self.running:
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.audio_queue.put(data)
                time.sleep(0.01)  # Small sleep to reduce CPU usage
        except KeyboardInterrupt:
            self.stop_recording()
    
    def stop_recording(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        print("\nRecording stopped.")
    
    def process_audio(self):
        buffer = []
        last_transcription_time = time.time()
        
        while self.running or not self.audio_queue.empty():
            # Get data from queue
            if not self.audio_queue.empty():
                data = self.audio_queue.get()
                buffer.append(data)
            
            current_time = time.time()
            # Process every 1 second
            if current_time - last_transcription_time > 1 and buffer:
                audio_data = b''.join(buffer)
                self.transcribe(audio_data)
                last_transcription_time = current_time
                # Keep last 0.5 seconds to maintain context
                buffer = buffer[-2:] if len(buffer) > 2 else buffer
            
            time.sleep(0.1)
    
    def transcribe(self, audio_data):
        # Convert to numpy array then to hex
        np_audio = np.frombuffer(audio_data, dtype=np.int16)
        hex_audio = np_audio.tobytes().hex()
        
        try:
            response = requests.post(
                f"{API_URL}/transcribe",
                json={
                    "audio_data": hex_audio,
                    "sample_rate": RATE,
                    "connection_id": self.connection_id
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("text") and result.get("text") != self.transcription:
                    self.transcription = result.get("text")
                    # Clear screen and show transcription
                    os.system('cls' if os.name == 'nt' else 'clear')
                    print("Transcription:")
                    print("=" * 50)
                    print(self.transcription)
                    print("=" * 50)
        except Exception as e:
            print(f"Error during transcription: {e}")

if __name__ == "__main__":
    client = WhisperClient()
    
    print(f"Connecting to Whisper service at {API_URL}")
    try:
        health_response = requests.get(f"{API_URL}/health")
        print(f"Service status: {health_response.json()}")
    except Exception as e:
        print(f"Error connecting to service: {e}")
        exit(1)
    
    client.start_recording()