from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import torch
import librosa
import soundfile as sf
import wave
import tempfile
import os
import io
import uvicorn
import logging
import time
from pathlib import Path
from transformers import pipeline
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_PATH = "Saintdannyyy/kasayie-asr"
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global variables
asr_pipe = None
model_loading = False

# Try to import PyAudio, but continue if not available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    logger.info("PyAudio is available - live recording enabled")
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available - live recording will be disabled")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup and shutdown."""
    global asr_pipe, model_loading

    # Startup logic: Load the ASR model
    model_loading = True
    try:
        logger.info(f"Loading ASR model from {MODEL_PATH}")
        logger.info(f"Using device: {DEVICE}")
        
        asr_pipe = pipeline(
            "automatic-speech-recognition",
            model=MODEL_PATH,
            device=DEVICE
        )
        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {e}")
        asr_pipe = None
    finally:
        model_loading = False

    yield  # Application runs here

    # Shutdown logic (if needed)
    logger.info("Shutting down application")

# Attach the lifespan handler to the app
app = FastAPI(
    title="Speech Recognition API for People with Speech Impairments",
    description="API for transcribing speech for people with speech impairments",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates and static file configuration
templates = Jinja2Templates(directory="templates")
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define data models
class TranscriptionRequest(BaseModel):
    language: str = "yo"  # Default to Yoruba

class TranscriptionResponse(BaseModel):
    text: str
    language: str
    success: bool
    processing_time: float = None

class StatusResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    languages: list

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the main frontend page."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...), language: str = Form("yo")):
    """
    Transcribe uploaded audio file using librosa.
    
    Args:
        file: The audio file to transcribe
        language: Language code for transcription
    
    Returns:
        TranscriptionResponse with the transcribed text
    """
    global asr_pipe
    
    if asr_pipe is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="ASR model is currently loading, please try again in a moment")
        else:
            raise HTTPException(status_code=503, detail="ASR model not loaded")
    
    try:
        start_time = time.time()
        
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
        
        # Process with librosa
        audio, sr = librosa.load(temp_file_path, sr=SAMPLE_RATE)
        
        # Use the ASR model
        result = asr_pipe(
            {"array": audio, "sampling_rate": sr},
            generate_kwargs={"language": language}
        )
        
        # Clean up
        os.unlink(temp_file_path)
        
        transcription = result["text"].strip()
        process_time = time.time() - start_time
        
        logger.info(f"Transcription successful: {transcription[:50]}...")
        
        return TranscriptionResponse(
            text=transcription,
            language=language,
            success=True,
            processing_time=round(process_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

@app.post("/transcribe_live/", response_model=TranscriptionResponse)
async def transcribe_live_audio(seconds: int = Form(5), language: str = Form("yo")):
    """
    Record and transcribe live audio.
    
    Args:
        seconds: Number of seconds to record (default: 5)
        language: Language code for transcription (default: yo for Yoruba)
    
    Returns:
        TranscriptionResponse with the transcribed text
    """
    # Check if PyAudio is available
    if not PYAUDIO_AVAILABLE:
        raise HTTPException(
            status_code=400, 
            detail="Live recording is not available in this deployment. Please use the file upload method instead."
        )
    
    global asr_pipe
    
    if asr_pipe is None:
        if model_loading:
            raise HTTPException(status_code=503, detail="ASR model is currently loading, please try again in a moment")
        else:
            raise HTTPException(status_code=503, detail="ASR model not loaded")
    
    # Validate parameters
    if seconds < 1 or seconds > 30:
        raise HTTPException(status_code=400, detail="Recording duration must be between 1 and 30 seconds")
    
    try:
        start_time = time.time()
        
        # Record audio
        logger.info(f"Starting {seconds}s recording")
        audio_frames = record_audio(duration=seconds, sample_rate=SAMPLE_RATE)
        if audio_frames is None:
            raise HTTPException(status_code=500, detail="Failed to record audio")
        
        # Convert to numpy array
        combined_audio = b''.join(audio_frames)
        audio_np = np.frombuffer(combined_audio, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run inference
        result = asr_pipe(
            {"array": audio_np, "sampling_rate": SAMPLE_RATE},
            generate_kwargs={"language": language}
        )
        
        transcription = result["text"].strip()
        process_time = time.time() - start_time
        
        logger.info(f"Live transcription successful: {transcription[:50]}...")
        
        return TranscriptionResponse(
            text=transcription,
            language=language,
            success=True,
            processing_time=round(process_time, 2)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Live transcription error: {e}")
        raise HTTPException(status_code=500, detail=f"Live transcription failed: {str(e)}")

def record_audio(duration=5, sample_rate=16000, channels=1, audio_format=pyaudio.paInt16):
    """Record audio for N seconds, then return frames."""
    if not PYAUDIO_AVAILABLE:
        return None
        
    p = None
    stream = None
    frames = []
    try:
        logger.info(f"Recording {duration} seconds of audio")
        p = pyaudio.PyAudio()
        chunk = 1024
        stream = p.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
        
        # Wait a brief moment to initialize the microphone
        time.sleep(0.2)
        
        chunks_to_record = int((sample_rate / chunk) * duration)
        for i in range(chunks_to_record):
            data = stream.read(chunk, exception_on_overflow=False)
            frames.append(data)
            
        logger.info("Recording completed")
    except Exception as e:
        logger.error(f"Recording error: {e}")
        return None
    finally:
        if stream:
            stream.stop_stream()
            stream.close()
        if p:
            p.terminate()
    return frames

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """API health check endpoint with detailed status."""
    return StatusResponse(
        status="healthy" if asr_pipe is not None else "loading" if model_loading else "not_loaded",
        model_loaded=asr_pipe is not None,
        device=DEVICE,
        languages=["yo", "en", "yo", "ha"]  # List supported languages
    )

@app.get("/supported_languages")
async def supported_languages():
    """Return list of supported languages for the ASR model."""
    return {
        "languages": [
            {"code": "yo", "name": "Yoruba"},
            {"code": "en", "name": "English"},
            {"code": "yo", "name": "Twi"},
            {"code": "ha", "name": "Hausa"}
        ]
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)