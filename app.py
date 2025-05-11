from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request, BackgroundTasks
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import numpy as np
import torch
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

# Model configuration - Now using the Hugging Face model path
MODEL_PATH = "Saintdannyyy/kasayie-asr"  # HuggingFace model ID
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Try to import PyAudio, but continue if not available
try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
    # Audio recording constants
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    logger.info("PyAudio is available - live recording enabled")
except ImportError:
    PYAUDIO_AVAILABLE = False
    logger.warning("PyAudio not available - live recording will be disabled")

# Global variables
asr_pipe = None
model_loading = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for app startup and shutdown."""
    global asr_pipe, model_loading

    # Startup logic: Load the ASR model
    model_loading = True
    try:
        logger.info(f"Loading ASR model from HuggingFace: {MODEL_PATH}")
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

@app.post("/transcribe/", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...), language: str = Form("yo")):
    """
    Transcribe uploaded audio file.
    
    Args:
        file: The audio file to transcribe
        language: Language code for transcription (default: yo for Yoruba)
    
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
        
        # Check file size (10MB limit)
        file_size = 0
        file_content = await file.read()
        file_size = len(file_content)
        if file_size > 10 * 1024 * 1024:  # 10MB
            raise HTTPException(status_code=413, detail="Audio file too large (max 10MB)")
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(file_content)
            temp_path = temp_file.name
        
        try:
            # For WAV files, convert to numpy array
            try:
                with wave.open(temp_path, 'rb') as wav_file:
                    # Check audio parameters
                    channels = wav_file.getnchannels()
                    sample_width = wav_file.getsampwidth()
                    framerate = wav_file.getframerate()
                    
                    logger.info(f"Audio file parameters: channels={channels}, sample_width={sample_width}, framerate={framerate}")
                    
                    frames = wav_file.readframes(wav_file.getnframes())
                    audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    # Resample if needed (basic resampling - for production you might want a better library)
                    if framerate != SAMPLE_RATE:
                        logger.warning(f"Resampling from {framerate} to {SAMPLE_RATE}")
                        # This is a very basic resampling method - consider librosa for better quality
                        target_length = int(len(audio_np) * SAMPLE_RATE / framerate)
                        audio_np = np.interp(
                            np.linspace(0, len(audio_np), target_length), 
                            np.arange(len(audio_np)), 
                            audio_np
                        )
            except Exception as e:
                # If WAV parsing failed, try to treat as another format (will need additional libraries)
                logger.error(f"Error parsing WAV: {e}, file might not be in WAV format")
                raise HTTPException(status_code=415, detail=f"Audio format error: {str(e)}. Please use WAV format.")
            
            # Run inference with timeout protection
            try:
                result = asr_pipe(
                    {"array": audio_np, "sampling_rate": SAMPLE_RATE}, 
                    generate_kwargs={"language": language}
                )
                
                transcription = result["text"].strip()
                process_time = time.time() - start_time
                
                logger.info(f"Transcription successful: {transcription[:50]}...")
                
                return TranscriptionResponse(
                    text=transcription,
                    language=language,
                    success=True,
                    processing_time=round(process_time, 2)
                )
                
            except Exception as e:
                logger.error(f"Inference error: {e}")
                raise HTTPException(status_code=500, detail=f"Transcription processing error: {str(e)}")
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error removing temp file: {e}")
    
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

def record_audio(duration=5, sample_rate=16000, channels=1, audio_format=None):
    """Record audio for N seconds, then return frames."""
    if not PYAUDIO_AVAILABLE:
        return None
        
    p = None
    stream = None
    frames = []
    try:
        logger.info(f"Recording {duration} seconds of audio")
        p = pyaudio.PyAudio()
        chunk = CHUNK
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
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
        languages=["yo", "en","ha"]  # List supported languages
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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)