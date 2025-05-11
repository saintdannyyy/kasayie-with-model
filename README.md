# Whisper Small Akan Non-Standard Speech Recognition API

This project provides a FastAPI-based speech recognition API designed to transcribe speech for people with speech impairments. It uses a fine-tuned Whisper model for automatic speech recognition (ASR).

## Features

- **ASR Model**: Transcribes audio files and live recordings.
- **Language Support**: Supports multiple languages, including Yoruba (`yo`), English (`en`), Twi (`tw`), and Hausa (`ha`).
- **Endpoints**:
  - `/transcribe/`: Transcribe uploaded audio files.
  - `/transcribe_live/`: Record and transcribe live audio.
  - `/health`: Check the health and status of the API.
  - `/supported_languages`: Get a list of supported languages.

## Requirements

- Python 3.8+
- Dependencies:
  - `fastapi`
  - `uvicorn`
  - `transformers`
  - `torch`
  - `numpy`
  - `pyaudio`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/saintdannyyy/kasayie-asr-api.git
   cd whisper-small-akan-non-standard-speech
   ```
