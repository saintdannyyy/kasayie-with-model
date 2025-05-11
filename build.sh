#!/bin/bash
# Upgrade pip
python -m pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install ffmpeg for audio processing
apt-get update && apt-get install -y ffmpeg