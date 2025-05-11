#!/bin/bash
# Upgrade pip
python -m pip install --upgrade pip

# Install Python dependencies only
pip install -r requirements.txt

# Don't try to install system packages - Render handles this differently
# apt-get update && apt-get install -y ffmpeg