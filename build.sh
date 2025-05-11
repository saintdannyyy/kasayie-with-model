#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y $(cat packages.txt)

# Upgrade pip
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Try to install PyAudio directly if needed
pip install PyAudio || echo "PyAudio installation failed, continuing anyway"