#!/usr/bin/env python3
"""
Model Structure Checker - A utility to verify the structure of a HuggingFace model directory.

This script checks if a given directory contains the necessary files for a HuggingFace model,
specifically checking for essential files like model.safetensors, config.json, tokenizer.json, etc.
"""

import os
import argparse
import logging
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("model-checker")

def check_model_structure(model_path):
    """
    Check if the given directory contains necessary files for a HuggingFace model.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        bool: True if the structure is valid, False otherwise
    """
    # Convert to Path object for easier handling
    model_dir = Path(model_path)
    
    # Check if directory exists
    if not model_dir.exists():
        logger.error(f"Model directory does not exist: {model_dir}")
        return False
    
    if not model_dir.is_dir():
        logger.error(f"Path is not a directory: {model_dir}")
        return False
    
    # Define required files for different model types
    required_files = {
        "standard": ["config.json"],
        "pytorch": ["pytorch_model.bin", "config.json"],
        "safetensors": ["model.safetensors", "config.json"],
        "tokenizer": ["tokenizer.json", "tokenizer_config.json"],
    }
    
    # Check for different model configurations
    found_configs = []
    
    # Check for standard config
    if all(model_dir.joinpath(f).exists() for f in required_files["standard"]):
        found_configs.append("standard")
    
    # Check for PyTorch model
    if all(model_dir.joinpath(f).exists() for f in required_files["pytorch"]):
        found_configs.append("pytorch")
    
    # Check for SafeTensors model
    if all(model_dir.joinpath(f).exists() for f in required_files["safetensors"]):
        found_configs.append("safetensors")
    
    # Check for tokenizer
    if any(model_dir.joinpath(f).exists() for f in required_files["tokenizer"]):
        found_configs.append("tokenizer")
    
    # List all files in the directory for debugging
    logger.info(f"Files found in {model_dir}:")
    for file in model_dir.iterdir():
        logger.info(f"  - {file.name}")
    
    # Check if we found any valid configurations
    if not found_configs:
        logger.error(f"No valid model configuration found in {model_dir}")
        logger.error("Required files not found. Expected one of:")
        for config_name, files in required_files.items():
            logger.error(f"  - {config_name}: {', '.join(files)}")
        return False
    
    logger.info(f"Found valid configurations: {', '.join(found_configs)}")
    return True

def fix_whisper_model(model_path):
    """
    Attempt to fix common issues with Whisper models.
    
    Args:
        model_path: Path to the model directory
    
    Returns:
        bool: True if fixes were applied, False otherwise
    """
    model_dir = Path(model_path)
    
    # Check for common nested structure issue
    subdirs = [d for d in model_dir.iterdir() if d.is_dir()]
    
    # If there's exactly one subdirectory and it contains model files
    if len(subdirs) == 1:
        subdir = subdirs[0]
        subdir_files = list(subdir.iterdir())
        
        # Check if the subdirectory has model files
        if any(file.name in ["model.safetensors", "pytorch_model.bin", "config.json", "tokenizer.json"] 
               for file in subdir_files):
            
            logger.info(f"Found nested model structure. Model appears to be in subdirectory: {subdir.name}")
            logger.info("Attempting to use the nested directory as the model path...")
            
            # Return the subdirectory as the corrected path
            return str(subdir)
    
    return None

def main():
    parser = argparse.ArgumentParser(description="Check if a directory has valid HuggingFace model structure")
    parser.add_argument("model_path", help="Path to the model directory")
    parser.add_argument("--fix", action="store_true", help="Attempt to fix common model structure issues")
    
    args = parser.parse_args()
    
    logger.info(f"Checking model directory: {args.model_path}")
    
    if check_model_structure(args.model_path):
        logger.info("Model directory structure verified successfully")
        return 0
    elif args.fix:
        corrected_path = fix_whisper_model(args.model_path)
        if corrected_path:
            logger.info(f"Attempting to use corrected path: {corrected_path}")
            if check_model_structure(corrected_path):
                logger.info(f"Model structure verified with corrected path: {corrected_path}")
                logger.info(f"Use this path in your application: {corrected_path}")
                return 0
    
    logger.error("Model directory structure verification failed")
    return 1

if __name__ == "__main__":
    sys.exit(main())