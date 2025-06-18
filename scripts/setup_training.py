#!/usr/bin/env python3
"""
Setup script for training the WordPiece tokenizer on CodeSearchNet data.
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import logging
from tqdm import tqdm
from datasets import load_dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingSetup:
    def __init__(self, base_dir: str = "data"):
        self.base_dir = Path(base_dir)
        self.code_dir = self.base_dir / "CodeSearchNet"
        self.processed_dir = self.base_dir / "processed"
        self.vocab_dir = self.base_dir / "vocab"
        
        # Create necessary directories
        for dir_path in [self.base_dir, self.processed_dir, self.vocab_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_data(self) -> bool:
        """Download the CodeSearchNet data using Hugging Face datasets."""
        required_files = ['train.txt', 'validation.txt', 'test.txt']
        
        # Check if all required files exist and have content
        files_exist = all((self.processed_dir / f).exists() for f in required_files)
        files_have_content = all((self.processed_dir / f).stat().st_size > 0 for f in required_files) if files_exist else False
        
        if files_exist and files_have_content:
            logger.info("All processed files exist and have content")
            return True
            
        # Clean up any existing files that are missing or empty
        logger.info("Cleaning up existing files...")
        for file_name in required_files:
            file_path = self.processed_dir / file_name
            if file_path.exists():
                if file_path.stat().st_size == 0:
                    logger.info(f"Removing empty file: {file_name}")
                    file_path.unlink()
                else:
                    logger.info(f"Removing existing file: {file_name}")
                    file_path.unlink()
            
        logger.info("Downloading CodeSearchNet data...")
        try:
            # Load the dataset with trust_remote_code=True
            dataset = load_dataset("code_search_net", "python", trust_remote_code=True)
            
            # Print dataset structure for debugging
            logger.info("Dataset structure:")
            logger.info(f"Features: {dataset['train'].features}")
            logger.info(f"First example: {dataset['train'][0]}")
            
            # Process and save the data
            logger.info("Processing and saving data...")
            
            # Process training data
            train_file = self.processed_dir / "train.txt"
            with open(train_file, 'w', encoding='utf-8') as f:
                for item in tqdm(dataset['train'], desc="Processing training data"):
                    # The dataset uses 'whole_func_string' for code and 'func_documentation_string' for docstrings
                    code = item['whole_func_string'].strip()
                    docstring = item['func_documentation_string'].strip()
                    if code:
                        f.write(code + '\n')
                    if docstring:
                        f.write(docstring + '\n')
            
            # Process validation data
            val_file = self.processed_dir / "validation.txt"
            with open(val_file, 'w', encoding='utf-8') as f:
                for item in tqdm(dataset['validation'], desc="Processing validation data"):
                    code = item['whole_func_string'].strip()
                    docstring = item['func_documentation_string'].strip()
                    if code:
                        f.write(code + '\n')
                    if docstring:
                        f.write(docstring + '\n')
            
            # Process test data
            test_file = self.processed_dir / "test.txt"
            with open(test_file, 'w', encoding='utf-8') as f:
                for item in tqdm(dataset['test'], desc="Processing test data"):
                    code = item['whole_func_string'].strip()
                    docstring = item['func_documentation_string'].strip()
                    if code:
                        f.write(code + '\n')
                    if docstring:
                        f.write(docstring + '\n')
            
            # Verify that files were created and have content
            for file_path in [train_file, val_file, test_file]:
                if not file_path.exists() or file_path.stat().st_size == 0:
                    raise Exception(f"Failed to create or write to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to download and process data: {e}")
            # Print more detailed error information
            import traceback
            logger.error(f"Error details:\n{traceback.format_exc()}")
            # Clean up any partially created files
            for file_name in required_files:
                file_path = self.processed_dir / file_name
                if file_path.exists():
                    file_path.unlink()
            return False
    
    def verify_data(self) -> bool:
        """Verify that the processed files exist and have content."""
        required_files = ['train.txt', 'validation.txt', 'test.txt']
        
        # Check if all files exist
        if not all((self.processed_dir / f).exists() for f in required_files):
            logger.error("Some required files are missing")
            return False
            
        # Check if files have content
        for file_name in required_files:
            file_path = self.processed_dir / file_name
            if file_path.stat().st_size == 0:
                logger.error(f"File {file_name} is empty")
                # Remove empty file so it can be recreated
                file_path.unlink()
                return False
                
        # Print file sizes for verification
        for file_name in required_files:
            file_path = self.processed_dir / file_name
            size_mb = file_path.stat().st_size / (1024 * 1024)
            logger.info(f"{file_name}: {size_mb:.2f} MB")
            
        return True
    
    def run(self) -> bool:
        """Run the complete setup process."""
        steps = [
            (self.download_data, "Downloading and processing data"),
            (self.verify_data, "Verifying data")
        ]
        
        for step_func, step_name in steps:
            logger.info(f"Starting: {step_name}")
            if not step_func():
                logger.error(f"Failed: {step_name}")
                return False
            logger.info(f"Completed: {step_name}")
        
        logger.info("Setup completed successfully!")
        return True

if __name__ == "__main__":
    setup = TrainingSetup()
    if setup.run():
        logger.info("""
        Setup completed! Next steps:
        1. Review the processed data in data/processed/
        2. Start training the tokenizer:
           python scripts/train_tokenizer.py
        3. Monitor training progress in data/vocab/
        """)
    else:
        logger.error("Setup failed. Please check the logs above for details.")