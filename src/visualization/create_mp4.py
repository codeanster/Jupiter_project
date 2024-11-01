"""Video generation module for Voyager mission imagery."""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import re
import multiprocessing as mp
from io import BytesIO

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, parent_dir)

import boto3
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from tqdm import tqdm

from src.config import (
    S3_BUCKET,
    PARALLEL_WORKERS,
    PNG_PREFIX,
    FONT_PATH,
    FONT_SIZE,
    VIDEO_FPS,
    VIDEO_BITRATE
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('visualization.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

def process_frame(args: Tuple[str, str, ImageFont.FreeTypeFont, pd.DataFrame]) -> Optional[np.ndarray]:
    """Process a single frame from S3.
    
    Args:
        args: Tuple of (bucket_name, key, font, metadata)
        
    Returns:
        Processed frame as numpy array, or None if processing fails
    """
    bucket_name, key, font, metadata = args
    try:
        # Create new S3 client for this process
        s3 = boto3.client('s3')
        
        # Download image from S3
        response = s3.get_object(Bucket=bucket_name, Key=key)
        image_data = response['Body'].read()
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        # Add metadata overlay
        draw = ImageDraw.Draw(image)
        
        # Extract date from filename
        date_match = re.search(r'\d{8}', os.path.basename(key))
        if date_match:
            date_str = datetime.strptime(date_match.group(), '%Y%m%d').strftime('%Y-%m-%d')
            
            # Get metadata
            filename = os.path.basename(key)
            meta_row = metadata[metadata['File Path'].str.contains(filename)]
            
            if not meta_row.empty:
                phase = meta_row['Mission Phase Name'].iloc[0]
                target = meta_row['Target Name'].iloc[0]
                
                # Create overlay text
                overlay_text = f"Date: {date_str} | Phase: {phase} | Target: {target}"
                
                # Calculate text size
                bbox = draw.textbbox((0, 0), overlay_text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                
                # Add semi-transparent background
                margin = 10
                background_box = [
                    0,
                    0,
                    image.width,
                    text_height + 2 * margin
                ]
                overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.rectangle(background_box, fill=(0, 0, 0, 128))
                image = Image.alpha_composite(image.convert('RGBA'), overlay)
                image = image.convert('RGB')
                
                # Add text
                draw = ImageDraw.Draw(image)
                text_position = (margin, margin)
                draw.text(text_position, overlay_text, font=font, fill=(255, 255, 255))
        
        # Convert to numpy array for OpenCV
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return frame
        
    except Exception as e:
        logger.error(f"Error processing frame {key}: {str(e)}")
        return None

class VideoGenerator:
    """Main class for generating timelapse videos."""
    
    def __init__(self):
        """Initialize the video generator."""
        # Initialize font
        try:
            self.font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
        except Exception as e:
            logger.warning(f"Could not load font: {str(e)}. Using default.")
            self.font = ImageFont.load_default()
        
        # Load metadata
        try:
            self.metadata = pd.read_csv('meta_file.csv')
        except Exception as e:
            logger.error(f"Error loading metadata: {str(e)}")
            raise
    
    def get_image_files(self, prefix: str, test_mode: bool = False) -> List[str]:
        """Get list of image files from S3."""
        try:
            s3 = boto3.client('s3')
            paginator = s3.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=S3_BUCKET,
                Prefix=prefix
            )
            
            image_files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        if obj['Key'].lower().endswith('.png'):
                            image_files.append(obj['Key'])
            
            # Sort by date in filename
            image_files.sort(key=lambda x: re.findall(r'\d{8}', x)[0])
            
            return image_files[:100] if test_mode else image_files
            
        except Exception as e:
            logger.error(f"Error listing S3 files: {str(e)}")
            return []
    
    def create_video(
        self,
        target: str,
        output_path: str,
        num_cores: Optional[int] = None,
        test_mode: bool = False
    ) -> bool:
        """Create video for a target."""
        try:
            # Get image files
            image_files = self.get_image_files(target, test_mode)
            if not image_files:
                logger.error(f"No images found for target {target}")
                return False
            
            logger.info(f"Processing {len(image_files)} frames for {target}")
            
            # Process first frame to get dimensions
            first_frame = process_frame((
                S3_BUCKET,
                image_files[0],
                self.font,
                self.metadata
            ))
            
            if first_frame is None:
                logger.error("Failed to process first frame")
                return False
            
            height, width = first_frame.shape[:2]
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                VIDEO_FPS,
                (width, height)
            )
            
            # Write first frame
            writer.write(first_frame)
            
            # Process remaining frames in parallel
            num_cores = num_cores or PARALLEL_WORKERS
            
            with mp.Pool(processes=num_cores) as pool:
                args = [
                    (S3_BUCKET, key, self.font, self.metadata)
                    for key in image_files[1:]
                ]
                
                for frame in tqdm(
                    pool.imap(process_frame, args),
                    total=len(args),
                    desc=f"Processing {target}"
                ):
                    if frame is not None:
                        writer.write(frame)
            
            writer.release()
            logger.info(f"Successfully created video: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating video: {str(e)}")
            return False

def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python create_mp4.py <target> <output_path> [num_cores] [--test]")
        sys.exit(1)
    
    target = sys.argv[1]
    output_path = sys.argv[2]
    num_cores = int(sys.argv[3]) if len(sys.argv) > 3 else None
    test_mode = '--test' in sys.argv
    
    generator = VideoGenerator()
    success = generator.create_video(target, output_path, num_cores, test_mode)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
