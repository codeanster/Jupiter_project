"""Voyager mission image processing module.

This module handles the processing of raw Voyager mission data (IMG/LBL pairs),
converting them to normalized PNG images while preserving scientific accuracy.
"""

import os
import sys
import pvl
import numpy as np
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from typing import Dict, List, Tuple, Optional, Any

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    S3_BUCKET,
    INPUT_PREFIX,
    PNG_PREFIX,
    TEMP_DIR,
    PARALLEL_WORKERS
)

# Initialize logging
logging.basicConfig(
    filename='process_images.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

#has to be in it's function because of multiprocessing
def process_image_pair(args: Tuple[str, str, str, str]) -> Optional[str]:
    """Process a single image pair: download, process, upload, and clean up.
    
    Args:
        args: Tuple of (bucket_name, base_name, img_s3_key, lbl_s3_key)
        
    Returns:
        Optional status message string
    """
    bucket_name, base_name, img_s3_key, lbl_s3_key = args
    
    # Create S3 client inside the worker process
    s3 = boto3.client('s3')
    
    try:
        # Define local paths
        local_lbl = os.path.join(TEMP_DIR, os.path.basename(lbl_s3_key))
        local_img = os.path.join(TEMP_DIR, os.path.basename(img_s3_key))

        # Download files
        for s3_key, local_path in [(lbl_s3_key, local_lbl), (img_s3_key, local_img)]:
            try:
                s3.download_file(bucket_name, s3_key, local_path)
            except Exception as e:
                logging.error(f"Failed to download {s3_key}: {str(e)}")
                return f"Failed to download {s3_key}: {str(e)}"

        # Load and parse LBL file
        with open(local_lbl, 'r') as lbl_file:
            lbl = pvl.load(lbl_file)

        # Extract image parameters
        image_info = lbl['IMAGE']
        image_lines = int(image_info['LINES'])
        line_samples = int(image_info['LINE_SAMPLES'])
        sample_type = image_info['SAMPLE_TYPE']
        sample_bits = int(image_info['SAMPLE_BITS'])
        scaling_factor = float(image_info['REFLECTANCE_SCALING_FACTOR'])

        # Get image timestamp
        image_time = lbl.get('IMAGE_TIME')
        date_str = image_time.strftime('%Y%m%d_%H%M%S') if image_time else 'unknown_date'

        # Determine numpy dtype
        if sample_type.upper() == 'LSB_INTEGER' and sample_bits == 16:
            dtype = '<i2'  # 16-bit little-endian integer
        elif sample_type.upper() == 'UNSIGNED_INTEGER' and sample_bits == 8:
            dtype = 'uint8'  # 8-bit unsigned integer
        else:
            raise ValueError(f"Unsupported SAMPLE_TYPE or SAMPLE_BITS: {sample_type}, {sample_bits}")

        # Process image data
        img_data = np.memmap(local_img, dtype=dtype, mode='r', 
                           shape=(image_lines, line_samples))
        image = img_data * scaling_factor

        # Normalize to 8-bit
        img_min, img_max = image.min(), image.max()
        image_8bit = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        img_pil = Image.fromarray(image_8bit)

        # Save and upload PNG
        temp_png_path = os.path.join(TEMP_DIR, f"{base_name}_{date_str}.png")
        img_pil.save(temp_png_path, 'PNG')

        png_s3_key = f"{PNG_PREFIX}{base_name}_{date_str}.png"
        try:
            s3.upload_file(temp_png_path, bucket_name, png_s3_key)
        except Exception as e:
            logging.error(f"Failed to upload {png_s3_key}: {str(e)}")
            return f"Failed to upload {png_s3_key}: {str(e)}"

        # Cleanup
        for file_path in [local_lbl, local_img, temp_png_path]:
            if os.path.exists(file_path):
                os.remove(file_path)

        logging.info(f"Processed and uploaded {base_name} successfully.")
        return None

    except Exception as e:
        logging.error(f"Error processing {base_name}: {e}")
        return f"Error processing {base_name}: {e}"

class VoyagerImageProcessor:
    """Processor for Voyager mission imagery."""

    def __init__(self, bucket_name: str = S3_BUCKET) -> None:
        """Initialize the processor.
        
        Args:
            bucket_name: Name of the S3 bucket containing the imagery.
        """
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        os.makedirs(TEMP_DIR, exist_ok=True)

    def list_s3_files(self) -> List[Tuple[str, str, str]]:
        """List and match IMG and LBL files in the S3 bucket.
        
        Returns:
            List of tuples containing (base_name, img_s3_key, lbl_s3_key)
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=self.bucket_name, Prefix=INPUT_PREFIX)

        img_files: Dict[str, str] = {}
        lbl_files: Dict[str, str] = {}

        for page in pages:
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith(('.img', '.IMG')):
                    base_name = os.path.splitext(os.path.basename(key))[0]
                    img_files[base_name] = key
                elif key.endswith(('.lbl', '.LBL')):
                    base_name = os.path.splitext(os.path.basename(key))[0]
                    lbl_files[base_name] = key

        base_names = set(img_files.keys()) & set(lbl_files.keys())
        return [(base, img_files[base], lbl_files[base]) for base in base_names]

    def process_all_images(self) -> None:
        """Process all available Voyager image pairs."""
        base_files = self.list_s3_files()
        if not base_files:
            print("No matching IMG and LBL file pairs found in S3.")
            return

        print(f"Found {len(base_files)} image pairs to process.")

        with tqdm(total=len(base_files), desc="Processing Image Pairs", unit="pair") as pbar:
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                # Create process arguments with bucket name
                process_args = [
                    (self.bucket_name, base, img_key, lbl_key)
                    for base, img_key, lbl_key in base_files
                ]
                
                futures = {
                    executor.submit(process_image_pair, args): args[1]  # args[1] is base_name
                    for args in process_args
                }

                for future in as_completed(futures):
                    result = future.result()
                    if result:  # Only print if there's an error message
                        print(result)
                    pbar.update(1)

        print("Processing completed. Check 'process_images.log' for detailed logs.")

def main() -> None:
    """Main entry point for the Voyager image processor."""
    processor = VoyagerImageProcessor()
    processor.process_all_images()

if __name__ == "__main__":
    main()
