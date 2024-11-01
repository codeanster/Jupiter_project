"""Astronomical object detection module.

This module provides functionality for detecting and analyzing celestial objects
in Voyager mission imagery using computer vision techniques.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Tuple, Any
import boto3
import cv2
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
import json
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    S3_BUCKET,
    BRIGHTNESS_THRESHOLD,
    MIN_SIZE,
    MAX_SIZE,
    CIRCULARITY_THRESHOLD
)

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('astronomical_detection.log')

# Create formatters and add it to handlers
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(log_format)
file_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class AstroObjectDetector:
    """Detector for astronomical objects in Voyager imagery."""

    def __init__(self, bucket_name: str = S3_BUCKET) -> None:
        """Initialize the detector.
        
        Args:
            bucket_name: Name of the S3 bucket containing the imagery.
        """
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        logger.info(f"Initialized AstroObjectDetector with bucket: {bucket_name}")
        
    def download_image_from_s3(self, key: str) -> Optional[np.ndarray]:
        """Download image from S3 and convert to numpy array.
        
        Args:
            key: S3 key of the image to download.
            
        Returns:
            Numpy array containing the grayscale image data, or None if download fails.
        """
        try:
            logger.debug(f"Attempting to download image: {key}")
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            image_data = obj['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert('L')  # Convert to grayscale
            logger.debug(f"Successfully downloaded and converted image: {key}")
            return np.array(image)
        except Exception as e:
            logger.error(f"Error downloading {key}: {str(e)}")
            return None

    def detect_astronomical_objects(
        self,
        image: np.ndarray,
        brightness_threshold: int = BRIGHTNESS_THRESHOLD,
        min_size: int = MIN_SIZE,
        max_size: int = MAX_SIZE,
        circularity_threshold: float = CIRCULARITY_THRESHOLD
    ) -> Tuple[bool, Dict[str, List[Dict[str, Any]]]]:
        """Detect astronomical objects in an image.
        
        Detection is based on:
        1. Brightness peaks
        2. Circular shape
        3. Size constraints
        
        Args:
            image: Input image as numpy array
            brightness_threshold: Minimum brightness for object detection
            min_size: Minimum object size in pixels
            max_size: Maximum object size in pixels
            circularity_threshold: Minimum circularity (0-1) for object detection
            
        Returns:
            Tuple of (objects_found: bool, detection_info: dict)
        """
        if image is None:
            logger.warning("Received None image for detection")
            return False, {'objects': []}

        logger.debug("Starting object detection with parameters: "
                    f"brightness_threshold={brightness_threshold}, "
                    f"min_size={min_size}, max_size={max_size}, "
                    f"circularity_threshold={circularity_threshold}")

        # Denoise image
        denoised = cv2.GaussianBlur(image, (5, 5), 0)
        
        # Find bright regions
        _, binary = cv2.threshold(denoised, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logger.debug(f"Found {len(contours)} initial contours")
        
        objects_found: List[Dict[str, Any]] = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            # Check both minimum and maximum size constraints
            if area < min_size or area > max_size:
                continue
                
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate peak brightness in object region
            mask = np.zeros_like(image)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            object_region = cv2.bitwise_and(image, image, mask=mask)
            peak_brightness = np.max(object_region)
            
            # If object meets our criteria
            if circularity > circularity_threshold:
                object_info = {
                    'center': (int(x + w/2), int(y + h/2)),
                    'size': int(area),
                    'circularity': float(circularity),
                    'peak_brightness': int(peak_brightness),
                    'bounding_box': {
                        'x': int(x),
                        'y': int(y),
                        'width': int(w),
                        'height': int(h)
                    }
                }
                objects_found.append(object_info)
                logger.debug(f"Detected object {i}: center={object_info['center']}, "
                           f"size={object_info['size']}, "
                           f"circularity={object_info['circularity']:.3f}")
        
        logger.info(f"Detection complete. Found {len(objects_found)} valid objects")
        return len(objects_found) > 0, {'objects': objects_found}

    def process_directory(
        self,
        prefix: str,
        brightness_threshold: int = BRIGHTNESS_THRESHOLD,
        min_size: int = MIN_SIZE,
        max_size: int = MAX_SIZE,
        circularity_threshold: float = CIRCULARITY_THRESHOLD
    ) -> None:
        """Process all images in a directory.
        
        Args:
            prefix: S3 prefix (directory) to process
            brightness_threshold: Minimum brightness for object detection
            min_size: Minimum object size in pixels
            max_size: Maximum object size in pixels
            circularity_threshold: Minimum circularity for object detection
        """
        logger.info(f"Starting directory processing: {prefix}")
        
        # List all images
        paginator = self.s3.get_paginator('list_objects_v2')
        images: List[str] = []
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            if 'Contents' in page:
                images.extend([
                    obj['Key'] for obj in page['Contents']
                    if obj['Key'].lower().endswith(('.png', '.jpg', '.jpeg'))
                ])
        
        if not images:
            logger.warning(f"No images found in directory: {prefix}")
            return
        
        logger.info(f"Found {len(images)} images to process")
        
        # Process images
        results: List[Dict[str, Any]] = []
        for image_key in tqdm(images, desc="Analyzing images"):
            logger.debug(f"Processing image: {image_key}")
            image = self.download_image_from_s3(image_key)
            if image is not None:
                has_objects, info = self.detect_astronomical_objects(
                    image, 
                    brightness_threshold=brightness_threshold,
                    min_size=min_size,
                    max_size=max_size,
                    circularity_threshold=circularity_threshold
                )
                
                if has_objects:
                    result = {
                        'image': image_key,
                        'timestamp': datetime.now().isoformat(),
                        'objects_detected': info
                    }
                    results.append(result)
                    logger.debug(f"Found {len(info['objects'])} objects in {image_key}")
        
        # Save results
        if results:
            # Create a more descriptive filename based on the directory
            dir_name = prefix.rstrip('/').split('/')[-1]
            result_key = f"object_detection/{dir_name}_analysis.json"
            
            analysis_data = {
                'directory': prefix,
                'analysis_parameters': {
                    'brightness_threshold': brightness_threshold,
                    'min_size': min_size,
                    'max_size': max_size,
                    'circularity_threshold': circularity_threshold
                },
                'results': results
            }
            
            try:
                self.s3.put_object(
                    Bucket=self.bucket_name,
                    Key=result_key,
                    Body=json.dumps(analysis_data, indent=2)
                )
                logger.info(f"Results saved to s3://{self.bucket_name}/{result_key}")
                logger.info(f"Found objects in {len(results)} images")
                
                # Log example detections
                if results:
                    logger.info("Example detections:")
                    for result in results[:3]:  # Show first 3 examples
                        logger.info(f"Image: {result['image']}")
                        logger.info(f"Objects detected: {len(result['objects_detected']['objects'])}")
            except Exception as e:
                logger.error(f"Error saving results to S3: {str(e)}")
        else:
            logger.info("No objects detected in any images")

def main() -> None:
    """Main entry point for running the detector on all directories."""
    logger.info("Starting astronomical object detection")
    detector = AstroObjectDetector()
    
    # List all target directories
    logger.info("Listing directories in sorted_png/...")
    paginator = detector.s3.get_paginator('list_objects_v2')
    directories: set = set()
    
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix="sorted_png/", Delimiter='/'):
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                directories.add(prefix['Prefix'])
    
    logger.info(f"Found {len(directories)} directories to process")
    
    # Process each directory
    for directory in sorted(directories):
        detector.process_directory(
            directory,
            brightness_threshold=BRIGHTNESS_THRESHOLD,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            circularity_threshold=CIRCULARITY_THRESHOLD
        )
    
    logger.info("Astronomical object detection completed")

if __name__ == "__main__":
    main()
