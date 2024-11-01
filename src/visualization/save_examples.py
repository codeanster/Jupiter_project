"""Example generation module for astronomical object detections.

This module provides functionality for categorizing and saving example images
of different types of astronomical object detections, organizing them into
meaningful categories like surface views and distant objects.
"""

import os
import sys
import logging
import boto3
import json
import numpy as np
from PIL import Image
import io
import cv2
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    S3_BUCKET,
    COLORS,
    VIS_STYLES
)

# Configure logging
logging.basicConfig(
    filename='visualization.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

logger = logging.getLogger(__name__)

@dataclass
class CategoryThresholds:
    """Configuration for view categorization thresholds."""
    
    surface_size: int = 1000
    surface_multi_size: int = 500
    surface_multi_count: int = 10
    faraway_max_size: int = 1000
    faraway_circularity: float = 0.75

class ExampleSaver:
    """Processor for categorizing and saving detection examples."""

    def __init__(
        self,
        bucket_name: str = S3_BUCKET,
        output_dir: str = "category_examples",
        thresholds: Optional[CategoryThresholds] = None
    ) -> None:
        """Initialize the example saver.
        
        Args:
            bucket_name: Name of the S3 bucket containing the imagery
            output_dir: Base directory for saving categorized examples
            thresholds: Optional custom categorization thresholds
        """
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.output_dir = output_dir
        self.thresholds = thresholds or CategoryThresholds()
        
        # Create category directories
        for category in ['surface', 'faraway', 'other']:
            os.makedirs(f"{output_dir}/{category}", exist_ok=True)
        
        logger.info(f"Initialized ExampleSaver with output directory: {output_dir}")
        
    def download_image(self, key: str) -> Optional[np.ndarray]:
        """Download image from S3 and convert to numpy array.
        
        Args:
            key: S3 key of the image to download
            
        Returns:
            Optional numpy array containing the image data
        """
        try:
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            image_data = obj['Body'].read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            return np.array(image)
        except Exception as e:
            logger.error(f"Error downloading {key}: {str(e)}")
            return None

    def draw_detections(
        self,
        image: np.ndarray,
        objects: List[Dict[str, Any]]
    ) -> np.ndarray:
        """Draw detection visualizations on an image.
        
        Args:
            image: Input image as numpy array
            objects: List of detected object properties
            
        Returns:
            Numpy array containing the annotated image
        """
        for i, obj in enumerate(objects, 1):
            # Get object info
            center = tuple(obj['center'])
            size = obj['size']
            circularity = obj['circularity']
            box = obj['bounding_box']
            
            # Draw bounding box
            x, y = box['x'], box['y']
            w, h = box['width'], box['height']
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                COLORS['BOX_COLOR'],
                VIS_STYLES['BOX_THICKNESS']
            )
            
            # Draw center point
            cv2.circle(
                image,
                center,
                VIS_STYLES['CENTER_RADIUS'],
                COLORS['CENTER_COLOR'],
                -1
            )
            
            # Add text with object info
            text = f"#{i} Size:{size} Circ:{circularity:.2f}"
            cv2.putText(
                image,
                text,
                (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                VIS_STYLES['TEXT_SCALE'],
                COLORS['TEXT_COLOR'],
                VIS_STYLES['TEXT_THICKNESS']
            )
            
        return image

    def categorize_view(self, result: Dict[str, Any]) -> Optional[str]:
        """Categorize a view based on object characteristics.
        
        Args:
            result: Detection result containing object information
            
        Returns:
            Category string ('surface', 'faraway', or 'other') or None if invalid
        """
        objects = result['objects_detected']['objects']
        
        if not objects:
            return None
        
        # Sort objects by size
        objects = sorted(objects, key=lambda x: x['size'], reverse=True)
        
        largest_obj = objects[0]
        num_objects = len(objects)
        avg_circularity = np.mean([obj['circularity'] for obj in objects])
        
        # Categorize based on thresholds
        if (largest_obj['size'] > self.thresholds.surface_size or 
            (num_objects > self.thresholds.surface_multi_count and 
             largest_obj['size'] > self.thresholds.surface_multi_size)):
            return 'surface'
        elif (largest_obj['size'] < self.thresholds.faraway_max_size and 
              avg_circularity > self.thresholds.faraway_circularity):
            return 'faraway'
        else:
            return 'other'

    def save_examples(
        self,
        analysis_key: str = 'object_detection/CALLISTO_analysis.json',
        num_examples: int = 5
    ) -> None:
        """Save example images from each category.
        
        Args:
            analysis_key: S3 key for the analysis JSON file
            num_examples: Number of examples to save per category
        """
        logger.info("\nProcessing detection analysis for examples...")
        
        try:
            # Load analysis results
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=analysis_key
            )
            analysis = json.loads(response['Body'].read().decode('utf-8'))
            
            # Categorize detections
            categories: Dict[str, List[Dict[str, Any]]] = {
                'surface': [], 'faraway': [], 'other': []
            }
            
            for result in analysis['results']:
                category = self.categorize_view(result)
                if category:
                    categories[category].append(result)
            
            logger.info("\nSaving example images from each category:")
            logger.info("=" * 50)
            
            # Save examples from each category
            for category, results in categories.items():
                # Sort based on category-specific criteria
                if category == 'surface':
                    results.sort(
                        key=lambda x: max(obj['size'] for obj in x['objects_detected']['objects']),
                        reverse=True
                    )
                elif category == 'faraway':
                    results.sort(
                        key=lambda x: len(x['objects_detected']['objects']),
                        reverse=True
                    )
                else:
                    results.sort(
                        key=lambda x: np.mean([obj['circularity'] for obj in x['objects_detected']['objects']]),
                        reverse=True
                    )
                
                logger.info(f"\n{category.upper()} Examples:")
                for i, result in enumerate(results[:num_examples]):
                    image = self.download_image(result['image'])
                    if image is not None:
                        # Draw detections
                        image = self.draw_detections(
                            image,
                            result['objects_detected']['objects']
                        )
                        
                        # Save image
                        filename = f"{self.output_dir}/{category}/example_{i+1}.png"
                        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                        
                        # Upload to S3
                        s3_key = f"analysis/categories/{category}/example_{i+1}.png"
                        try:
                            self.s3.upload_file(filename, self.bucket_name, s3_key)
                        except Exception as e:
                            logger.error(f"Error uploading to S3: {str(e)}")
                            continue
                        
                        # Log statistics
                        objects = result['objects_detected']['objects']
                        num_objects = len(objects)
                        largest_size = max(obj['size'] for obj in objects)
                        avg_circ = np.mean([obj['circularity'] for obj in objects])
                        
                        logger.info(
                            f"Saved example {i+1}: {num_objects} objects, "
                            f"largest size: {largest_size}, "
                            f"avg circularity: {avg_circ:.3f}"
                        )
            
        except Exception as e:
            logger.error(f"Error processing analysis: {str(e)}")

def main() -> None:
    """Main entry point for example generation."""
    # Example of custom thresholds
    thresholds = CategoryThresholds(
        surface_size=1000,
        surface_multi_size=500,
        surface_multi_count=10,
        faraway_max_size=1000,
        faraway_circularity=0.75
    )
    
    saver = ExampleSaver(thresholds=thresholds)
    saver.save_examples()
    logger.info(f"\nDone! Check the {saver.output_dir} directory for examples.")

if __name__ == "__main__":
    main()
