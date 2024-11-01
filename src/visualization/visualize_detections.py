"""Visualization module for astronomical object detections.

This module provides functionality for visualizing detected celestial objects
in Voyager mission imagery, including bounding boxes, object properties,
and example generation.
"""

import os
import sys
import logging
import boto3
import json
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    S3_BUCKET,
    BRIGHTNESS_THRESHOLD,
    MIN_SIZE,
    CIRCULARITY_THRESHOLD,
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
class VisualizationStyle:
    """Configuration for visualization appearance."""
    
    box_color: Tuple[int, int, int] = COLORS['BOX_COLOR']
    box_thickness: int = VIS_STYLES['BOX_THICKNESS']
    center_color: Tuple[int, int, int] = COLORS['CENTER_COLOR']
    center_radius: int = VIS_STYLES['CENTER_RADIUS']
    text_color: Tuple[int, int, int] = COLORS['TEXT_COLOR']
    text_scale: float = VIS_STYLES['TEXT_SCALE']
    text_thickness: int = VIS_STYLES['TEXT_THICKNESS']
    font: int = cv2.FONT_HERSHEY_SIMPLEX

class DetectionVisualizer:
    """Visualizer for astronomical object detections."""

    def __init__(
        self,
        bucket_name: str = S3_BUCKET,
        output_dir: str = "detection_examples",
        style: Optional[VisualizationStyle] = None
    ) -> None:
        """Initialize the visualizer.
        
        Args:
            bucket_name: Name of the S3 bucket containing the imagery
            output_dir: Directory to store visualization examples
            style: Optional custom visualization style
        """
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        self.output_dir = output_dir
        self.style = style or VisualizationStyle()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized DetectionVisualizer with output directory: {output_dir}")
        
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
            # Get object properties
            center = tuple(obj['center'])
            size = obj['size']
            circularity = obj['circularity']
            brightness = obj['peak_brightness']
            box = obj['bounding_box']
            
            # Draw bounding box
            x, y = box['x'], box['y']
            w, h = box['width'], box['height']
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                self.style.box_color,
                self.style.box_thickness
            )
            
            # Draw center point
            cv2.circle(
                image,
                center,
                self.style.center_radius,
                self.style.center_color,
                -1
            )
            
            # Add object information
            text = f"#{i} Size:{size} Circ:{circularity:.2f} Bright:{brightness}"
            cv2.putText(
                image,
                text,
                (x, y-5),
                self.style.font,
                self.style.text_scale,
                self.style.text_color,
                self.style.text_thickness
            )
            
        return image

    def process_analysis(
        self,
        analysis_key: str = 'object_detection/CALLISTO_analysis.json',
        num_examples: int = 5
    ) -> None:
        """Process detection analysis and save example visualizations.
        
        Args:
            analysis_key: S3 key for the analysis JSON file
            num_examples: Number of example visualizations to generate
        """
        logger.info("\nProcessing detection analysis...")
        
        try:
            # Load analysis results
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key=analysis_key
            )
            analysis = json.loads(response['Body'].read().decode('utf-8'))
            
            # Print analysis parameters
            params = analysis['analysis_parameters']
            logger.info("\nDetection Parameters:")
            logger.info(f"Brightness threshold: {params['brightness_threshold']}")
            logger.info(f"Minimum size: {params['min_size']}")
            logger.info(f"Circularity threshold: {params['circularity_threshold']}")
            
            # Categorize detections
            single_detections = []
            multiple_detections = []
            
            for result in analysis['results']:
                num_objects = len(result['objects_detected']['objects'])
                if num_objects == 1:
                    single_detections.append(result)
                elif num_objects > 1:
                    multiple_detections.append(result)
            
            # Sort multiple detections by object count
            multiple_detections.sort(
                key=lambda x: len(x['objects_detected']['objects']),
                reverse=True
            )
            
            logger.info(f"\nTotal detections: {len(analysis['results'])}")
            logger.info(f"Single object detections: {len(single_detections)}")
            logger.info(f"Multiple object detections: {len(multiple_detections)}")
            
            # Process examples
            logger.info("\nGenerating example visualizations...")
            logger.info("=" * 50)
            
            # Process single detections
            for i, detection in enumerate(single_detections[:num_examples]):
                image = self.download_image(detection['image'])
                if image is not None:
                    image = self.draw_detections(
                        image,
                        detection['objects_detected']['objects']
                    )
                    
                    filename = os.path.join(self.output_dir, f"single_{i+1}.png")
                    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    logger.info(f"Saved {filename} - 1 object")
            
            # Process multiple detections
            for i, detection in enumerate(multiple_detections[:num_examples]):
                image = self.download_image(detection['image'])
                if image is not None:
                    image = self.draw_detections(
                        image,
                        detection['objects_detected']['objects']
                    )
                    
                    num_objects = len(detection['objects_detected']['objects'])
                    filename = os.path.join(self.output_dir, f"multiple_{i+1}.png")
                    cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    logger.info(f"Saved {filename} - {num_objects} objects")
            
            # Upload examples to S3
            logger.info("\nUploading examples to S3...")
            for filename in os.listdir(self.output_dir):
                if filename.endswith('.png'):
                    filepath = os.path.join(self.output_dir, filename)
                    s3_key = f"detection_examples/{filename}"
                    self.s3.upload_file(filepath, self.bucket_name, s3_key)
                    logger.info(f"Uploaded {filename} to s3://{self.bucket_name}/{s3_key}")
            
        except Exception as e:
            logger.error(f"Error processing analysis: {str(e)}")

def main() -> None:
    """Main entry point for detection visualization."""
    # Example of custom visualization style
    style = VisualizationStyle(
        box_color=COLORS['BOX_COLOR'],
        box_thickness=VIS_STYLES['BOX_THICKNESS'],
        center_color=COLORS['CENTER_COLOR'],
        text_color=COLORS['TEXT_COLOR']
    )
    
    visualizer = DetectionVisualizer(style=style)
    visualizer.process_analysis()
    logger.info(f"\nDone! Check the {visualizer.output_dir} directory for visualizations.")

if __name__ == "__main__":
    main()
