"""Tests for the astronomical object detection functionality."""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
import json
from datetime import datetime
from PIL import Image
import io

from src.detection.detect_objects import AstroObjectDetector

@pytest.fixture
def mock_s3_client():
    with patch('boto3.client') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def detector(mock_s3_client):
    return AstroObjectDetector()

def create_test_image(size=200, shape='circle', brightness=255):
    """Create a test image with a specified bright shape."""
    image = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    
    if shape == 'circle':
        radius = 20
        cv2.circle(image, center, radius, brightness, -1)
    elif shape == 'rectangle':
        x, y = center[0] - 20, center[1] - 20
        cv2.rectangle(image, (x, y), (x + 40, y + 40), brightness, -1)
    elif shape == 'multiple':
        # Add three circles
        cv2.circle(image, (50, 50), 15, brightness, -1)
        cv2.circle(image, (150, 150), 20, brightness, -1)
        cv2.circle(image, (100, 100), 18, brightness, -1)
    elif shape == 'large_circle':
        radius = 50  # Much larger radius
        cv2.circle(image, center, radius, brightness, -1)
    elif shape == 'ellipse':
        # Create an ellipse that's somewhat circular but not perfect
        axes = (25, 20)  # Major and minor axes
        cv2.ellipse(image, center, axes, 0, 0, 360, brightness, -1)
    
    return image

def test_basic_object_detection(detector):
    """Test basic object detection with a circular object."""
    test_image = create_test_image(shape='circle')
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        circularity_threshold=0.7
    )
    
    assert has_objects
    assert len(info['objects']) == 1
    obj = info['objects'][0]
    assert 0.8 < obj['circularity'] <= 1.0  # Circle should have high circularity
    assert obj['peak_brightness'] > 200

def test_multiple_object_detection(detector):
    """Test detection of multiple objects with size ordering and position uniqueness."""
    test_image = create_test_image(shape='multiple')
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        circularity_threshold=0.7
    )
    
    assert has_objects
    assert len(info['objects']) == 3
    
    # Check size ordering with more readable assertion
    sizes = [obj['size'] for obj in info['objects']]
    assert sizes == sorted(sizes, reverse=True), "Objects should be sorted by size in descending order"
    
    # Check that each object has unique coordinates
    positions = [obj['center'] for obj in info['objects']]
    assert len(set(positions)) == 3, "Each object should have unique coordinates"

def test_max_size_constraint(detector):
    """Test that objects exceeding max_size are not detected."""
    # Create an image with a large circle
    test_image = create_test_image(shape='large_circle')
    
    # First verify the object is detected with a high max_size
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        max_size=10000,  # Very high max_size
        circularity_threshold=0.7
    )
    assert has_objects, "Large object should be detected with high max_size"
    assert len(info['objects']) == 1
    
    # Now verify the same object is rejected when max_size is small
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        max_size=100,  # Small max_size
        circularity_threshold=0.7
    )
    assert not has_objects, "Large object should not be detected with small max_size"
    assert len(info['objects']) == 0

def test_circularity_threshold_with_ellipse(detector):
    """Test detection behavior with almost-circular objects (ellipse)."""
    test_image = create_test_image(shape='ellipse')
    
    # Test with lenient circularity threshold (should detect)
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        circularity_threshold=0.7  # More lenient threshold
    )
    assert has_objects, "Ellipse should be detected with lenient threshold"
    assert len(info['objects']) == 1
    obj = info['objects'][0]
    assert 0.7 <= obj['circularity'] < 1.0, "Ellipse should have good but not perfect circularity"
    
    # Test with strict circularity threshold (should not detect)
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        circularity_threshold=0.95  # Very strict threshold
    )
    assert not has_objects, "Ellipse should not be detected with strict threshold"
    assert len(info['objects']) == 0

def test_non_circular_object(detector):
    """Test detection with a non-circular object."""
    test_image = create_test_image(shape='rectangle')
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=20,
        circularity_threshold=0.9  # Higher threshold to reject rectangle
    )
    
    # Rectangle should not be detected with high circularity threshold
    assert not has_objects
    assert len(info['objects']) == 0

def test_size_constraints(detector):
    """Test object size constraints."""
    test_image = create_test_image(shape='circle')
    
    # Test with size too small
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=2000,  # Much larger than actual object
        circularity_threshold=0.7
    )
    assert not has_objects
    
    # Test with appropriate size
    has_objects, info = detector.detect_astronomical_objects(
        test_image,
        brightness_threshold=75,
        min_size=100,
        circularity_threshold=0.7
    )
    assert has_objects

def test_brightness_threshold(detector):
    """Test brightness threshold parameter."""
    # Create a dim circle
    dim_image = create_test_image(shape='circle', brightness=100)
    
    # Test with threshold above object brightness (should not detect)
    has_objects, info = detector.detect_astronomical_objects(
        dim_image,
        brightness_threshold=150,  # Higher than object brightness
        min_size=20,
        max_size=2000,
        circularity_threshold=0.7
    )
    assert not has_objects
    
    # Test with threshold below object brightness (should detect)
    has_objects, info = detector.detect_astronomical_objects(
        dim_image,
        brightness_threshold=50,  # Lower than object brightness
        min_size=20,
        max_size=2000,
        circularity_threshold=0.7
    )
    assert has_objects

def test_s3_download(detector, mock_s3_client):
    """Test S3 image downloading."""
    # Create test image
    test_image = create_test_image()
    img_pil = Image.fromarray(test_image)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Mock S3 response
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=lambda: img_byte_arr)
    }
    
    # Test download
    result = detector.download_image_from_s3('test.png')
    assert isinstance(result, np.ndarray)
    assert result.shape == (200, 200)

def test_s3_download_error(detector, mock_s3_client):
    """Test error handling in S3 download."""
    mock_s3_client.get_object.side_effect = Exception("Download failed")
    result = detector.download_image_from_s3('nonexistent.png')
    assert result is None

def test_process_directory(detector, mock_s3_client):
    """Test directory processing functionality."""
    # Create test image
    test_image = create_test_image()
    img_pil = Image.fromarray(test_image)
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    # Mock S3 responses
    mock_paginator = Mock()
    mock_s3_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {'Key': 'test/image1.png'},
            {'Key': 'test/image2.png'}
        ]
    }]
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=lambda: img_byte_arr)
    }
    
    # Process directory
    with patch('src.detection.detect_objects.tqdm', new=lambda x, **kwargs: x):
        detector.process_directory('test/')
    
    # Verify S3 operations
    assert mock_s3_client.get_object.called
    assert mock_s3_client.put_object.called
    
    # Verify the analysis JSON was uploaded
    call_args = mock_s3_client.put_object.call_args[1]
    assert 'object_detection/test_analysis.json' in call_args['Key']
    
    # Verify results format
    results = json.loads(call_args['Body'])
    assert 'analysis_parameters' in results
    assert 'results' in results
    assert len(results['results']) > 0

def test_empty_directory(detector, mock_s3_client):
    """Test processing an empty directory."""
    mock_paginator = Mock()
    mock_s3_client.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{'Contents': []}]
    
    with patch('src.detection.detect_objects.tqdm', new=lambda x, **kwargs: x):
        detector.process_directory('empty/')
    
    assert not mock_s3_client.put_object.called

def test_invalid_image(detector):
    """Test detection with invalid image input."""
    has_objects, info = detector.detect_astronomical_objects(None)
    assert not has_objects
    assert len(info['objects']) == 0
