"""Tests for the video generation functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from io import BytesIO
import pandas as pd
from src.visualization.create_mp4 import VideoGenerator

@pytest.fixture
def mock_s3_client():
    with patch('boto3.client') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def video_generator(mock_s3_client):
    with patch('pandas.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame({
            'File Path': ['test_20240101.png'],
            'Mission Phase Name': ['Test Phase'],
            'Target Name': ['Test Target']
        })
        return VideoGenerator()

def create_test_image(width=800, height=600):
    """Create a test image with specific dimensions."""
    return Image.new('RGB', (width, height), color='black')

def test_process_image_from_s3(video_generator, mock_s3_client):
    """Test processing a single image from S3 with metadata overlay."""
    # Create a test image
    test_image = create_test_image()
    img_byte_arr = BytesIO()
    test_image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Mock S3 response
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=lambda: img_byte_arr)
    }

    # Process test image
    result = video_generator.process_image_from_s3((
        'test-bucket',
        'test_20240101.png',
        video_generator.font
    ))

    # Verify the result
    assert isinstance(result, np.ndarray)
    assert result.shape == (600, 800, 3)
    assert result.dtype == np.uint8

def test_get_image_files(video_generator, mock_s3_client):
    """Test getting and sorting image files from S3."""
    # Mock S3 paginator
    mock_paginator = Mock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    # Mock paginator pages
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {'Key': 'test_20240102.png'},
            {'Key': 'test_20240101.png'},
            {'Key': 'test_20240103.png'},
            {'Key': 'invalid.txt'}
        ]
    }]

    # Get image files
    files = video_generator.get_image_files('test', test_mode=False)

    # Verify files are sorted by date
    assert len(files) == 3
    assert files[0].endswith('20240101.png')
    assert files[1].endswith('20240102.png')
    assert files[2].endswith('20240103.png')

def test_error_handling_invalid_image(video_generator, mock_s3_client):
    """Test handling of invalid image data."""
    # Mock S3 response with invalid image data
    mock_s3_client.get_object.return_value = {
        'Body': Mock(read=lambda: b'invalid_image_data')
    }

    # Process invalid image
    result = video_generator.process_image_from_s3((
        'test-bucket',
        'invalid_20240101.png',
        video_generator.font
    ))

    # Verify the result is None for invalid image
    assert result is None

def test_test_mode_limitation(video_generator, mock_s3_client):
    """Test that test mode limits the number of processed files."""
    # Mock S3 paginator
    mock_paginator = Mock()
    mock_s3_client.get_paginator.return_value = mock_paginator

    # Create more than 100 test files
    test_files = [{'Key': f'test_{i:08d}.png'} for i in range(150)]
    mock_paginator.paginate.return_value = [{
        'Contents': test_files
    }]

    # Get image files in test mode
    files = video_generator.get_image_files('test', test_mode=True)

    # Verify only 100 files are returned in test mode
    assert len(files) == 100
