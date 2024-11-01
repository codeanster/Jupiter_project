"""Tests for the Voyager image processing functionality."""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch, mock_open
import pvl
from datetime import datetime
from PIL import Image
from io import BytesIO, StringIO

from src.processing.process_images import VoyagerImageProcessor, process_image_pair
from src.processing.create_meta import create_meta_file, process_file, list_s3_files

# ============= Test Data =============

@pytest.fixture
def valid_label():
    """Sample valid PVL label data."""
    return """PDS_VERSION_ID                  = PDS3
RECORD_TYPE                     = FIXED_LENGTH
RECORD_BYTES                    = 800
FILE_RECORDS                    = 802
^VICAR_HEADER                   = ("TEST.IMG", 1)
^IMAGE                          = ("TEST.IMG", 3)

DATA_SET_ID                     = "VG2-N-ISS-2/3/4/6-PROCESSED-V1.0"
PRODUCT_ID                      = "TEST.IMG"
PRODUCT_CREATION_TIME           = 2013-10-23T16:00:00
PRODUCT_TYPE                    = CLEANED_IMAGE

INSTRUMENT_HOST_NAME            = "VOYAGER 2"
INSTRUMENT_HOST_ID              = VG2
INSTRUMENT_NAME                 = "IMAGING SCIENCE SUBSYSTEM - NARROW ANGLE"
INSTRUMENT_ID                   = "ISSN"
MISSION_PHASE_NAME              = "JUPITER ENCOUNTER"
TARGET_NAME                     = "EUROPA"
IMAGE_TIME                      = 1989-06-05T08:54:48.00

OBJECT                          = IMAGE
  LINES                         = 100
  LINE_SAMPLES                  = 100
  SAMPLE_TYPE                   = LSB_INTEGER
  SAMPLE_BITS                   = 16
  SAMPLE_DISPLAY_DIRECTION      = RIGHT
  LINE_DISPLAY_DIRECTION        = DOWN
  REFLECTANCE_SCALING_FACTOR    = 0.01
END_OBJECT                      = IMAGE

END"""

@pytest.fixture
def invalid_label():
    """Sample invalid PVL label data."""
    return """PDS_VERSION_ID                  = PDS3
RECORD_TYPE                     = FIXED_LENGTH
RECORD_BYTES                    = 800
FILE_RECORDS                    = 802
^IMAGE                          = ("TEST.IMG", 3)

INSTRUMENT_HOST_NAME            = "VOYAGER 2"
INSTRUMENT_HOST_ID              = VG2
MISSION_PHASE_NAME              = "JUPITER ENCOUNTER"
TARGET_NAME                     = "EUROPA"
IMAGE_TIME                      = 1989-06-05T08:54:48.00

OBJECT                          = IMAGE
  LINES                         = 100
  LINE_SAMPLES                  = 100
  SAMPLE_TYPE                   = INVALID_TYPE
  SAMPLE_BITS                   = 16
  SAMPLE_DISPLAY_DIRECTION      = RIGHT
  LINE_DISPLAY_DIRECTION        = DOWN
  REFLECTANCE_SCALING_FACTOR    = 0.01
END_OBJECT                      = IMAGE

END"""

@pytest.fixture
def minimal_label():
    """Minimal PVL label data for testing error cases."""
    return """PDS_VERSION_ID                  = PDS3
RECORD_TYPE                     = FIXED_LENGTH
RECORD_BYTES                    = 800
FILE_RECORDS                    = 802
END"""

# ============= Fixtures =============

@pytest.fixture
def mock_s3():
    """Mock S3 client."""
    with patch('boto3.client') as mock_client:
        mock_client.return_value.head_object.return_value = {'ContentLength': 1000}
        yield mock_client.return_value

@pytest.fixture
def processor(mock_s3):
    """VoyagerImageProcessor instance."""
    return VoyagerImageProcessor()

@pytest.fixture
def sample_image():
    """Sample image data."""
    return np.random.randint(0, 32767, (100, 100), dtype=np.int16)

@pytest.fixture
def mock_filesystem():
    """Mock filesystem operations."""
    with patch('os.path.exists', return_value=True), \
         patch('os.makedirs', return_value=None), \
         patch('os.remove', return_value=None):
        yield

# ============= Image Processing Tests =============

def test_list_s3_files(processor, mock_s3):
    """Test listing and matching of IMG/LBL pairs in S3."""
    mock_paginator = Mock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {'Key': 'raw/image1.IMG'},
            {'Key': 'raw/image1.LBL'},
            {'Key': 'raw/image2.IMG'},
            {'Key': 'raw/image2.LBL'},
            {'Key': 'raw/unpaired.IMG'}
        ]
    }]
    
    file_pairs = processor.list_s3_files()
    assert len(file_pairs) == 2
    assert any(pair[0] == 'image1' for pair in file_pairs)
    assert any(pair[0] == 'image2' for pair in file_pairs)

def test_process_image_pair(mock_s3, mock_filesystem, sample_image, valid_label):
    """Test successful processing of an image pair."""
    def mock_download(bucket, key, filename):
        if key.endswith('.LBL'):
            with open(filename, 'w') as f:
                f.write(valid_label)
        elif key.endswith('.IMG'):
            sample_image.tofile(filename)
    
    mock_s3.download_file.side_effect = mock_download
    
    result = process_image_pair((
        'test-bucket',
        'test_image',
        'raw/test_image.IMG',
        'raw/test_image.LBL'
    ))
    
    assert result is None
    assert mock_s3.upload_file.called

def test_error_handling_invalid_sample_type(mock_s3, mock_filesystem, invalid_label):
    """Test handling of invalid sample type in label."""
    def mock_download(bucket, key, filename):
        if key.endswith('.LBL'):
            with open(filename, 'w') as f:
                f.write(invalid_label)
    
    mock_s3.download_file.side_effect = mock_download
    
    result = process_image_pair((
        'test-bucket',
        'invalid_image',
        'raw/invalid_image.IMG',
        'raw/invalid_image.LBL'
    ))
    
    assert result is not None
    assert "INVALID_TYPE" in str(result)

def test_image_normalization():
    """Test image normalization to 8-bit."""
    test_image = np.array([[0, 32767], [16384, 24576]], dtype=np.int16)
    
    img_min, img_max = test_image.min(), test_image.max()
    normalized = ((test_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    assert normalized.dtype == np.uint8
    assert normalized.min() == 0
    assert normalized.max() == 255
    assert normalized[0][0] == 0
    assert normalized[0][1] == 255

# ============= Metadata Creation Tests =============

def test_meta_list_s3_files(mock_s3):
    """Test listing files for metadata creation."""
    mock_paginator = Mock()
    mock_s3.get_paginator.return_value = mock_paginator
    mock_paginator.paginate.return_value = [{
        'Contents': [
            {'Key': 'raw/image1.IMG'},
            {'Key': 'raw/image1.LBL'},
            {'Key': 'raw/image2.IMG'},
            {'Key': 'raw/image2.LBL'}
        ]
    }]
    
    with patch('src.processing.create_meta.s3', mock_s3):
        file_pairs = list_s3_files()
        assert len(file_pairs) == 2

def test_process_meta_file(mock_s3, mock_filesystem, valid_label):
    """Test metadata extraction from a valid label."""
    def mock_download(bucket, key, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(valid_label)
    
    mock_s3.download_file.side_effect = mock_download
    
    with patch('src.processing.create_meta.s3', mock_s3):
        result = process_file(('test_image', 'raw/test_image.IMG', 'raw/test_image.LBL'))
    
    assert result is not None
    assert result['Mission Phase Name'] == 'JUPITER ENCOUNTER'
    assert result['Target Name'] == 'EUROPA'

def test_create_meta_file(mock_s3, mock_filesystem):
    """Test metadata file creation."""
    with patch('src.processing.create_meta.list_s3_files') as mock_list:
        mock_list.return_value = [
            ('test_image1', 'raw/test_image1.IMG', 'raw/test_image1.LBL'),
            ('test_image2', 'raw/test_image2.IMG', 'raw/test_image2.LBL')
        ]
        
        with patch('builtins.open', mock_open()):
            create_meta_file()

def test_meta_error_handling(mock_s3, mock_filesystem):
    """Test metadata error handling."""
    with patch('src.processing.create_meta.download_from_s3', 
              return_value=(False, "Download failed")):
        result = process_file(('error_image', 'raw/error_image.IMG', 'raw/error_image.LBL'))
    
    assert result is None

def test_invalid_meta_label_data(mock_s3, mock_filesystem, minimal_label):
    """Test handling of invalid metadata."""
    def mock_download(bucket, key, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(minimal_label)
    
    mock_s3.download_file.side_effect = mock_download
    
    with patch('src.processing.create_meta.s3', mock_s3):
        result = process_file(('invalid_image', 'raw/invalid_image.IMG', 'raw/invalid_image.LBL'))
    
    assert result is not None
    assert result['Mission Phase Name'] == 'Unknown'
    assert result['Target Name'] == 'Unknown'
