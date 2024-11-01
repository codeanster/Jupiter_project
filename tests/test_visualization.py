"""Tests for the visualization functionality."""

import pytest
import numpy as np
import cv2
import json
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io
import concurrent.futures
import subprocess
import logging
from pathlib import Path

from src.visualization.visualize_detections import DetectionVisualizer, VisualizationStyle
from src.visualization.create_all_videos import VideoOrchestrator
from src.visualization.save_examples import ExampleSaver, CategoryThresholds

class MockFuture:
    """Mock Future class for testing concurrent operations."""
    def __init__(self, result_value):
        self._condition = MagicMock()
        self._state = 'FINISHED'
        self._result = result_value
        self._waiters = []

    def result(self):
        return self._result

    def add_done_callback(self, fn):
        fn(self)

@pytest.fixture
def mock_s3():
    """Fixture for mocking S3 client."""
    with patch('boto3.client') as mock_client:
        yield mock_client.return_value

@pytest.fixture
def test_image():
    """Create a test image for visualization."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[40:60, 40:60] = 255  # Add a white square
    return img

class TestDetectionVisualizer:
    """Tests for the DetectionVisualizer class."""

    @pytest.fixture
    def visualizer(self, mock_s3):
        with patch('os.makedirs'):
            return DetectionVisualizer(output_dir="test_output")

    @pytest.fixture
    def test_detection(self):
        """Sample detection data."""
        return {
            'center': (50, 50),
            'size': 100,
            'circularity': 0.95,
            'peak_brightness': 200,
            'bounding_box': {'x': 25, 'y': 25, 'width': 50, 'height': 50}
        }

    def test_init(self, visualizer):
        """Test initialization."""
        assert visualizer.output_dir == "test_output"
        assert isinstance(visualizer.style, VisualizationStyle)

    def test_draw_detections(self, visualizer, test_image, test_detection):
        """Test drawing detections on image."""
        result = visualizer.draw_detections(test_image.copy(), [test_detection])
        assert np.any(result != test_image)  # Image should be modified
        assert result.shape == test_image.shape

    def test_download_image(self, visualizer, mock_s3, test_image):
        """Test S3 image downloading."""
        img_bytes = io.BytesIO()
        Image.fromarray(test_image).save(img_bytes, format='PNG')
        mock_s3.get_object.return_value = {'Body': Mock(read=lambda: img_bytes.getvalue())}
        
        result = visualizer.download_image('test.png')
        assert isinstance(result, np.ndarray)
        assert result.shape == test_image.shape

class TestVideoOrchestrator:
    """Tests for the VideoOrchestrator class."""

    @pytest.fixture
    def orchestrator(self):
        with patch('os.makedirs'):
            return VideoOrchestrator(max_workers=2, output_dir="test_videos")

    def test_create_video(self, orchestrator):
        """Test single video creation."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            assert orchestrator.create_video("JUPITER")
            assert "JUPITER" in orchestrator.completed

    def test_process_all_targets(self, orchestrator):
        """Test processing multiple targets."""
        targets = ["JUPITER", "EUROPA", "IO"]
        
        # Create a real create_video method that updates sets properly
        def mock_create_video(target):
            if target == "EUROPA":
                orchestrator.failed.add(target)
                return False
            orchestrator.completed.add(target)
            return True
        
        # Patch the create_video method
        with patch.object(orchestrator, 'create_video', side_effect=mock_create_video):
            # Create the executor context
            with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                # Setup the mock executor to actually call our mocked create_video
                mock_executor.return_value.__enter__.return_value.submit.side_effect = \
                    lambda fn, target: MockFuture(fn(target))
                
                # Process all targets
                orchestrator.process_all_targets(targets)
                
                # Verify results
                assert len(orchestrator.completed) == 2
                assert len(orchestrator.failed) == 1
                assert "JUPITER" in orchestrator.completed
                assert "IO" in orchestrator.completed
                assert "EUROPA" in orchestrator.failed

class TestExampleSaver:
    """Tests for the ExampleSaver class."""

    @pytest.fixture
    def saver(self, mock_s3):
        with patch('os.makedirs'):
            return ExampleSaver()

    @pytest.fixture
    def test_analysis(self):
        """Sample analysis data."""
        return {
            'results': [
                {
                    'image': 'test1.png',
                    'objects_detected': {
                        'objects': [{
                            'center': [50, 50],
                            'size': 1500,
                            'circularity': 0.8,
                            'peak_brightness': 200,
                            'bounding_box': {
                                'x': 25, 'y': 25,
                                'width': 50, 'height': 50
                            }
                        }]
                    }
                }
            ]
        }

    def test_categorize_view(self, saver):
        """Test view categorization."""
        # Surface view
        surface = {'objects_detected': {'objects': [{'size': 1500, 'circularity': 0.8}]}}
        assert saver.categorize_view(surface) == 'surface'

        # Faraway view
        faraway = {'objects_detected': {'objects': [{'size': 500, 'circularity': 0.9}]}}
        assert saver.categorize_view(faraway) == 'faraway'

    def test_save_examples(self, saver, mock_s3, test_analysis, test_image):
        """Test saving example images."""
        # Mock S3 responses
        mock_s3.get_object.side_effect = [
            {'Body': Mock(read=lambda: json.dumps(test_analysis).encode())},
            {'Body': Mock(read=lambda: cv2.imencode('.png', test_image)[1].tobytes())}
        ]

        with patch('cv2.imwrite') as mock_imwrite, \
             patch('os.listdir', return_value=['example_1.png']):
            saver.save_examples(num_examples=1)
            assert mock_imwrite.called

    def test_error_handling(self, saver, mock_s3):
        """Test error handling."""
        mock_s3.get_object.side_effect = Exception("Test error")
        
        with patch('src.visualization.save_examples.logger.error') as mock_log:
            saver.save_examples()
            mock_log.assert_called_with("Error processing analysis: Test error")

if __name__ == '__main__':
    pytest.main(['-v'])
