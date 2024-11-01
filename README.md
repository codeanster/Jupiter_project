# Voyager Mission Data Analysis System

A comprehensive data processing system for analyzing Voyager space mission imagery, implementing parallel processing pipelines that transform raw scientific data (IMG/LBL) into analyzed, categorized, and visualized datasets.

## Features

- Raw Voyager data processing (IMG/LBL conversion)
- Automated celestial object detection with configurable parameters
- Intelligent view categorization (surface, faraway, and transitional views)
- Video generation for 50+ celestial targets including major bodies and satellites
- Statistical analysis and visualization
- Parallel processing optimization with configurable worker count
- Extensive test coverage for all major components

## System Requirements

- Python 3.7+
- AWS Account with S3 access
- OpenCV
- NumPy
- Pandas
- PIL (Pillow)
- boto3
- pvl (Planetary Data System Label)
- tqdm

## Project Structure

```
jupiter_project/
├── src/                    # Source code
│   ├── config.py          # Configuration settings
│   ├── processing/        # Data processing modules
│   │   ├── process_images.py    # Image processing pipeline
│   │   └── create_meta.py       # Metadata extraction
│   ├── detection/         # Object detection modules
│   │   ├── detect_objects.py    # Core detection logic
│   │   └── detect_all_objects.py # Batch processing
│   ├── visualization/     # Visualization tools
│   │   ├── create_mp4.py        # Video generation
│   │   ├── create_all_videos.py # Batch video creation
│   │   ├── visualize_detections.py # Detection visualization
│   │   └── save_examples.py     # Example image saving
│   └── utils/            # Utility functions
├── scripts/              # Analysis scripts
│   ├── analyze_callisto.py    # Callisto-specific analysis
│   ├── analyze_examples.py    # Example analysis
│   └── categorize_views.py    # View categorization
├── tests/               # Test suite
│   ├── test_detector.py       # Detection tests
│   ├── test_image_processor.py # Processing tests
│   ├── test_video_generator.py # Video generation tests
│   └── test_visualization.py  # Visualization tests
├── category_examples/   # Categorized image examples
├── detection_examples/  # Detection result examples
├── test_images/        # Test image data
└── videos/             # Generated video outputs
```

## Core Components

### Data Processing Pipeline
- Converts raw Voyager mission data (IMG/LBL pairs) into normalized imagery
- Extracts and processes metadata from PVL labels
- Implements parallel processing for large datasets
- Handles multiple data formats and error cases

### Object Detection System
- Configurable detection parameters:
  - Brightness threshold: 75
  - Size constraints: 20-39550 pixels
  - Circularity threshold: 0.7
- Supports multiple celestial targets including planets and moons
- Provides detailed object metrics (size, circularity, brightness)

### View Classification
- Categorizes images into:
  - Surface views
  - Faraway views
  - Transitional views
- Uses size and circularity thresholds for classification
- Saves categorized examples for verification

### Video Generation
- Creates timelapse sequences for 50+ celestial targets
- Configurable FPS (default: 3) and bitrate (default: 2000k)
- Supports parallel video generation
- Includes metadata overlay on frames

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/jupiter_project.git
cd jupiter_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure AWS credentials for S3 access (still working on this)

4. Update configuration in `src/config.py` if needed

## Usage

1. Process raw Voyager data:
```bash
python src/processing/process_images.py
```

2. Detect celestial objects:
```bash
python src/detection/detect_all_objects.py
```

3. Generate videos:
```bash
python src/visualization/create_all_videos.py
```

4. Run analysis:
```bash
python scripts/analyze_callisto.py  # For Callisto-specific analysis
python scripts/categorize_views.py  # For view categorization
```

## Testing

The project includes comprehensive tests for all major components:

```bash
pytest tests/  # Run all tests
pytest tests/test_detector.py  # Test detection system
pytest tests/test_image_processor.py  # Test image processing
pytest tests/test_video_generator.py  # Test video generation
pytest tests/test_visualization.py  # Test visualization
```

## Data Organization

The system organizes data into the following categories:
- Raw Voyager data (IMG/LBL pairs)
- Processed PNG images
- Detection results (JSON format)
- Categorized examples
- Generated videos
- Statistical analyses

## Supported Celestial Targets

Major targets include:
- Planets: Jupiter, Saturn, Uranus, Neptune
- Galilean moons: Io, Europa, Ganymede, Callisto
- Other satellites: Titan, Miranda, and many more

Full list of 50+ targets is configured in `src/config.py`

## Acknowledgments

- NASA Voyager Mission data
- AWS for cloud infrastructure
- OpenCV community
- Python scientific computing community