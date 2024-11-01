"""Video generation orchestrator for Voyager mission imagery.

This module coordinates the creation of timelapse videos for all celestial targets
in the dataset, managing parallel processing and tracking completion status.
"""

import os
import sys
import logging
import subprocess
import concurrent.futures
from pathlib import Path
from typing import List, Set, Dict, Optional

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.config import (
    EXISTING_VIDEOS,
    ALL_TARGETS,
    PARALLEL_WORKERS
)

# Configure logging
logging.basicConfig(
    filename='visualization.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

logger = logging.getLogger(__name__)

class VideoOrchestrator:
    """Orchestrator for managing video creation across multiple targets."""

    def __init__(
        self,
        max_workers: int = 4,
        output_dir: str = "videos"
    ) -> None:
        """Initialize the video orchestrator.
        
        Args:
            max_workers: Maximum number of parallel video creation processes
            output_dir: Directory to store the generated videos
        """
        self.max_workers = max_workers
        self.output_dir = output_dir
        self.completed: Set[str] = set()
        self.failed: Set[str] = set()
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Initialized VideoOrchestrator with output directory: {output_dir}")

    def create_video(self, target: str) -> bool:
        """Create video for a specific target.
        
        Args:
            target: Name of the celestial target
            
        Returns:
            bool: True if video creation was successful, False otherwise
        """
        output_file = f"{self.output_dir}/{target.lower()}.mp4"
        input_dir = f"sorted_png/{target}"
        
        logger.info(f"Starting video creation for {target}")
        try:
            subprocess.run(
                ['python3', 'src/visualization/create_mp4.py', input_dir, output_file],
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Completed video creation for {target}")
            self.completed.add(target)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Error creating video for {target}: {e.stderr}")
            self.failed.add(target)
            return False
        except Exception as e:
            logger.error(f"Unexpected error creating video for {target}: {str(e)}")
            self.failed.add(target)
            return False

    def process_all_targets(self, targets: Optional[List[str]] = None) -> None:
        """Process all targets in parallel.
        
        Args:
            targets: Optional list of targets to process. If None, processes all
                    targets except those in EXISTING_VIDEOS.
        """
        targets_to_process = targets or [t for t in ALL_TARGETS if t not in EXISTING_VIDEOS]
        
        logger.info(f"Starting video creation for {len(targets_to_process)} targets")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.create_video, target): target 
                for target in targets_to_process
            }
            
            for future in concurrent.futures.as_completed(futures):
                target = futures[future]
                try:
                    success = future.result()
                    if success:
                        logger.info(f"Successfully created video for {target}")
                    else:
                        logger.error(f"Failed to create video for {target}")
                except Exception as e:
                    logger.error(f"Error processing {target}: {str(e)}")
                    self.failed.add(target)

    def get_status(self) -> Dict[str, Set[str]]:
        """Get the current status of video creation.
        
        Returns:
            Dict containing sets of completed and failed targets
        """
        return {
            'completed': self.completed,
            'failed': self.failed
        }

def main() -> None:
    """Main entry point for batch video creation."""
    orchestrator = VideoOrchestrator(max_workers=PARALLEL_WORKERS)
    orchestrator.process_all_targets()
    
    status = orchestrator.get_status()
    logger.info("\nProcessing completed!")
    logger.info(f"Successfully created videos: {len(status['completed'])}")
    logger.info(f"Failed videos: {len(status['failed'])}")
    
    if status['failed']:
        logger.info("\nFailed targets:")
        for target in status['failed']:
            logger.info(f"- {target}")

if __name__ == "__main__":
    main()
