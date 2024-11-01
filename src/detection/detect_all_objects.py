from detect_objects import AstroObjectDetector

def main():
    # List of targets to analzye
    major_targets = [
        'CALLISTO',
        'EUROPA',
        'GANYMEDE',
        'IO',
        'TITAN',
        'JUPITER',
        'SATURN',
        'URANUS',
        'NEPTUNE'
    ]
    
    detector = AstroObjectDetector()
    
    print("Starting object detection for major solar system bodies...")
    
    for target in major_targets:
        print(f"\n{'='*50}")
        print(f"Processing {target}")
        print(f"{'='*50}")
        
        detector.process_directory(
            f'sorted_png/{target}/',
            brightness_threshold=75,  # Detect bright features
            min_size=20,             # Minimum size to avoid noise
            circularity_threshold=0.7 # How circular the object should be
        )

if __name__ == "__main__":
    main()
