import json
import boto3
from datetime import datetime

def get_image_info(result):
    """Format detection information for an image"""
    info = []
    image_key = result['image']
    objects = result['objects_detected']['objects']
    timestamp = result['timestamp']
    
    info.append(f"\nImage: {image_key.split('/')[-1]}")
    info.append(f"Timestamp: {timestamp}")
    info.append(f"Number of objects: {len(objects)}")
    
    # Sort objects by size to highlight main features
    objects = sorted(objects, key=lambda x: x['size'], reverse=True)
    
    for i, obj in enumerate(objects, 1):
        info.append(f"\nObject #{i}:")
        info.append(f"  Size: {obj['size']} pixels")
        info.append(f"  Circularity: {obj['circularity']:.3f}")
        info.append(f"  Peak Brightness: {obj['peak_brightness']}")
        info.append(f"  Center: ({obj['center'][0]}, {obj['center'][1]})")
        box = obj['bounding_box']
        info.append(f"  Bounding Box: x={box['x']}, y={box['y']}, w={box['width']}, h={box['height']}")
    
    return "\n".join(info)

def main():
    s3 = boto3.client('s3')
    
    try:
        # Get the analysis JSON
        response = s3.get_object(
            Bucket='astro-data-io',
            Key='object_detection/CALLISTO_analysis.json'
        )
        analysis = json.loads(response['Body'].read().decode('utf-8'))
        
        print("\n=== Callisto Detection Examples Analysis ===")
        print(f"Analysis Parameters:")
        print(f"Brightness threshold: {analysis['analysis_parameters']['brightness_threshold']}")
        print(f"Minimum size: {analysis['analysis_parameters']['min_size']}")
        print(f"Circularity threshold: {analysis['analysis_parameters']['circularity_threshold']}")
        
        # Create lookup of all results by image name
        results_by_image = {result['image']: result for result in analysis['results']}
        
        print("\n=== Single Object Detection Examples ===")
        print("=" * 50)
        for i in range(1, 6):
            for key in results_by_image.keys():
                if f"C{i}" in key:  # Match the image by its sequence number
                    print(get_image_info(results_by_image[key]))
                    break
        
        print("\n=== Multiple Object Detection Examples ===")
        print("=" * 50)
        # Get results with most objects first
        multi_results = sorted(
            [r for r in analysis['results'] if len(r['objects_detected']['objects']) > 1],
            key=lambda x: len(x['objects_detected']['objects']),
            reverse=True
        )
        
        for result in multi_results[:5]:
            print(get_image_info(result))
            
    except Exception as e:
        print(f"Error processing analysis: {str(e)}")

if __name__ == "__main__":
    main()
