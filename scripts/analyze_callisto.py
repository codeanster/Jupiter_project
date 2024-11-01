import json
import boto3
from collections import defaultdict
import statistics

def download_analysis(bucket, key):
    """Download and parse the analysis JSON file"""
    s3 = boto3.client('s3')
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return json.loads(response['Body'].read().decode('utf-8'))
    except Exception as e:
        print(f"Error downloading analysis: {str(e)}")
        return None

def analyze_detections(data):
    """Analyze the detection results"""
    if not data:
        return
    
    total_images = len(data['results'])
    total_objects = sum(len(result['objects_detected']['objects']) 
                       for result in data['results'])
    
    # Collect statistics about detected objects
    sizes = []
    circularities = []
    brightnesses = []
    
    # Count objects per image
    objects_per_image = defaultdict(int)
    
    for result in data['results']:
        img_name = result['image'].split('/')[-1]
        num_objects = len(result['objects_detected']['objects'])
        objects_per_image[num_objects] += 1
        
        for obj in result['objects_detected']['objects']:
            sizes.append(obj['size'])
            circularities.append(obj['circularity'])
            brightnesses.append(obj['peak_brightness'])
    
    print("\n=== Callisto Object Detection Analysis ===")
    print(f"\nAnalysis Parameters:")
    print(f"Brightness threshold: {data['analysis_parameters']['brightness_threshold']}")
    print(f"Minimum size: {data['analysis_parameters']['min_size']}")
    print(f"Circularity threshold: {data['analysis_parameters']['circularity_threshold']}")
    
    print(f"\nOverall Statistics:")
    print(f"Total images analyzed: {total_images}")
    print(f"Total objects detected: {total_objects}")
    print(f"Average objects per image: {total_objects/total_images:.2f}")
    
    print("\nObject Distribution:")
    for num_obj, count in sorted(objects_per_image.items()):
        print(f"Images with {num_obj} object(s): {count}")
    
    if sizes:
        print("\nObject Characteristics:")
        print(f"Size - Min: {min(sizes)}, Max: {max(sizes)}, Avg: {statistics.mean(sizes):.2f}")
        print(f"Circularity - Min: {min(circularities):.3f}, Max: {max(circularities):.3f}, Avg: {statistics.mean(circularities):.3f}")
        print(f"Peak Brightness - Min: {min(brightnesses)}, Max: {max(brightnesses)}, Avg: {statistics.mean(brightnesses):.2f}")

def main():
    analysis_data = download_analysis(
        'astro-data-io',
        'object_detection/CALLISTO_analysis.json'
    )
    
    if analysis_data:
        analyze_detections(analysis_data)
    else:
        print("Could not retrieve analysis data!")

if __name__ == "__main__":
    main()
