import boto3
import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class ViewAnalyzer:
    def __init__(self, bucket_name="astro-data-io"):
        self.s3 = boto3.client('s3')
        self.bucket_name = bucket_name
        
    def load_analysis(self):
        """Load the analysis results from S3"""
        try:
            response = self.s3.get_object(
                Bucket=self.bucket_name,
                Key='object_detection/CALLISTO_analysis.json'
            )
            return json.loads(response['Body'].read().decode('utf-8'))
        except Exception as e:
            print(f"Error loading analysis: {str(e)}")
            return None
    
    def categorize_view(self, result):
        """Categorize a view based on object characteristics"""
        objects = result['objects_detected']['objects']
        
        # Sort objects by size
        objects = sorted(objects, key=lambda x: x['size'], reverse=True)
        
        if not objects:
            return None
        
        largest_obj = objects[0]
        num_objects = len(objects)
        
        # Calculate average circularity
        avg_circularity = np.mean([obj['circularity'] for obj in objects])
        
        # Surface view criteria
        if largest_obj['size'] > 1000 or (num_objects > 10 and largest_obj['size'] > 500):
            return 'surface'
        # Faraway view criteria
        elif largest_obj['size'] < 1000 and avg_circularity > 0.75:
            return 'faraway'
        else:
            return 'other'
    
    def analyze_views(self):
        """Analyze and categorize all views"""
        analysis = self.load_analysis()
        if not analysis:
            return
        
        categories = {
            'surface': [],
            'faraway': [],
            'other': []
        }
        
        for result in analysis['results']:
            category = self.categorize_view(result)
            if category:
                # Store relevant metrics
                objects = result['objects_detected']['objects']
                largest_obj = max(objects, key=lambda x: x['size'])
                metrics = {
                    'image': result['image'],
                    'num_objects': len(objects),
                    'largest_size': largest_obj['size'],
                    'avg_circularity': np.mean([obj['circularity'] for obj in objects]),
                    'timestamp': result['timestamp']
                }
                categories[category].append(metrics)
        
        return categories
    
    def plot_categories(self, categories):
        """Create plots to visualize the categories"""
        plt.style.use('dark_background')
        
        # Plot 1: Size vs Number of Objects scatter plot
        plt.figure(figsize=(12, 8))
        colors = {'surface': 'red', 'faraway': 'blue', 'other': 'gray'}
        
        for category, metrics in categories.items():
            if metrics:
                sizes = [m['largest_size'] for m in metrics]
                nums = [m['num_objects'] for m in metrics]
                plt.scatter(sizes, nums, c=colors[category], label=category, alpha=0.6)
        
        plt.xscale('log')
        plt.xlabel('Largest Object Size (pixels)')
        plt.ylabel('Number of Objects Detected')
        plt.title('View Classification by Object Size and Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('view_classification_size_count.png', dpi=300, bbox_inches='tight')
        
        # Plot 2: Size vs Circularity scatter plot
        plt.figure(figsize=(12, 8))
        
        for category, metrics in categories.items():
            if metrics:
                sizes = [m['largest_size'] for m in metrics]
                circularity = [m['avg_circularity'] for m in metrics]
                plt.scatter(sizes, circularity, c=colors[category], label=category, alpha=0.6)
        
        plt.xscale('log')
        plt.xlabel('Largest Object Size (pixels)')
        plt.ylabel('Average Circularity')
        plt.title('View Classification by Object Size and Circularity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('view_classification_size_circularity.png', dpi=300, bbox_inches='tight')
        
        # Print statistics
        print("\nView Classification Statistics:")
        print("=" * 50)
        for category, metrics in categories.items():
            if metrics:
                print(f"\n{category.upper()} Views:")
                print(f"Count: {len(metrics)}")
                print(f"Average object count: {np.mean([m['num_objects'] for m in metrics]):.2f}")
                print(f"Average largest size: {np.mean([m['largest_size'] for m in metrics]):.2f}")
                print(f"Average circularity: {np.mean([m['avg_circularity'] for m in metrics]):.3f}")
        
        # Upload plots to S3
        try:
            self.s3.upload_file(
                'view_classification_size_count.png',
                self.bucket_name,
                'analysis/view_classification_size_count.png'
            )
            self.s3.upload_file(
                'view_classification_size_circularity.png',
                self.bucket_name,
                'analysis/view_classification_size_circularity.png'
            )
            print("\nPlots uploaded to S3 in the analysis/ directory!")
        except Exception as e:
            print(f"\nError uploading plots: {str(e)}")

def main():
    analyzer = ViewAnalyzer()
    categories = analyzer.analyze_views()
    if categories:
        analyzer.plot_categories(categories)

if __name__ == "__main__":
    main()
