import os
import pvl
import csv
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import logging
from datetime import datetime

from src.config import (
    S3_BUCKET,
    INPUT_PREFIX,
    TEMP_DIR,
    PARALLEL_WORKERS,
    META_FILE_PATH
)

# Initialize logging
logging.basicConfig(
    filename='/home/ubuntu/jupiter_project/create_meta_file.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

# Initialize S3 client
s3 = boto3.client('s3')
bucket_name = S3_BUCKET
input_prefix = INPUT_PREFIX

# Temporary directory for processing files
temp_dir = TEMP_DIR
os.makedirs(temp_dir, exist_ok=True)

def list_s3_files():
    """
    List and match IMG and LBL files in the specified S3 bucket and prefix.
    Returns a list of tuples: (base_name, img_s3_key, lbl_s3_key)
    """
    paginator = s3.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket_name, Prefix=input_prefix)

    img_files = {}
    lbl_files = {}

    for page in pages:
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith(('.img', '.IMG')):
                base_name = os.path.splitext(os.path.basename(key))[0]
                img_files[base_name] = key
            elif key.endswith(('.lbl', '.LBL')):
                base_name = os.path.splitext(os.path.basename(key))[0]
                lbl_files[base_name] = key

    # Match IMG and LBL files by base_name
    base_names = set(img_files.keys()) & set(lbl_files.keys())
    matched_files = [(base, img_files[base], lbl_files[base]) for base in base_names]
    return matched_files

def download_from_s3(s3_key, local_path):
    """
    Download a file from S3 to a local path.
    Returns True and empty string on success, False and error message on failure.
    """
    try:
        s3.download_file(bucket_name, s3_key, local_path)
        return True, ""
    except (NoCredentialsError, ClientError) as e:
        return False, str(e)

def process_file(args):
    """
    Process a single image pair, download the LBL file, extract metadata, and return the data.
    """
    base_name, img_s3_key, lbl_s3_key = args
    local_lbl = os.path.join(temp_dir, os.path.basename(lbl_s3_key))

    success, error = download_from_s3(lbl_s3_key, local_lbl)
    if not success:
        logging.error(f"Failed to download {lbl_s3_key}: {error}")
        return None

    try:
        with open(local_lbl, 'r') as lbl_file:
            lbl = pvl.load(lbl_file)

        mission_phase_name = lbl.get('MISSION_PHASE_NAME', 'Unknown')
        target_name = lbl.get('TARGET_NAME', 'Unknown')

        # Extract the image time to format the new filename
        image_time = lbl.get('IMAGE_TIME', None)
        if image_time:
            # Updated format to handle time zone and no 'T' separator
            date_str = datetime.strptime(str(image_time), '%Y-%m-%d %H:%M:%S%z')
            formatted_date = date_str.strftime('%Y%m%d_%H%M%S')
        else:
            formatted_date = "unknown_date"

        # Check if "CLEANED" is already in the base name
        if "CLEANED" not in base_name:
            new_file_name = f"{base_name}_CLEANED_{formatted_date}.png"
        else:
            new_file_name = f"{base_name}_{formatted_date}.png"

        # Clean up the local file
        os.remove(local_lbl)

        return {
            'File Path': new_file_name,
            'Mission Phase Name': mission_phase_name,
            'Target Name': target_name
        }

    except Exception as e:
        logging.error(f"Error processing {lbl_s3_key}: {e}")
        return None

def create_meta_file():
    """
    Create a CSV metafile containing the file paths, mission phase, and target name.
    """
    base_files = list_s3_files()
    if not base_files:
        print("No matching IMG and LBL file pairs found in S3.")
        return

    with open(META_FILE_PATH, mode='w', newline='') as csv_file:
        fieldnames = ['File Path', 'Mission Phase Name', 'Target Name']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        # Progress bar and multiprocessing
        with tqdm(total=len(base_files), desc="Creating Meta File", unit="file") as pbar:
            with ProcessPoolExecutor(max_workers=PARALLEL_WORKERS) as executor:
                # Submit all tasks to the executor
                futures = [executor.submit(process_file, args) for args in base_files]
                
                # Process the results as they complete
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        writer.writerow(result)
                    pbar.update(1)

    logging.info(f"Metafile created at {META_FILE_PATH}.")
    print(f"Metafile created at {META_FILE_PATH}.")

if __name__ == "__main__":
    create_meta_file()
