import os
import json
import numpy as np
import nibabel as nib
from scipy.ndimage import label

# Finds largest continuous prostate chunk
def find_largest_prostate(volume):
    labeled_volume, num_features = label(volume)
    largest_chunk = np.zeros_like(volume)

    if num_features == 0:
        return largest_chunk

    sizes = np.bincount(labeled_volume.flat)
    largest_label = sizes[1:].argmax() + 1  # Ignore label 0 (background)
    largest_chunk[labeled_volume == largest_label] = 1

    return largest_chunk

# Finding largest chunk and setting everything to 0
def preprocess_data(volume):
    largest_chunk = find_largest_prostate(volume)
    return largest_chunk

# Add padding 
def crop_prostate(volume, min_indices, max_indices, padding=10):
    min_indices = np.maximum(min_indices - padding, 0)
    max_indices = np.minimum(max_indices + padding, volume.shape)

    slices = tuple(slice(min_idx, max_idx) for min_idx, max_idx in zip(min_indices, max_indices))
    cropped_volume = volume[slices]
    return cropped_volume, min_indices, max_indices

def find_bounding_box(volume):
    # Find the non-zero regions in all dimensions
    non_zero_indices = np.argwhere(volume)
    min_indices = non_zero_indices.min(axis=0)
    max_indices = non_zero_indices.max(axis=0)
    return min_indices, max_indices

def process_directory(input_directory, output_json):
    bounding_boxes = {}

    for filename in os.listdir(input_directory):
        if filename.endswith(".nii.gz"):
            filepath = os.path.join(input_directory, filename)
            volume = nib.load(filepath).get_fdata()

            # Preprocess the volume to keep only the largest continuous chunk
            processed_volume = preprocess_data(volume)

            # Find the bounding box coordinates
            min_indices, max_indices = find_bounding_box(processed_volume)

            # Crop around the prostate center
            cropped_volume, min_indices, max_indices = crop_prostate(processed_volume, min_indices, max_indices)

            # Store the bounding box dimensions and coordinates
            bounding_boxes[filename] = {
                "min_coordinates": min_indices.tolist(),
                "max_coordinates": max_indices.tolist(),
                "dimensions": (max_indices - min_indices).tolist()
            }

    # Save bounding box information to a JSON file
    with open(output_json, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=4)


if __name__ == "__main__":
    input_directory = "/Users/averykuo/Downloads/Prostate/Prostates"  # Change as needed
    output_json = "bounding_boxes.json"

    process_directory(input_directory, output_json)
    print(f"Bounding box information saved to {output_json}")
