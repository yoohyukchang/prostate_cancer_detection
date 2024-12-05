import json
import math
import argparse
import sys
from typing import Dict, Any, List

# Run this file using the command: python3 process_bounding_box_output.py -i bounding_boxes.json -o result.json

# Parses command-line arguments for input and output file paths
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process NIfTI files to find the largest, add padding, and round dimensions.")
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Path to the input JSON file containing NIfTI file data."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="largest_bb_dims.json",
        help="Path to the output JSON file. Defaults to 'largest_bb_dims'."
    )
    return parser.parse_args()

# Loading the json data into a dictionary
def load_json(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Calculate the volume given dimensions of the bounding box
def calculate_volume(dimensions: List[int]) -> int:
    return dimensions[0] * dimensions[1] * dimensions[2]

# Find the file with the largest volume
def find_largest_file(data: Dict[str, Any]) -> str:
    largest_file = None
    max_volume = -1
    for file, info in data.items():
        dimensions = info.get("dimensions")
        volume = calculate_volume(dimensions)
        if volume > max_volume:
            max_volume = volume
            largest_file = file
    return largest_file

# Round dimensions to the nearest power of 2 for easier processing if within specifed tolerance
def round_to_nearest_power_of_2(value: int, tolerance: float = 0.25) -> int:
    if value <= 0:
        return value  # Non-positive values are returned as is
    
    exponent = math.log2(value)
    lower_power = 2 ** int(math.floor(exponent))
    upper_power = 2 ** int(math.ceil(exponent))
    
    # Handle cases where value is exactly a power of 2
    if lower_power == value:
        return value
    
    # Calculate relative differences
    lower_diff = abs(value - lower_power) / value
    upper_diff = abs(upper_power - value) / value
    
    # Decide whether to round down or up
    if lower_diff <= tolerance:
        return lower_power
    elif upper_diff <= tolerance:
        return upper_power
    else:
        return value

# Add padding to the dimension coordinates and round the values
def add_padding_and_round(file_info: Dict[str, Any], padding: int = 2, tolerance: float = 0.25) -> Dict[str, Any]:
    # Add padding to min and max coordinates
    padded_min = [coord - padding for coord in file_info.get("min_coordinates", [])]
    padded_max = [coord + padding for coord in file_info.get("max_coordinates", [])]
    
    if len(padded_min) != 3 or len(padded_max) != 3:
        print("Warning: Coordinates do not have exactly three values. Skipping padding.", file=sys.stderr)
        padded_min = file_info.get("min_coordinates", [])
        padded_max = file_info.get("max_coordinates", [])
    
    # Update dimensions by adding padding on both sides (total +4)
    original_dimensions = file_info.get("dimensions", [])
    if len(original_dimensions) != 3:
        print("Warning: 'dimensions' do not have exactly three values. Skipping dimension update.", file=sys.stderr)
        updated_dimensions = original_dimensions
    else:
        updated_dimensions = [dim + 4 for dim in original_dimensions]
    
    # Round dimensions to nearest power of 2 if within tolerance
    # rounded_dimensions = [
    #     round_to_nearest_power_of_2(dim, tolerance) for dim in updated_dimensions
    # ]
    
    return {
        "min_coordinates": padded_min,
        "max_coordinates": padded_max,
        "dimensions": updated_dimensions
    }

# Processing the data to find the largest bounding box dimensions, add padding, and round dimensions
def process_nifti_files(data: Dict[str, Any]) -> Dict[str, Any]:
    try:
        largest_file = find_largest_file(data)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    original_info = data[largest_file]
    updated_info = add_padding_and_round(original_info)
    
    output = {
        "file_name": largest_file,
        "dimensions": updated_info["dimensions"]
    }
    
    return output

# Save the data to a JSON file.
def save_json(data: Dict[str, Any], file_path: str):
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Output successfully saved to '{file_path}'.")
    except IOError as e:
        print(f"Error: Failed to write to '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

def main():
    # Parse command-line arguments
    args = parse_arguments()
    
    # Load input JSON data
    data = load_json(args.input)
    
    # Process the data
    output = process_nifti_files(data)
    
    # Save the output JSON
    save_json(output, args.output)

if __name__ == "__main__":
    main()
