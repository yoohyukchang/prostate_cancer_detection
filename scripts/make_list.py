import os

# File paths
common_files_path = r"D:\Ryan\datasets\Prostate\Model Files\common_files.txt"
legions_dir = r"D:\Ryan\datasets\Prostate\Model Files\legions"

# Load the patient indices from common_files.txt
with open(common_files_path, 'r') as file:
    valid_indices = {line.strip() for line in file}

# Check for corresponding legion files and remove invalid indices
updated_indices = set()
for patient_id in valid_indices:
    legion_file = os.path.join(legions_dir, f"legion{patient_id}.nii.gz")
    if os.path.isfile(legion_file):
        updated_indices.add(patient_id)

# Write updated indices back to common_files.txt
with open(common_files_path, 'w') as file:
    for patient_id in sorted(updated_indices):
        file.write(f"{patient_id}\n")

# Delete any legion files that aren't in the updated list
for legion_file in os.listdir(legions_dir):
    if legion_file.startswith("legion") and legion_file.endswith(".nii.gz"):
        patient_id = legion_file[6:10]  # Extract patient number from file name
        if patient_id not in updated_indices:
            os.remove(os.path.join(legions_dir, legion_file))
            print(f"Deleted {legion_file}")

print("Update and cleanup complete.")