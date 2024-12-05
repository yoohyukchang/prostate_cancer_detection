import os
import slicer
import SimpleITK as sitk

# Define paths
input_root = r"D:\Ryan\datasets\Prostate\Prostate-MRI-US-Biopsy-DICOM-annotations"
output_root = r"D:\Ryan\datasets\Prostate\Model Files\legions"

# Loop over each patient
for patient_num in range(1, 1152):
    # Format patient directory and check if it exists
    patient_dir = os.path.join(input_root, f"Prostate-MRI-US-Biopsy-{patient_num:04d}")
    if not os.path.isdir(patient_dir):
        continue

    # Get list of subfolders (patient IDs) within the patient directory
    subfolders = [f for f in os.listdir(patient_dir) if os.path.isdir(os.path.join(patient_dir, f))]
    if len(subfolders) != 1:
        # Skip if multiple patient ID subfolders are found
        continue

    # Define paths for DICOM file and output NIfTI file
    dicom_path = os.path.join(patient_dir, subfolders[0], "300-SEG-Segmentation of prostate lesion 1.dcm")
    output_path = os.path.join(output_root, f"legion{patient_num:04d}.nii.gz")

    # Check if the DICOM file exists
    if not os.path.isfile(dicom_path):
        continue

    # Read DICOM and convert to NIfTI
    dicom_image = sitk.ReadImage(dicom_path)
    sitk.WriteImage(dicom_image, output_path)

    print(f"Converted {dicom_path} to {output_path}")
