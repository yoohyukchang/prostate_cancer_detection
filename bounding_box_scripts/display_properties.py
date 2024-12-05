import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import statistics as st

# Define the directory containing the .nii.gz files
image_dir = r"D:\Ryan\datasets\Prostate\Model Files\images"

# Initialize lists to store dimensions and spacings
x_dims, y_dims, z_dims = [], [], []
x_spacing, y_spacing, z_spacing = [], [], []

# Iterate over the image files
for i in range(1, 1152):
    # Format the file name with zero padding (e.g., image0001.nii.gz)
    file_path = os.path.join(image_dir, f"image{i:04d}.nii.gz")

    # Check if the file exists
    if os.path.isfile(file_path):
        # Load the NIfTI file using nibabel
        img = nib.load(file_path)

        # Get the dimensions (shape) and spacing (affine)
        dims = img.shape  # (x_dim, y_dim, z_dim)
        spacing = img.header.get_zooms()  # (x_spacing, y_spacing, z_spacing)

        # Append the dimensions and spacings to their respective lists
        x_dims.append(dims[0])
        y_dims.append(dims[1])
        z_dims.append(dims[2])
        x_spacing.append(spacing[0])
        y_spacing.append(spacing[1])
        z_spacing.append(spacing[2])

# Display modes
print(f'X dimension mode: {st.mode(x_dims)}')
print(f'Y dimension mode: {st.mode(y_dims)}')
print(f'Z dimension mode: {st.mode(z_dims)}')
print(f'X spacing mode: {st.mode(x_spacing)}')
print(f'Y spacing mode: {st.mode(y_spacing)}')
print(f'Z spacing mode: {st.mode(z_spacing)}')

# Create histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 5))

axes[0, 0].hist(x_dims, bins=500, color='blue', edgecolor='black')
axes[0, 0].set_title('Distribution of X Dimensions (Pixels)')
axes[0, 0].set_xlabel('Number of Pixels (X)')
axes[0, 0].set_ylabel('Frequency')

axes[0, 1].hist(y_dims, bins=500, color='green', edgecolor='black')
axes[0, 1].set_title('Distribution of Y Dimensions (Pixels)')
axes[0, 1].set_xlabel('Number of Pixels (Y)')
axes[0, 1].set_ylabel('Frequency')

axes[0, 2].hist(z_dims, bins=500, color='red', edgecolor='black')
axes[0, 2].set_title('Distribution of Z Dimensions (Pixels)')
axes[0, 2].set_xlabel('Number of Pixels (Z)')
axes[0, 2].set_ylabel('Frequency')

axes[1, 0].hist(x_spacing, bins=500, color='blue', edgecolor='black')
axes[1, 0].set_title('Distribution of X Spacing (mm)')
axes[1, 0].set_xlabel('Spacing (mm) (X)')
axes[1, 0].set_ylabel('Frequency')

axes[1, 1].hist(y_spacing, bins=500, color='green', edgecolor='black')
axes[1, 1].set_title('Distribution of Y Spacing (mm)')
axes[1, 1].set_xlabel('Spacing (mm) (Y)')
axes[1, 1].set_ylabel('Frequency')

axes[1, 2].hist(z_spacing, bins=500, color='red', edgecolor='black')
axes[1, 2].set_title('Distribution of Z Spacing (mm)')
axes[1, 2].set_xlabel('Spacing (mm) (Z)')
axes[1, 2].set_ylabel('Frequency')

# Adjust layout for spacings
plt.tight_layout()
plt.show()