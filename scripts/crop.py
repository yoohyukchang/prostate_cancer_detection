import json
import os
import nibabel as nib
import numpy as np


def crop_files(image_path, label_path, image_output_path, label_output_path):
    # Read json
    with open('bounding_boxes.json', 'r') as file:
        data = json.load(file)

    # Loop over every file
    for filename in os.listdir(label_path):
        print(filename)
        print(image_path)
        # Read nifti
        index = filename[5:]
        image_filepath = os.path.join(image_path, f'legion{index}')
        label_filepath = os.path.join(label_path, f'label{index}')
        print(image_filepath)
        image_nifti = nib.load(image_filepath)
        label_nifti = nib.load(label_filepath)

        # Extract fdata
        image_data = image_nifti.get_fdata()
        label_data = label_nifti.get_fdata()

        # Extract affine information
        image_affine = image_nifti.affine
        label_affine = label_nifti.affine

        # Read bounding box coordinates (label only)
        min_indices = data[filename]['min_coordinates']
        max_indices = data[filename]['max_coordinates']

        # Update bounding box to allow space
        signs = [np.sign(label_affine[0][0]), np.sign(label_affine[1][1]), np.sign(label_affine[2][2])]
        min_indices[0], max_indices[0] = min_indices[0] + 10 * signs[0], max_indices[0] - 10 * signs[0]
        min_indices[1], max_indices[1] = min_indices[1] + 10 * signs[1], max_indices[1] - 10 * signs[1]
        min_indices[2], max_indices[2] = min_indices[2] - 5 * signs[2], max_indices[2] + 5 * signs[2]
        min_indices[0], max_indices[0] = int(min_indices[0]), int(max_indices[0])
        min_indices[1], max_indices[1] = int(min_indices[1]), int(max_indices[1])
        min_indices[2], max_indices[2] = int(min_indices[2]), int(max_indices[2])

        # Convert to world coordinates using label_affine
        min_world = np.dot(label_affine, [*min_indices, 1])[:3]  # Add 1 for homogeneous coordinates
        max_world = np.dot(label_affine, [*max_indices, 1])[:3]

        # Convert world coordinates to voxel indices in the image using image_affine
        image_affine_inv = np.linalg.inv(image_affine)
        converted_min_indices = np.dot(image_affine_inv, [*min_world, 1])[:3].astype(int)
        converted_max_indices = np.dot(image_affine_inv, [*max_world, 1])[:3].astype(int)

        # Round and cast to integers
        image_min_indices = np.minimum(converted_min_indices, converted_max_indices)
        image_max_indices = np.maximum(converted_min_indices, converted_max_indices)

        # Crop file
        image_data = image_data[
            image_min_indices[0]:image_max_indices[0],
            image_min_indices[1]:image_max_indices[1],
            image_min_indices[2]:image_max_indices[2]
        ]
        label_data = label_data[
            min_indices[0]:max_indices[0],
            min_indices[1]:max_indices[1],
            min_indices[2]:max_indices[2],
        ]

        # Configure the new image affine
        image_affine[0][3] = image_affine[0][3] + image_affine[0][0] * image_min_indices[0]
        image_affine[1][3] = image_affine[1][3] + image_affine[1][1] * image_min_indices[1]
        image_affine[2][3] = image_affine[2][3] + image_affine[2][2] * image_min_indices[2]

        # Configure the new label affine
        label_affine[0][3] = label_affine[0][3] + label_affine[0][0] * min_indices[0]
        label_affine[1][3] = label_affine[1][3] + label_affine[1][1] * min_indices[1]
        label_affine[2][3] = label_affine[2][3] + label_affine[2][2] * min_indices[2]

        # Create a new NIfTI image using the cropped data and the original affine
        image_nifti = nib.Nifti1Image(image_data, image_affine)
        label_nifti = nib.Nifti1Image(label_data, label_affine)

        # Save the cropped image
        final_image_output_path = os.path.join(image_output_path, f'cropped_legion{index}')
        final_label_output_path = os.path.join(label_output_path, f'cropped_label{index}')
        nib.save(image_nifti, final_image_output_path)
        # nib.save(label_nifti, final_label_output_path)

        print(f"Cropped images saved to {image_output_path} and {label_output_path}")


def main():
    image_path = r'D:\Ryan\datasets\Prostate\Model Files\legions'
    label_path = r'D:\Ryan\datasets\Prostate\Model Files\labels'
    image_output_path = r'D:\Ryan\datasets\Prostate\Model Files\cropped_legions'
    label_output_path = r'D:\Ryan\datasets\Prostate\Model Files\cropped_labels'
    crop_files(image_path, label_path, image_output_path, label_output_path)


if __name__ == '__main__':
    main()