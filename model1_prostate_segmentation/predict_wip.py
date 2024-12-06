from pathlib import Path
import torch
import monai
from monai.transforms import Invertd
import nibabel as nib
import numpy as np
from numpy.linalg import inv
import os
import warnings
from model import DualNet_seperate_load
from data_loader import general_transform
from nibabel.orientations import aff2axcodes


def main():
    # Silence warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Basic parameters
    crop_size = [128, 128, 64]
    resample_spacing = [1.5, 1.5, 1.5]

    # Setup paths
    current_path = Path(__file__).parent
    root_path = current_path.parent
    data_path = root_path / 'data'

    # For the 'Original' dataset, we assume data is in data/Test/MRIs and data/Test/Prostates
    test_image_dir = data_path / 'Test' / 'MRIs'
    test_label_dir = data_path / 'Test' / 'Prostates'

    # Get test images and labels
    test_images = sorted(list(test_image_dir.glob('*.nii.gz')))
    test_labels = sorted(list(test_label_dir.glob('*.nii.gz')))

    # Create test dict
    test_dicts = [{'image': str(img), 'label': str(lbl)} for img, lbl in zip(test_images, test_labels)]

    # Define custom transforms
    transform = general_transform(crop_size, resample_spacing)

    test_dataset = monai.data.Dataset(data=test_dicts, transform=transform)
    test_loader = monai.data.DataLoader(test_dataset, batch_size=1)
    print('Loader created')

    # Load model
    # Make sure the UNET_Transformer_model_prostate.pt file is in the current directory
    model_path = current_path / 'UNET_Transformer_model_prostate.pt'
    model = DualNet_seperate_load(device=device, crop_patch_size=crop_size, out_channels=1)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print('Model loaded from', model_path)

    # Create directory for predicted labels if needed
    pred_dir = current_path / 'predicted_labels'
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Create directories for outputs, segmentation, truth, and image files
    output_dir = pred_dir / 'output'
    segmentation_dir = pred_dir / 'segmentation'
    truth_dir = pred_dir / 'truth'
    image_dir = pred_dir / 'image'

    output_dir.mkdir(parents=True, exist_ok=True)
    segmentation_dir.mkdir(parents=True, exist_ok=True)
    truth_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    # Prediction loop
    for i, batch in enumerate(test_loader):
        model.zero_grad()

        # Extract the original image volume and segmentation information
        original_image = nib.load(test_dicts[i]['image'])
        original_affine = original_image.affine
        original_size = original_image.shape
        original_spacing = [abs(original_affine[0][0]), abs(original_affine[0][0]), abs(original_affine[0][0])]
        signs = np.sign([original_affine[0][0], original_affine[1][1], original_affine[2][2]])
        corner_1 = np.dot(original_affine, np.array([[0], [0], [0], [0]]))
        corner_2 = np.dot(original_affine, np.transpose(np.array(list(original_size) + [0])))
        print(corner_1, corner_2)
        break

        # Perform prediction
        image = batch['image'].to(device)
        output = model(image).squeeze().cpu().detach().numpy()
        
        # Generate binary segmentation
        output_segmentation = (output > 0.5).astype(np.uint8)

        # Construct output filename
        image_filename = os.path.basename(test_dicts[i]['image'])
        # Example naming strategy if needed
        # Adjust if you have a specific naming format
        identifier = image_filename.split('image')[1].split('.nii.gz')[0]
        output_filename = f'image{identifier}_prediction.nii.gz'

        # Save outputs
        nifti_output = nib.Nifti1Image(output, affine=original_affine)
        nifti_segmentation = nib.Nifti1Image(output_segmentation, affine=original_affine)

        # Save files in respective directories
        nib.save(nifti_output, str(output_dir / output_filename))
        nib.save(nifti_segmentation, str(segmentation_dir / f'segmentation_{output_filename}'))

        print(f'Saved {output_filename} to organized directories')

if __name__ == '__main__':
    main()