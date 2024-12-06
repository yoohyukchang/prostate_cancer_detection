from pathlib import Path
import torch
import monai
from monai.transforms import ScaleIntensity, Spacing, ResizeWithPadOrCrop, ToTensor, EnsureType
import nibabel as nib
import numpy as np
import os
import warnings
from sklearn.model_selection import train_test_split  # if needed, else remove
from model import DualNet_seperate_load
from data_loader import general_transform

if __name__ == '__main__':
    # Silence warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Choose dataset type: 'Example' or 'Original'
    dataset = 'Original'

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
    if dataset == 'Example':
        # Example dataset paths (if you have them)
        # Adjust if you have a different structure for 'Example'
        test_image_dir = data_path / 'Test' / 'MRIs'
        test_label_dir = data_path / 'Test' / 'Prostates'
    elif dataset == 'Original':
        test_image_dir = data_path / 'Test' / 'MRIs'
        test_label_dir = data_path / 'Test' / 'Prostates'
    else:
        raise ValueError("dataset must be 'Example' or 'Original'.")

    # Get test images and labels
    test_images = sorted(list(test_image_dir.glob('*.nii.gz')))
    test_labels = sorted(list(test_label_dir.glob('*.nii.gz')))

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

    # Create base directory for predicted labels
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

        # Extract the original image volume information
        original_image = nib.load(test_dicts[i]['image'])
        original_affine = original_image.affine
        original_size = original_image.shape

        # Determine flip dimensions if needed (currently commented out)
        flip_dimensions = []
        # if you need orientation correction, adjust here:
        # for dim in range(3):
        #     if original_affine[dim][dim] < 0:
        #         flip_dimensions.append(dim + 2)

        image = batch['image'].to(device)
        if len(flip_dimensions) > 0:
            image = torch.flip(image, dims=flip_dimensions)

        # Perform prediction
        output = model(image).squeeze().cpu().detach().numpy()
        truth = batch['label'].detach().squeeze().numpy()

        # Generate binary segmentation
        output_segmentation = (output > 0.5).astype(np.uint8)

        # Use a simple affine for output (if you prefer original affine, adapt as needed)
        # Here we just use a scaling affine from resample_spacing
        output_affine = np.diag([resample_spacing[0], resample_spacing[1], resample_spacing[2], 1.0])

        # Construct output filename
        image_filename = os.path.basename(test_dicts[i]['image'])
        if dataset == 'Example':
            # Example naming strategy if needed
            # Adjust if you have a specific naming format
            identifier = image_filename.replace('.nii.gz', '')
            output_filename = f'{identifier}_prediction.nii.gz'
        elif dataset == 'Original':
            identifier = image_filename.split('image')[1].split('.nii.gz')[0]
            output_filename = f'image{identifier}_prediction.nii.gz'

        # Save outputs
        nifti_output = nib.Nifti1Image(output, affine=output_affine)
        nifti_segmentation = nib.Nifti1Image(output_segmentation, affine=output_affine)
        nifti_truth = nib.Nifti1Image(truth, affine=output_affine)
        nifti_image = nib.Nifti1Image(image.squeeze().cpu().numpy(), affine=output_affine)

        # Save files in respective directories
        nib.save(nifti_output, str(output_dir / output_filename))
        nib.save(nifti_segmentation, str(segmentation_dir / f'segmentation_{output_filename}'))
        nib.save(nifti_truth, str(truth_dir / f'truth_{output_filename}'))
        nib.save(nifti_image, str(image_dir / f'image_{output_filename}'))

        print(f'Saved {output_filename} to organized directories')
