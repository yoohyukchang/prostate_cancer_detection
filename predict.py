from model import DualNet_seperate, DualNet_seperate_load
from data_loader import CustomTransform
from monai.data import MetaTensor
from monai.transforms import Compose, LoadImage, ScaleIntensity, Resize, ResizeWithPadOrCrop, Spacing, \
    EnsureChannelFirst, ToTensor, EnsureType, InvertibleTransform
import torch
import glob
from sklearn.model_selection import train_test_split
import monai
import nibabel as nib
import numpy as np
import os
import warnings

if __name__ == '__main__':
    # Silence warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Select sample dataset
    dataset = 'Original'  # Options are 'Example' or 'Original'

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device {device}')

    # Create loader and transform specifications
    crop_size = crop_size = [128, 128, 64]
    resample_spacing = [1.5, 1.5, 1.5]

    # Define custom transforms
    img_transforms = [
        ScaleIntensity(),
        Spacing(pixdim=resample_spacing, mode='bilinear'),
        ResizeWithPadOrCrop(spatial_size=crop_size),
        ToTensor(),
        EnsureType()
    ]
    lbl_transforms = [
        ScaleIntensity(),
        Spacing(pixdim=resample_spacing, mode='nearest'),
        ResizeWithPadOrCrop(spatial_size=crop_size),
        ToTensor(),
        EnsureType()
    ]
    transform = CustomTransform(img_transforms, lbl_transforms)

    # Define image paths
    if dataset == 'Example':
        test_images = glob.glob(r'D:\Ryan\datasets\Prostate\Model Files\test\images\iRUF**_T2.nii', recursive=True)
        test_labels = glob.glob(r'D:\Ryan\datasets\Prostate\Model Files\test\labels\iRUF**_T2_label_prostate.nii', recursive=True)
    elif dataset == 'Original':
        test_images = glob.glob(r'D:\Ryan\datasets\Prostate\Model Files\images\image**.nii.gz', recursive=True)
        test_labels = glob.glob(r'D:\Ryan\datasets\Prostate\Model Files\labels\label**.nii.gz', recursive=True)
        _, test_images, _, test_labels = train_test_split(test_images, test_labels, test_size=0.2, random_state=42)
        _, test_images, _, test_labels = train_test_split(test_images, test_labels, test_size=0.5, random_state=42)

    # Create dictionaries
    test_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(test_images, test_labels)]

    # Create datasets
    test_dataset = monai.data.Dataset(data=test_dicts, transform=transform)

    # Create loaders
    test_loader = monai.data.DataLoader(test_dataset, batch_size=1)
    print('Loader created')

    # Load model
    model = DualNet_seperate_load(device=device, crop_patch_size=crop_size, out_channels=1)
    checkpoint = torch.load(
        r'C:\Users\rmcgove3\prostate_project\model_save\Prostate_seg_DualNet_seperate_sigmoid_bs_16_dropout_0.2_mul_batch_iter_1_patch_969664_paper_fold_5_epoch_47.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    print('Model loaded')

    # Prediction loop
    dices = []
    for i, batch in enumerate(test_loader):
        model.zero_grad()

        # Extract the original image volume information
        original_image = nib.load(test_dicts[i]['image'])
        original_affine = original_image.affine
        original_spacing = (abs(original_affine[0, 0]), abs(original_affine[1, 1]), abs(original_affine[2, 2]))
        original_size = original_image.shape
        flip_dimensions = []
        # for dim in range(3):
        #     if original_affine[dim][dim] < 0:
        #         flip_dimensions.append(dim + 2)

        # Properly orient input
        image = batch['image'].to(device)
        image = torch.flip(image, dims=flip_dimensions)

        # Perform prediction
        output = model(image).squeeze(0)

        # Configure the new image name
        image_filename = os.path.basename(test_dicts[i]['image'])
        if dataset == 'Example':
            identifier = image_filename.split('iRUF')[1].split('_T2')[0]
            output_filename = f'iRUF{identifier}_T2_prediction.nii'
        elif dataset == 'Original':
            identifier = image_filename.split('image')[1].split('.nii.gz')[0]
            output_filename = f'image{identifier}_prediction.nii.gz'

        # Apply reverse transform
        output = output.cpu().detach().squeeze().numpy()
        truth = batch['label'].detach().squeeze().numpy()

        output_segmentation = np.where(output > 0.5, 1, 0)

        output_affine = torch.tensor([
            [1.5, 0.0, 0.0, 0.0],
            [0.0, 1.5, 0.0, 0.0],
            [0.0, 0.0, 1.5, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        # Save the inverse-transformed output with the new filename
        image = image.cpu().detach().squeeze().numpy()
        nifti_output = nib.Nifti1Image(output, affine=output_affine)
        nifti_segmentation = nib.Nifti1Image(output_segmentation, affine=output_affine)
        nifti_truth = nib.Nifti1Image(truth, affine=output_affine)
        nifti_image = nib.Nifti1Image(image, affine=output_affine)
        nib.save(nifti_output, os.path.join(r'D:\Ryan\datasets\Prostate\Model Files\test\predicted_labels', output_filename))
        nib.save(nifti_segmentation, os.path.join(r'D:\Ryan\datasets\Prostate\Model Files\test\predicted_labels',
                                                  f'segmentation_{output_filename}'))
        nib.save(nifti_truth, os.path.join(r'D:\Ryan\datasets\Prostate\Model Files\test\predicted_labels', f'truth_{output_filename}))
        nib.save(nifti_image, os.path.join(r'D:\Ryan\datasets\Prostate\Model Files\test\predicted_labels',
                                                  f'image_{output_filename}'))

        print(f'Saved {output_filename}')
