import torch
from torch.nn import Sigmoid
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import monai.data
from monai.losses import DiceLoss, DiceCELoss
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImage, ScaleIntensity, Resize, ResizeWithPadOrCrop, EnsureChannelFirst, \
    Spacing, ToTensor, EnsureType
from sklearn.model_selection import train_test_split
import nibabel as nib

from data_loader import general_transform
from model import DualNet_seperate

import warnings


def main():
    # Silent warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on torch.device -', device)

    # Parameter about dataset organization
    oar_type = 'Legion'  # 'Prostate', 'Legion'

    # Parameter about preprocessing
    if oar_type == 'Prostate':
        crop_size = [128, 128, 64]
        best_metric = 0.4
        resample_spacing = [1.5, 1.5, 1.5]
    elif oar_type == 'Legion':
        crop_size = [128, 128, 64]
        best_metric = 0.4
        resample_spacing = [0.6640625, 0.6640625, 1.5]
    else:
        ValueError(f"Invalid value: {oar_type}. Must be 'Prostate' or 'Legion'.")

    # Parameter about train
    batch_size = 16
    val_batch_size = 1
    epochs = 150
    lr = 1e-3  # 5*1e-3
    weight_decay = 1e-2

    val_check_epoch_intv = 1

    in_channels = 1
    out_channels = 1

    drop_rate = 0.2
    attn_drop_rate = 0.2
    dropout_path_rate = 0.2
    depths = (2, 2, 2)  # (2,2,2)
    self_atts = ["Local", "Local", "Local"]  # ["Local", "Local", "Local"]
    patch_size = (2, 2, 2)
    window_size = (4, 4, 4)
    feature_size = 12  # 12
    use_checkpoint = True

    drop_rate_conv = 0.2
    spatial_dims = 3
    channels = (16, 32, 64, 128)  # (16, 32, 64, 128)
    strides = (2, 2, 2, 2)  # (2, 2, 2, 2)
    num_res_units = 2

    # Set path for saving model
    model_base_path = 'C:\\Users\\rmcgove3\\prostate_project\\model_save\\'
    model_name = f'oar_type_model'

    # Set path for saving cache data
    cache_path = 'C:\\Users\\rmcgove3\\prostate_project\\cached_data\\' + model_name
    if not os.path.exists(cache_path):
        os.makedirs(cache_path)

    # Set tensorboard
    writer_dir = os.path.join(model_base_path, 'logs', model_name)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=writer_dir)

    # Set datasets
    image_paths = glob.glob(r'D:\Ryan\datasets\Prostate\Model Files\cropped_images\cropped_image**.nii.gz',
                            recursive=True)
    label_paths = glob.glob(r'D:\Ryan\datasets\Prostate\Model Files\cropped_legions\cropped_legion**.nii.gz',
                            recursive=True)

    # Split data
    train_images, temp_images, train_labels, temp_labels = train_test_split(image_paths, label_paths, test_size=0.2,
                                                                            random_state=42)
    val_images, test_images, val_labels, test_labels = train_test_split(temp_images, temp_labels, test_size=0.5,
                                                                        random_state=42)

    # Create dictionaries
    train_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(train_images, train_labels)]
    val_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(val_images, val_labels)]
    test_dicts = [{'image': img, 'label': lbl} for img, lbl in zip(test_images, test_labels)]

    # Define transforms
    data_transform = general_transform(crop_size=crop_size, resample_spacing=resample_spacing)

    # Create datasets
    train_dataset = monai.data.Dataset(data=train_dicts, transform=data_transform)
    val_dataset = monai.data.Dataset(data=val_dicts, transform=data_transform)
    test_dataset = monai.data.Dataset(data=test_dicts, transform=data_transform)

    # Create loaders
    train_loader = monai.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_dataset, batch_size=val_batch_size)
    test_loader = monai.data.DataLoader(test_dataset, batch_size=1)

    # Create model
    model = DualNet_seperate(
        img_size=crop_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depths=depths,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        patch_size=patch_size,
        window_size=window_size,
        drop_rate_conv=drop_rate_conv,
        spatial_dims=spatial_dims,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        self_atts=self_atts,
        adn_ordering="NAD",
    ).to(device)
    model.load_state_dict(torch.load(r'C:\Users\rmcgove3\prostate_project\model_save\oar_type_model_epoch_19.pt'))

    # Set optimizer and loss function
    torch.backends.cudnn.benchmark = True
    loss_func = DiceCELoss(sigmoid=True, lambda_dice=0.7, lambda_ce=0.3, batch=True)
    dice_func = DiceLoss()

    post_sigmoid = mt.Activations(sigmoid=True)
    post_pred = mt.AsDiscrete(argmax=False)
    post_label = mt.AsDiscrete(argmax=False)

    # Set up required classes
    s = Sigmoid()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train model
    epoch_result_total = np.zeros((1, 4))
    epoch_result = np.zeros((1, 4))

    for epoch in range(20, epochs):
        print("-" * 20)
        print('Epoch: {} / {}'.format(epoch+1, epochs))

        # Train
        model.train()
        tr_loss = 0
        tr_dice = 0
        tr_check_no = 0
        optimizer.zero_grad()  # Sets the gradients of all optimized torch.Tensor to zero

        for i, batch in enumerate(train_loader):
            # Separate and move data to device
            inputs, labels = batch['image'].to(device), batch['label'].to(device)  # data to device

            # Run model
            outputs = model(inputs)
            outputs = s(outputs)
            if i == 1:
                affine = np.eye(4)
                affine[3][3] = 2.5
                image = inputs[0, 0].cpu().detach().numpy()
                label = labels[0, 0].cpu().detach().numpy()
                output = outputs[0, 0].cpu().detach().numpy()
                nifti_image = nib.Nifti1Image(image, affine)
                nifti_label = nib.Nifti1Image(label, affine)
                nifti_output = nib.Nifti1Image(output, affine)
                nib.save(nifti_image, 'image.nii.gz')
                nib.save(nifti_label, 'label.nii.gz')
                nib.save(nifti_output, 'output.nii.gz')
                print('Saved')

            # Compute loss
            loss = loss_func(outputs, labels)

            # Perform backpropagation
            loss.backward()
            tr_loss += loss.item()

            # Update weights
            optimizer.step()
            optimizer.zero_grad()

            # Check dice score of train dataset
            tr_check_no += 1
            outputs_detached = outputs.detach()
            labels_detached = labels.detach()
            dice = dice_func(outputs_detached, labels_detached)
            dice = 1 - dice.item()
            tr_dice += dice
            print('>> Batch: {} / {}, loss: {:.4f}, dice: {:.4f}'.format(i+1, len(train_loader), loss.item(), dice)) 
        
        tr_loss_mean = tr_loss/len(train_loader)
        tr_dice_mean = tr_dice/tr_check_no
        epoch_result[0, 0] = tr_loss_mean
        epoch_result[0, 1] = tr_dice_mean

        model.eval()
        total_val_loss = 0
        total_val_dice = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = s(model(inputs))

                # compute loss
                loss = loss_func(outputs, labels)
                val_loss = loss.item()
                total_val_loss += val_loss

                # compute dice score
                val_dice = dice_func(outputs, labels)
                val_dice = 1 - val_dice.item()
                total_val_dice += val_dice

        val_loss_mean = total_val_loss / len(val_loader)
        val_dice_mean = total_val_dice / len(val_loader)
        epoch_result[0, 2] = val_loss_mean
        epoch_result[0, 3] = val_dice_mean

        print('Average metric = tr_loss: {:.4f}, tr_dice:{:.4f}, val_loss:{:.4f}, val_dice:{:.4f}'.format(0, 0, val_loss_mean, val_dice_mean))

        epoch_result_total = np.vstack((epoch_result_total, epoch_result))

        # Save best model
        if val_dice_mean > best_metric:
            best_metric = val_dice_mean

            tr_best_loss = tr_loss_mean
            tr_best_dice = tr_dice_mean
            val_best_loss = val_loss_mean

            best_dir = '\\' + model_name + '_epoch_{}.pt'.format(epoch+1)

            torch.save(model.state_dict(), model_base_path + best_dir)
            print('Save model!!')

        # write to summary
        loss_dict = {'train': tr_loss_mean, 'val': val_loss_mean}
        dice_dict = {'train': tr_dice_mean, 'val': val_dice_mean}
        writer.add_scalars('Loss', loss_dict, epoch+1)
        writer.add_scalars('Dice', dice_dict, epoch+1)

        if scheduler is not None:
            scheduler.step()

    epoch_result_total = np.delete(epoch_result_total, (0), axis=0)

    print("-" * 20)
    print(best_dir)
    print("Train is Finished!")
    print('Best metric = tr_loss: {:.4f}, tr_dice:{:.4f}, val_loss:{:.4f}, val_dice: {:.4f}'.format(tr_best_loss, tr_best_dice, val_best_loss, best_metric))


    #### Test model ####
    model = DualNet_seperate(
        img_size=crop_size,
        in_channels=in_channels,
        out_channels=out_channels,
        depths=depths,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
        patch_size=patch_size,
        window_size=window_size,
        drop_rate_conv=drop_rate_conv,
        spatial_dims=spatial_dims,
        channels=channels,
        strides=strides,
        num_res_units=num_res_units,
        self_atts=self_atts,
        adn_ordering="NAD",
    ).to(device)

    model.load_state_dict(torch.load(model_base_path + best_dir))
    model.to(device)
    model.eval()

    test_result_each = []
    with torch.no_grad():
        test_dice = 0
        for i, batch in enumerate(test_loader):

            inputs, labels = batch['image'].to(device), batch['label'].to(device) # data to device

            outputs = sliding_window_inference(inputs, crop_size, len(test_loader), model, overlap=0.7)

            # compute dice score
            post_outputs = [post_pred(post_sigmoid(x)) for x in monai.data.decollate_batch(outputs)]
            post_labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
            dice_func(post_outputs, post_labels)
            dice = dice_func.aggregate().item()
            test_dice += dice
            dice_func.reset()
            test_result_each.append(dice)
        test_dice_mean = test_dice/len(test_loader)

    print("-" * 20)
    print("Test is Finished!")
    print('Test result = test_dice_1:{:.4f}'.format(test_dice_mean))



    #### Draw learning curve ####
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_result_total[:,0], label="tr_loss")
    plt.plot(epoch_result_total[:,2], label="val_loss")
    plt.subplot(1, 2, 2)
    plt.title("Epoch Average Dice")
    plt.xlabel("epoch")
    plt.ylabel("Dice")
    plt.plot(epoch_result_total[:,1], label="tr_dice")
    plt.plot(epoch_result_total[:,3], label="val_dice")
    plt.show()


if __name__ == '__main__':
    main()

