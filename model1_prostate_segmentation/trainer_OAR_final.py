from pathlib import Path
import torch
from torch.nn import Sigmoid
import os
import numpy as np
import matplotlib.pyplot as plt

import monai.data
from monai.losses import DiceLoss, DiceCELoss
import monai.transforms as mt
from monai.inferers import sliding_window_inference
from sklearn.model_selection import train_test_split
import nibabel as nib
import warnings

from data_loader import general_transform
from model import DualNet_seperate

def main():
    # Silent warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('training on torch.device -', device)

    # Choose OAR type
    # 'Prostate' or 'Legion'
    oar_type = 'Prostate'  

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
        raise ValueError(f"Invalid value: {oar_type}. Must be 'Prostate' or 'Legion'.")

    # Training parameters
    batch_size = 16
    val_batch_size = 1
    epochs = 150
    lr = 1e-3
    weight_decay = 1e-2

    in_channels = 1
    out_channels = 1

    drop_rate = 0.2
    attn_drop_rate = 0.2
    dropout_path_rate = 0.2
    depths = (2, 2, 2)
    self_atts = ["Local", "Local", "Local"]
    patch_size = (2, 2, 2)
    window_size = (4, 4, 4)
    feature_size = 12
    use_checkpoint = True
    drop_rate_conv = 0.2
    spatial_dims = 3
    channels = (16, 32, 64, 128)
    strides = (2, 2, 2, 2)
    num_res_units = 2

    # Paths
    current_path = Path(__file__).parent
    root_path = current_path.parent
    data_path = root_path / 'data'

    # Depending on oar_type, choose label directory
    if oar_type == 'Prostate':
        label_dir = data_path / 'Train' / 'Prostates'
    elif oar_type == 'Legion':
        label_dir = data_path / 'Train' / 'Legions'

    image_dir = data_path / 'Train' / 'MRIs'

    # Get image and label files
    image_paths = sorted(list(image_dir.glob('*.nii.gz')))
    label_paths = sorted(list(label_dir.glob('*.nii.gz')))

    # Split data
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        image_paths, label_paths, test_size=0.2, random_state=42
    )
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images, temp_labels, test_size=0.5, random_state=42
    )

    # Create dictionaries
    train_dicts = [{'image': str(img), 'label': str(lbl)} for img, lbl in zip(train_images, train_labels)]
    val_dicts = [{'image': str(img), 'label': str(lbl)} for img, lbl in zip(val_images, val_labels)]
    test_dicts = [{'image': str(img), 'label': str(lbl)} for img, lbl in zip(test_images, test_labels)]

    # Define transforms
    data_transform = general_transform(crop_size=crop_size, resample_spacing=resample_spacing)

    # Create datasets and loaders
    train_dataset = monai.data.Dataset(data=train_dicts, transform=data_transform)
    val_dataset = monai.data.Dataset(data=val_dicts, transform=data_transform)
    test_dataset = monai.data.Dataset(data=test_dicts, transform=data_transform)

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

    # If you had a previous model checkpoint to load, do it here (commented out):
    # model_path = current_path / 'oar_type_model_epoch_19.pt'
    # model.load_state_dict(torch.load(model_path, map_location=device))

    torch.backends.cudnn.benchmark = True
    loss_func = DiceCELoss(sigmoid=True, lambda_dice=0.7, lambda_ce=0.3, batch=True)
    dice_func = DiceLoss()

    post_sigmoid = mt.Activations(sigmoid=True)
    post_pred = mt.AsDiscrete(argmax=False)
    post_label = mt.AsDiscrete(argmax=False)

    s = Sigmoid()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Setup tensorboard writer
    model_name = f'{oar_type.lower()}_type_model'
    writer_dir = current_path / 'logs' / model_name
    writer_dir.mkdir(parents=True, exist_ok=True)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=str(writer_dir))

    epoch_result_total = np.zeros((1, 4))
    epoch_result = np.zeros((1, 4))
    best_dir = None

    for epoch in range(epochs):
        print("-" * 20)
        print('Epoch: {} / {}'.format(epoch+1, epochs))

        # Train
        model.train()
        tr_loss = 0
        tr_dice = 0
        tr_check_no = 0
        optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            inputs, labels = batch['image'].to(device), batch['label'].to(device)

            outputs = model(inputs)
            outputs = s(outputs)

            # Compute loss
            loss = loss_func(outputs, labels)
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

            # Compute dice score
            dice_val = dice_func(outputs.detach(), labels.detach())
            dice_val = 1 - dice_val.item()
            tr_dice += dice_val
            tr_check_no += 1

            print('>> Batch: {} / {}, loss: {:.4f}, dice: {:.4f}'.format(i+1, len(train_loader), loss.item(), dice_val))

        tr_loss_mean = tr_loss / len(train_loader)
        tr_dice_mean = tr_dice / tr_check_no
        epoch_result[0, 0] = tr_loss_mean
        epoch_result[0, 1] = tr_dice_mean

        # Validation
        model.eval()
        total_val_loss = 0
        total_val_dice = 0
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = s(model(inputs))

                val_loss = loss_func(outputs, labels).item()
                total_val_loss += val_loss

                val_dice = dice_func(outputs, labels)
                val_dice = 1 - val_dice.item()
                total_val_dice += val_dice

        val_loss_mean = total_val_loss / len(val_loader)
        val_dice_mean = total_val_dice / len(val_loader)
        epoch_result[0, 2] = val_loss_mean
        epoch_result[0, 3] = val_dice_mean

        print('Average metric = tr_loss: {:.4f}, tr_dice:{:.4f}, val_loss:{:.4f}, val_dice:{:.4f}'.format(
            tr_loss_mean, tr_dice_mean, val_loss_mean, val_dice_mean
        ))

        epoch_result_total = np.vstack((epoch_result_total, epoch_result))

        # Save best model
        if val_dice_mean > best_metric:
            best_metric = val_dice_mean
            tr_best_loss = tr_loss_mean
            tr_best_dice = tr_dice_mean
            val_best_loss = val_loss_mean
            best_dir = current_path / f"{model_name}_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), best_dir)
            print('Saved best model:', best_dir)

        # Write to tensorboard
        loss_dict = {'train': tr_loss_mean, 'val': val_loss_mean}
        dice_dict = {'train': tr_dice_mean, 'val': val_dice_mean}
        writer.add_scalars('Loss', loss_dict, epoch+1)
        writer.add_scalars('Dice', dice_dict, epoch+1)

        if scheduler is not None:
            scheduler.step()

    epoch_result_total = np.delete(epoch_result_total, (0), axis=0)

    print("-" * 20)
    if best_dir is not None:
        print("Best model saved at:", best_dir)
        print("Train is Finished!")
        print('Best metric results:')
        print('Best tr_loss: {:.4f}, tr_dice:{:.4f}, val_loss:{:.4f}, val_dice: {:.4f}'.format(
            tr_best_loss, tr_best_dice, val_best_loss, best_metric))
    else:
        print("No improvement in validation dice score, no best model saved.")

    #### Test model ####
    if best_dir is not None:
        model.load_state_dict(torch.load(best_dir, map_location=device))
        model.eval()

        test_result_each = []
        with torch.no_grad():
            test_dice = 0
            for i, batch in enumerate(test_loader):
                inputs, labels = batch['image'].to(device), batch['label'].to(device)
                outputs = sliding_window_inference(inputs, crop_size, len(test_loader), model, overlap=0.7)

                post_outputs = [post_pred(post_sigmoid(x)) for x in monai.data.decollate_batch(outputs)]
                post_labels = [post_label(x) for x in monai.data.decollate_batch(labels)]
                dice_func(post_outputs, post_labels)
                dice_score = dice_func.aggregate().item()
                test_dice += dice_score
                dice_func.reset()
                test_result_each.append(dice_score)
            test_dice_mean = test_dice/len(test_loader)

        print("-" * 20)
        print("Test is Finished!")
        print('Test result dice: {:.4f}'.format(test_dice_mean))

    #### Draw learning curve ####
    plt.figure("train", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Epoch Average Loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.plot(epoch_result_total[:, 0], label="tr_loss")
    plt.plot(epoch_result_total[:, 2], label="val_loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Epoch Average Dice")
    plt.xlabel("epoch")
    plt.ylabel("Dice")
    plt.plot(epoch_result_total[:, 1], label="tr_dice")
    plt.plot(epoch_result_total[:, 3], label="val_dice")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
