import monai.transforms as mt


def general_transform(crop_size, resample_spacing, input=True, axcodes="RAS"):

    if input:
        transform = mt.Compose(
            [
                mt.LoadImaged(keys=["image", "label"]),
                mt.EnsureChannelFirstd(keys=["image", "label"]),
                mt.Orientationd(keys=["image", "label"], axcodes=axcodes),
                mt.Spacingd(keys=["image", "label"], 
                            pixdim=(resample_spacing[0], resample_spacing[1], resample_spacing[2]),
                            mode=("nearest", "nearest")),
                mt.ScaleIntensityRangePercentilesd(keys="image", lower=10, upper=90, b_min=0, b_max=1, relative=True,
                                                channel_wise=True),
                mt.BorderPadd(keys=["image", "label"], spatial_border=(crop_size[0], crop_size[1], crop_size[2]),
                            mode="constant"),
                mt.CenterSpatialCropd(keys=["image", "label"], roi_size=crop_size),
                mt.ToTensord(keys=["image", "label"])
            ]
        )
    else:
        transform = mt.Compose(
            [
                mt.EnsureChannelFirstd(keys=["pred"]),
                mt.Orientationd(keys=["pred"], axcodes=axcodes),
                mt.Spacingd(keys=["pred"], 
                            pixdim=(resample_spacing[0], resample_spacing[1], resample_spacing[2]),
                            mode=("pred")),
                mt.BorderPadd(keys=["pred"], spatial_border=(crop_size[0], crop_size[1], crop_size[2]),
                            mode="constant"),
                mt.CenterSpatialCropd(keys=["pred"], roi_size=crop_size),
                mt.ToTensord(keys=["pred"])
            ]
        )

    return transform
