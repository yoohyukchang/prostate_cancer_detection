# Code for create model structure


#%%
# Code for design model structure

from typing import Sequence, Tuple, Union

import numpy as np
import torch.nn as nn

from monai.networks.layers.factories import Act, Norm
from monai.utils import ensure_tuple_rep, look_up_option

from block_swinunetr import *
from block_unet import Unet_en, Unet_de, Unet_out, Dual_en

class DualNet_seperate(nn.Module):
    """
    # Dual endcoder to combine convolution (unet) and transformer (swinunetr)
    # Two branch operate (extract feature) seperately, after then combine each feature maps
    """

    def __init__(
        self,
        img_size: Union[Sequence[int], int],
        in_channels: int,
        out_channels: int,
        channels: Sequence[int] = (16, 32, 64),
        strides: Sequence[int]=(2, 2, 2),
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        bias: bool = True,
        adn_ordering: str = "NDA",
        depths: Sequence[int] = (2, 2, 2, 2),
        self_atts: Sequence[str] = ["Local", "Local", "Local", "Local"],
        num_heads: Sequence[int] = (3, 6, 12, 24),
        feature_size: int = 24,
        norm_name: Union[Tuple, str] = "instance",
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        dropout_path_rate: float = 0.0,
        drop_rate_conv: float = 0.0,
        normalize: bool = True,
        use_checkpoint: bool = False,
        spatial_dims: int = 3,
        downsample="merging",
        patch_size: Sequence[int] = (2, 2, 2),
        window_size: Sequence[int] = (7, 7, 7),
    ) -> None:
        """
        Args:
            img_size: dimension of input image.
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            drop_rate: dropout rate.

            channels: sequence of channels for unet. Top block first. The length of `channels` should be no less than 2.
            strides: sequence of convolution strides for unet. The length of `stride` should equal to `len(channels) - 1`.
            kernel_size: convolution kernel size, the value(s) should be odd. If sequence, its length should equal to dimensions. Defaults to 3.
            up_kernel_size: upsampling convolution kernel size, the value(s) should be odd. If sequence, its length should equal to dimensions. Defaults to 3.
            num_res_units: int = 0,
            act: Union[Tuple, str] = Act.PRELU,
            norm: Union[Tuple, str] = Norm.INSTANCE,
            dropout: float = 0.0,
            bias: bool = True,
            adn_ordering: str = "NDA",
            
            feature_size: dimension of network feature size.
            depths: number of layers in each stage.
            num_heads: number of attention heads.
            norm_name: feature normalization type and arguments.
            
            attn_drop_rate: attention dropout rate.
            dropout_path_rate: drop path rate.
            normalize: normalize output intermediate features in each stage.
            use_checkpoint: use gradient checkpointing for reduced memory usage.
            spatial_dims: number of spatial dims.
            downsample: module used for downsampling, available options are `"mergingv2"`, `"merging"` and a
                user-specified `nn.Module` following the API defined in :py:class:`monai.networks.nets.PatchMerging`.
                The default is currently `"merging"` (the original version defined in v0.9.0).
        """

        super().__init__()

        img_size = ensure_tuple_rep(img_size, spatial_dims) # ensure_tuple_rep(tup, dim) = Returns a copy of tup with dim values by either shortened or duplicated input
        patch_size = ensure_tuple_rep(patch_size, spatial_dims)
        window_size = ensure_tuple_rep(window_size, spatial_dims)

        if not (spatial_dims == 2 or spatial_dims == 3):
            raise ValueError("spatial dimension should be 2 or 3.")

        for m, p in zip(img_size, patch_size): # check whether img can be divided patch with patch_size
            for i in range(5):
                if m % np.power(p, i + 1) != 0:
                    raise ValueError("input image size (img_size) should be divisible by stage-wise image resolution.")

        if not (0 <= drop_rate <= 1):
            raise ValueError("dropout rate should be between 0 and 1.")

        if not (0 <= attn_drop_rate <= 1):
            raise ValueError("attention dropout rate should be between 0 and 1.")

        if not (0 <= dropout_path_rate <= 1):
            raise ValueError("drop path rate should be between 0 and 1.")

        if feature_size % 12 != 0:
            raise ValueError("feature_size should be divisible by 12.")

        if not len(depths)+1 == len(channels):
            raise ValueError("len(channels) and len(depths)+1 should be same")

        self.num_layers = len(channels)
        self.kernel_size = kernel_size

        self.normalize = normalize

        self.res_block = True

        self.swinViT = SwinTransformer_V2(
            in_chans=in_channels,
            embed_dim=feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=depths,
            num_heads=num_heads,
            self_atts=self_atts,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=dropout_path_rate,
            norm_layer=nn.InstanceNorm3d,
            use_checkpoint=use_checkpoint,
            spatial_dims=spatial_dims,
            downsample=look_up_option(downsample, MERGING_MODE) if isinstance(downsample, str) else downsample,
        )
        
        self.encoder0 = Unet_en(
                spatial_dims = spatial_dims,
                in_channels_layer = in_channels,
                out_channels_layer = channels[0],
                strides = strides[0],
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
        )
        
        self.encoders = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            encoder = Unet_en(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[i_layer],
                out_channels_layer = channels[i_layer+1],
                strides = strides[i_layer+1],
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
            )
            self.encoders.append(encoder)
        
        
        self.endecoder_1 = Unet_en(
            spatial_dims = spatial_dims,
            in_channels_layer = channels[-1],
            out_channels_layer = channels[-1]*2,
            strides = 1,
            kernel_size = kernel_size,
            num_res_units = num_res_units,
            act = act,
            norm = norm,
            dropout = drop_rate_conv,
            bias = bias,
            adn_ordering = adn_ordering,
        )

        self.dualencoders = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dualencoder = Dual_en(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[i_layer]+feature_size*(2**i_layer),
                out_channels_layer = channels[i_layer],
                strides = 1,
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
            )
            self.dualencoders.append(dualencoder)

        self.endecoder_2 = Unet_de(
            spatial_dims = spatial_dims,
            in_channels_layer = channels[-1]*2,
            out_channels_layer = channels[-1],
            strides = 1,
            up_kernel_size = up_kernel_size,
            kernel_size = kernel_size,
            num_res_units = num_res_units,
            act = act,
            norm = norm,
            dropout = drop_rate_conv,
            bias = bias,
            adn_ordering = adn_ordering,
            is_top = False,
        )

        self.decoders = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            decoder = Unet_de(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[-(i_layer+1)],
                out_channels_layer = channels[-(i_layer+2)],
                strides = strides[-(i_layer+1)],
                up_kernel_size = up_kernel_size,
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
                is_top = False,
            )
            self.decoders.append(decoder)

        self.out = Unet_out(
                spatial_dims = spatial_dims,
                in_channels_layer = channels[0],
                out_channels_layer = out_channels,
                strides = strides[0],
                up_kernel_size = up_kernel_size,
                kernel_size = kernel_size,
                num_res_units = num_res_units,
                act = act,
                norm = norm,
                dropout = drop_rate_conv,
                bias = bias,
                adn_ordering = adn_ordering,
                is_top = True,
            )

    def forward(self, x_in):
        '''
        hidden_states_out_1=torch.Size([2, 12, 32, 64, 64])
        hidden_states_out_2=torch.Size([2, 24, 16, 32, 32])
        hidden_states_out_3=torch.Size([2, 48, 8, 16, 16])
        hidden_states_out_4=torch.Size([2, 96, 4, 8, 8])

        enc_1=torch.Size([2, 16, 32, 64, 64])
        enc_2=torch.Size([2, 32, 16, 32, 32])
        enc_3=torch.Size([2, 64, 8, 16, 16])
        enc_4=torch.Size([2, 128, 4, 8, 8])

        ende_1=torch.Size([2, 256, 4, 8, 8])

        dual_enc_1=torch.Size([2, 16, 32, 64, 64])/encs=torch.Size([2, 16, 32, 64, 64])/hidden_states_out=torch.Size([2, 12, 32, 64, 64])
        dual_enc_2=torch.Size([2, 32, 16, 32, 32])/encs=torch.Size([2, 32, 16, 32, 32])/hidden_states_out=torch.Size([2, 24, 16, 32, 32])
        dual_enc_3=torch.Size([2, 64, 8, 16, 16])/encs=torch.Size([2, 64, 8, 16, 16])/hidden_states_out=torch.Size([2, 48, 8, 16, 16])
        dual_enc_4=torch.Size([2, 128, 4, 8, 8])/encs=torch.Size([2, 128, 4, 8, 8])/hidden_states_out=torch.Size([2, 96, 4, 8, 8])
        
        dec_1=torch.Size([2, 128, 4, 8, 8])
        dec_2=torch.Size([2, 64, 8, 16, 16])
        dec_3=torch.Size([2, 32, 16, 32, 32])
        dec_4=torch.Size([2, 16, 32, 64, 64])

        out=torch.Size([2, 1, 64, 128, 128])
        '''

        # encoder of transformer branch (1 layer ~ last layer)
        hidden_states_out = self.swinViT(x_in, self.normalize)

        # encoder of convolution branch (1 layer)
        encs = []
        enc = self.encoder0(x_in) # encs[0] = torch.Size([2, 16, 32, 64, 64])
        encs.append(enc)

        # encoder of convolution branch (2 layer ~ last layer)
        for i_layer in range(self.num_layers - 1):
            enc = self.encoders[i_layer](encs[i_layer]) # encs[1] = torch.Size([2, 32, 16, 32, 32]) / encs[2] = torch.Size([2, 64, 8, 16, 16]) / encs[3] = torch.Size([2, 128, 4, 8, 8])
            encs.append(enc)

        # bottom (endocer - decoder)
        ende_1 = self.endecoder_1(encs[-1]) # torch.Size([2, 256, 4, 8, 8])

        # combine convolution branch and transformer branch
        dual_encs = []
        for i_layer in range(self.num_layers):
            dual_enc = self.dualencoders[i_layer](encs[i_layer], hidden_states_out[i_layer])
            dual_encs.append(dual_enc)

        # decoder last layer
        decs = []
        dec = self.endecoder_2(ende_1, dual_encs[-1]) # torch.Size([2, 256, 4, 8, 8])
        decs.append(dec)

        # decoder (last-1) layer ~ first layer
        for i_layer in range(self.num_layers - 1):
            dec = self.decoders[i_layer](decs[i_layer], dual_encs[-(i_layer+2)])
            decs.append(dec)

        out = self.out(decs[-1])

        return out

def DualNet_seperate_load(device, crop_patch_size, out_channels):

    # Parameter for model
    in_channels = 1
    drop_rate = 0.2
    attn_drop_rate = 0.2
    dropout_path_rate = 0.2
    depths = (2,2,2) 
    self_atts=["Local", "Local", "Local"]
    patch_size = (2,2,2)
    window_size = (4,4,4)
    feature_size = 12 
    use_checkpoint = True
    drop_rate_conv = 0.2
    spatial_dims=3
    channels=(16, 32, 64, 128) 
    strides=(2, 2, 2, 2)
    num_res_units=2

    # model structure
    coarse_model = DualNet_seperate(
        img_size=crop_patch_size,
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
        adn_ordering = "NAD",
    ).to(device)

    return coarse_model

