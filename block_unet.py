# Code for create unet model structure

#%%

from typing import Optional, Sequence, Tuple, Union
#import warnings

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
#from monai.networks.layers.simplelayers import SkipConnection
#from monai.utils import alias, deprecated_arg, export


class Unet_en(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        if self.num_res_units > 0:
            self.down = ResidualUnit(
                self.dimensions,
                self.in_channels,
                self.out_channels,
                strides=self.strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
        else:
            self.down = Convolution(
                self.dimensions,
                self.in_channels,
                self.out_channels,
                strides=self.strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )

    def forward(self, enc):

        out = self.down(enc)

        return out


class Unet_de(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        is_top: bool = False,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_top = is_top

        self.conv_up = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=self.strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=self.is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        self.conv = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        self.ru = ResidualUnit(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=1,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=self.is_top,
            adn_ordering=self.adn_ordering,
        )

    def forward(self, dec, enc):

        up = self.conv_up(dec)
        skip = torch.cat((up, enc), dim=1)
        if self.num_res_units > 0:
            out = self.ru(skip)
        else:
            out = self.conv(skip)

        return out

class Unet_out(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        is_top: bool = False,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_top = is_top

        self.conv_up = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=self.strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=self.is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        self.conv = Convolution(
            self.dimensions,
            self.out_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        self.ru = ResidualUnit(
            self.dimensions,
            self.out_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=1,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=self.is_top,
            adn_ordering=self.adn_ordering,
        )

    def forward(self, dec):

        up = self.conv_up(dec)

        if self.num_res_units > 0:
            out = self.ru(up)
        else:
            out = self.conv(up)

        return out

class Dual_en(nn.Module): # fuse the feature map of two branchs on each layers

    def __init__(
        self,
        spatial_dims: int,
        in_channels_layer: int,
        out_channels_layer: int,
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        is_top: bool = False,
    ) -> None:

        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels_layer
        self.out_channels = out_channels_layer
        self.strides = strides
        self.kernel_size = kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        self.is_top = is_top

        self.conv = Convolution(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )

        self.ru = ResidualUnit(
            self.dimensions,
            self.in_channels,
            self.out_channels,
            strides=1,
            kernel_size=self.kernel_size,
            subunits=1,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            last_conv_only=self.is_top,
            adn_ordering=self.adn_ordering,
        )

    def forward(self, br_1, br_2):

        skip = torch.cat((br_1, br_2), dim=1)
        if self.num_res_units > 0:
            out = self.ru(skip)
        else:
            out = self.conv(skip)

        return out





#%%

'''

class UNet(nn.Module):


    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 3,
        num_res_units: int = 0,
        act: Union[Tuple, str] = Act.PRELU,
        norm: Union[Tuple, str] = Norm.INSTANCE,
        dropout: float = 0.0,
        bias: bool = True,
        adn_ordering: str = "NDA",
        dimensions: Optional[int] = None,
    ) -> None:

        super().__init__()

        if len(channels) < 2:
            raise ValueError("the length of `channels` should be no less than 2.")
        delta = len(strides) - (len(channels) - 1)
        if delta < 0:
            raise ValueError("the length of `strides` should equal to `len(channels) - 1`.")
        if delta > 0:
            warnings.warn(f"`len(strides) > len(channels) - 1`, the last {delta} values of strides will not be used.")
        if dimensions is not None:
            spatial_dims = dimensions
        if isinstance(kernel_size, Sequence):
            if len(kernel_size) != spatial_dims:
                raise ValueError("the length of `kernel_size` should equal to `dimensions`.")
        if isinstance(up_kernel_size, Sequence):
            if len(up_kernel_size) != spatial_dims:
                raise ValueError("the length of `up_kernel_size` should equal to `dimensions`.")


        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering

        def _create_block(
            inc: int, outc: int, channels: Sequence[int], strides: Sequence[int], is_top: bool
        ) -> nn.Module:
            """
            Builds the UNet structure from the bottom up by recursing down to the bottom block, then creating sequential
            blocks containing the downsample path, a skip connection around the previous block, and the upsample path.

            Args:
                inc: number of input channels.
                outc: number of output channels.
                channels: sequence of channels. Top block first.
                strides: convolution stride.
                is_top: True if this is the top block.
            """
            c = channels[0]
            s = strides[0]

            subblock: nn.Module

            if len(channels) > 2:
                subblock = _create_block(c, c, channels[1:], strides[1:], False)  # continue recursion down
                upc = c * 2
            else:
                # the next layer is the bottom so stop recursion, create the bottom layer as the sublock for this layer
                subblock = self._get_bottom_layer(c, channels[1])
                upc = c + channels[1]

            down = self._get_down_layer(inc, c, s, is_top)  # create layer in downsampling path
            up = self._get_up_layer(upc, outc, s, is_top)  # create layer in upsampling path

            return self._get_connection_block(down, up, subblock)

        self.model = _create_block(in_channels, out_channels, self.channels, self.strides, True)

    def _get_connection_block(self, down_path: nn.Module, up_path: nn.Module, subblock: nn.Module) -> nn.Module:
        """
        Returns the block object defining a layer of the UNet structure including the implementation of the skip
        between encoding (down) and decoding (up) sides of the network.

        Args:
            down_path: encoding half of the layer
            up_path: decoding half of the layer
            subblock: block defining the next layer in the network.
        Returns: block for this layer: `nn.Sequential(down_path, SkipConnection(subblock), up_path)`
        """
        return nn.Sequential(down_path, SkipConnection(subblock), up_path)

    def _get_down_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the encoding (down) part of a layer of the network. This typically will downsample data at some point
        in its structure. Its output is used as input to the next layer down and is concatenated with output from the
        next layer to form the input for the decode (up) part of the layer.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        #mod: nn.Module
        if self.num_res_units > 0:

            mod = ResidualUnit(
                self.dimensions,
                in_channels,
                out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                adn_ordering=self.adn_ordering,
            )
            return mod
        mod = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering,
        )
        return mod

    def _get_bottom_layer(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Returns the bottom or bottleneck layer at the bottom of the network linking encode to decode halves.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
        """
        return self._get_down_layer(in_channels, out_channels, 1, False)

    def _get_up_layer(self, in_channels: int, out_channels: int, strides: int, is_top: bool) -> nn.Module:
        """
        Returns the decoding (up) part of a layer of the network. This typically will upsample data at some point
        in its structure. Its output is used as input to the next layer up.

        Args:
            in_channels: number of input channels.
            out_channels: number of output channels.
            strides: convolution stride.
            is_top: True if this is the top block.
        """
        conv: Union[Convolution, nn.Sequential]

        conv = Convolution(
            self.dimensions,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            conv_only=is_top and self.num_res_units == 0,
            is_transposed=True,
            adn_ordering=self.adn_ordering,
        )

        if self.num_res_units > 0:
            ru = ResidualUnit(
                self.dimensions,
                out_channels,
                out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                last_conv_only=is_top,
                adn_ordering=self.adn_ordering,
            )
            conv = nn.Sequential(conv, ru)

        return conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return x

'''
