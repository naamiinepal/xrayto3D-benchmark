# Copyright (c) NAAMII, Nepal.
# For more information, visit https://www.naamii.org.np.
# Licensed under the GNU General Public License v3.0 (GPL-3.0).
# See https://www.gnu.org/licenses/gpl-3.0.html for details.


"""arch utils"""
from typing import Sequence


def calculate_1d_vec_channels(
    spatial_dims, image_size, encoder_strides: Sequence, encoder_last_channel
) -> int:
    """calculate size of the 1D Low-dim embedding for given model layers"""

    for stride in encoder_strides:
        image_size /= stride

    lat_encoder_1d_vecsize = ap_encoder_1d_vecsize = (
        image_size**spatial_dims
    ) * encoder_last_channel

    decoded_cube_channels = ap_encoder_1d_vecsize + lat_encoder_1d_vecsize

    return int(decoded_cube_channels)


if __name__ == "__main__":
    from monai.networks.layers.convutils import (calculate_out_shape,
                                                 same_padding)

    print(calculate_1d_vec_channels(2, 128, (2, 2, 2, 2), 32))
    strides = (2, 2, 2)
    ks = 3
    print(calculate_out_shape(128, ks, strides, same_padding(ks, 1)))
