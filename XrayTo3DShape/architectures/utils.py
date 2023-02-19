from typing import Sequence
def calculate_1d_vec_channels(image_size,encoder_strides:Sequence,encoder_last_channel)->int:
    """calculate size of the 1D Low-dim embedding for given model layers"""
    iw,ih = image_size,image_size # width , height
    
    for stride in encoder_strides:
        iw /= stride
        ih /= stride
    lat_encoder_1d_vecsize = ap_encoder_1d_vecsize = ih * iw * encoder_last_channel

    decoded_cube_channels = ap_encoder_1d_vecsize  + lat_encoder_1d_vecsize

    return int(decoded_cube_channels)