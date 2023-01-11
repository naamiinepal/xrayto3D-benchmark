if __name__ == "__main__":
    from XrayTo3DShape.architectures.chen_model import OneDConcatModel
    import torch
    
    ap_img = torch.zeros((1, 1, 64, 64))
    lat_img = torch.zeros((1, 1, 64, 64))
    config = {"input_image_size":[64,64],
        "encoder": {'in_channels':[1,8,16], 'out_channels':[8,16,32],'strides':[2,2,2]},
        "decoder": {'in_channels':[4096,1024,512,8,4,4], 'out_channels':[1024,512,8,4,4,1],'strides':[2,2,2,2,2,2]},
        "kernel_size": 3,
        "act": "RELU",
        "norm": "BATCH",
        "dropout": 0.0,
    }
    model = OneDConcatModel(config)
    print(model)
    pred = model(ap_img,lat_img)
    print(pred.shape)
    print(model._calculate_1d_vec_channels())