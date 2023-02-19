if __name__ == "__main__":
    from XrayTo3DShape.architectures import *
    import torch
    from torchview import draw_graph
    from pytorch_lightning.utilities.model_summary.model_summary import ModelSummary

    ap_img = torch.zeros((1, 1, 64, 64))
    lat_img = torch.zeros((1, 1, 64, 64))
    config = {"input_image_size":[64,64],
        "encoder": {'in_channels':[1,8,16], 'out_channels':[8,16,32],'strides':[2,2,2]},
        "decoder": {'in_channels':[4096,1024,512,8,4,4], 'out_channels':[1024,512,8,4,4,1],'strides':[2,2,2,2,2,2]},
        "kernel_size": 3,
        "act": "RELU",
        "norm": "BATCH",
        "dropout": 0.0,
        'bias': False
    }
    model = OneDConcatModel(config)
    pred = model(ap_img,lat_img)
    print(model._calculate_1d_vec_channels())
    model_graph = draw_graph(model,input_data=[ap_img,lat_img],save_graph=True,filename='docs/arch_viz/onedconcat_modelgraph')


    config_bayat = {
        "input_image_size": [64, 64],
        "encoder": {
            "in_channels": [1, 16, 32, 32, 32, 32],
            "out_channels": [16, 32, 32, 32, 32, 32],
            "strides": [2, 2, 1, 1, 1, 1],
            "kernel_size": 7,
        },
        "ap_expansion": {
            "in_channels": [32, 32, 32, 32],
            "out_channels": [32, 32, 32, 32],
            "strides": ((2, 1, 1),) * 4,
            "kernel_size": 3,
        },
        "lat_expansion": {
            "in_channels": [32, 32, 32, 32],
            "out_channels": [32, 32, 32, 32],
            "strides": ((1, 1, 2),) * 4,
            "kernel_size": 3,
        },
        "decoder": {
            "in_channels": [64, 64, 64, 64, 64, 32, 16],
            "out_channels": [64, 64, 64, 64, 32, 16, 1],
            "strides": (1, 1, 1, 1, 2, 2, 1),
            "kernel_size": (3,3,3,3,3,3,7),
        },
        "act": "RELU",
        "norm": "BATCH",
        "dropout": 0.0,
        "bias": False
    }
    model = TwoDPermuteConcatModel(config_bayat)
    pred_tensor = model(ap_img,lat_img)
    from torchview import draw_graph

    model_graph = draw_graph(model,input_data=[ap_img,lat_img],save_graph=True,filename='docs/arch_viz/TwoDPermuteConcat_modelgraph')
