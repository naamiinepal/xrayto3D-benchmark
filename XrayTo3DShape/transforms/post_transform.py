from monai.transforms import Compose,Activations,AsDiscrete

post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
post_transform_oh = Compose([AsDiscrete(argmax=True)])
