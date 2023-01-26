from monai.transforms import Compose,Activations,AsDiscrete

post_transform = Compose([Activations(sigmoid=True), AsDiscrete(threshold=0.5)])
