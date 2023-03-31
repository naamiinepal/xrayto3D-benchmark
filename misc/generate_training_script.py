expt_dict = {
    "OneDConcat": "ParallelHeadsExperiment",
    "MultiScale2DPermuteConcat": "ParallelHeadsExperiment",
    "TwoDPermuteConcat": "ParallelHeadsExperiment",
    "AttentionUnet": "VolumeAsInputExperiment",
    "UNet": "VolumeAsInputExperiment",
}
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--trainpaths")
parser.add_argument("--testpaths")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--lr", default=0.0002)
parser.add_argument("--steps")
parser.add_argument("--img_size")
parser.add_argument("--res")
parser.add_argument("--tags")
parser.add_argument("--loss", default="DiceLoss")

args = parser.parse_args()
for model_name in expt_dict.keys():
    command = f"python train.py  {args.trainpaths} {args.testpaths} --gpu {args.gpu} --tags {args.tags} --size {args.img_size} --batch_size {args.batch_size} --accelerator gpu --res {args.res} --model_name {model_name} --epochs -1 --loss {args.loss}  --lr {args.lr} --steps {args.steps} \n"
    print(command)
