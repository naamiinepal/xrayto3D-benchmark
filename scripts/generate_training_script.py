"""generate training script"""
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
parser.add_argument("--tags", nargs="*")
parser.add_argument("--loss", default="DiceLoss")
parser.add_argument("--dropout", default=False, action="store_true")
parser.add_argument('--patch',default=False,action='store_true')
parser.add_argument('--num_workers',default=8)

expt_dict = {
    "OneDConcat": "ParallelHeadsExperiment",
    "MultiScale2DPermuteConcat": "ParallelHeadsExperiment",
    "TwoDPermuteConcat": "ParallelHeadsExperiment",
    "AttentionUnet": "VolumeAsInputExperiment",
    "UNet": "VolumeAsInputExperiment",
    "UNETR": "VolumeAsInputExperiment",
    'SwinUNETR': 'VolumeAsInputExperiment'
}

args = parser.parse_args()

batch_size = [args.batch_size,]*len(expt_dict.keys()) if not args.patch else [args.batch_size,2,args.batch_size,args.batch_size,args.batch_size, 1,1]
for model_name,model_wise_batch_size in zip(expt_dict.keys(),batch_size):
    command = f"python train.py  {args.trainpaths} {args.testpaths} --gpu {args.gpu} --size {args.img_size} --batch_size {model_wise_batch_size} --accelerator gpu --res {args.res} --model_name {model_name} --epochs -1 --loss {args.loss}  --lr {args.lr} --steps {args.steps} --num_workers {args.num_workers}"
    command += " --dropout" if args.dropout else ""
    tag_command = ' --tags'
    for tag in args.tags:
        tag_command += f' {tag}'
    command += tag_command
    print(command)
