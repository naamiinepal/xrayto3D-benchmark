"""generate training script for 2 stage embed and decode"""
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
parser.add_argument("--dropout", default=False, action="store_true")

args = parser.parse_args()

# #!/usr/bin/bash
# entity_name="msrepo"
# project_name="2d-3d-benchmark"
# trainpaths="configs/paths/verse19/Verse2019-DRR-full_train+val.csv"
# testpaths="configs/paths/verse19/Verse2019-DRR-full_test.csv"
# gpu=0
# batch_size=8
# lr=0.002
# steps=5000
# tag="model-compare"
# img_size=64
# res=1.5

var_definition = "#!/usr/bin/bash\n"
var_definition += "entity_name=msrepo\n"
var_definition += "project_name=2d-3d-benchmark\n"
var_definition += f"trainpaths={args.trainpaths}\n"
var_definition += f"testpaths={args.testpaths}\n"
var_definition += f"gpu={args.gpu}\n"
var_definition += f"batch_size={args.batch_size}\n"
var_definition += f"lr={args.lr}\n"
var_definition += f"steps={args.steps}\n"
var_definition += f"tag={args.tags}\n"
var_definition += f"img_size={args.img_size}\n"
var_definition += f"res={args.res}\n"
var_definition += f"loss={args.loss}\n"

with open("scripts/embed_decode_template.sh", "r") as f:
    template_script = f.readlines()
    full_script = str(var_definition) + "\n".join(template_script)
    print(full_script)
