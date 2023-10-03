"""generate evaluation script for 2 stage embed and decode"""
import argparse
from pathlib import Path
from XrayTo3DShape import get_anatomy_from_path
parser = argparse.ArgumentParser()
parser.add_argument("--testpaths")
parser.add_argument("--gpu", default=0, type=int)
parser.add_argument("--batch_size", default=8)
parser.add_argument("--img_size")
parser.add_argument("--res")
parser.add_argument("--tags")
args = parser.parse_args()

# #!/usr/bin/bash
# entity_name="msrepo"
# project_name="2d-3d-benchmark"
# testpaths="configs/paths/verse19/Verse2019-DRR-full_test.csv"
# gpu=0
# batch_size=8
# img_size=64
# res=1.5

var_definition = "#!/usr/bin/bash\n"
var_definition += "entity_name=msrepo\n"
var_definition += "project_name=2d-3d-benchmark\n"
var_definition += f"anatomy={get_anatomy_from_path(args.testpaths)}\n"
var_definition += f"img_size={args.img_size}\n"
var_definition += f"res={args.res}\n"
var_definition += f"gpu={args.gpu}\n"
var_definition += f"batch_size={args.batch_size}\n"

ANGLE_PERTURBATIONS = [ 1,2,5,10]
with open("scripts/script_templates/evaluate_embed_decode_angle_preturbation_template.sh", "r") as f:
    for idx, angle in enumerate(ANGLE_PERTURBATIONS):

        
        angle_perturbation_csv_path = Path(args.testpaths).with_name(Path(args.testpaths).stem+'_'+str(angle)).with_suffix('.csv')
        
        if idx != 0: 
            var_definition = ''
        var_definition += f"testpaths={angle_perturbation_csv_path}\n"
        var_definition +=f'angle={angle}\n'

        template_script = f.readlines()
        full_script = str(var_definition) + "\n".join(template_script)
        print(full_script)
