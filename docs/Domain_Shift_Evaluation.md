There are two steps to benchmark evaluation of Domain Shift.

1. Dataset -> csv
   
   Assuming the dataset exists in some location, We need a csv file containing columns [ap, lat, seg] that list full paths of the input-groundtruth pair. 
   In our implementation, we have a config.yaml file that is used to generate this csv. The config file contains the subject_id list, the basepath of the dataset, 
   directories where x-rays, CT, and Segmentations are stored, as well as file name conventions. This config is also used to generate the processed dataset from the raw dataset.
   
2. script_template -> evaluation_script_generator.sh -> evaluation_script
   
   Next, we need to obtain checkpoint paths for corresponding architectures (specifically run-id of these architectures) and run `python evaluate.py parameters`.
   The automation of this is done using utilities that can identify Wandb run-id. The final evaluation script is stored at `bash_scripts>evaluate>domain_shift>`. 
   Since Embed-Decode architecture takes a two-step approach, the evaluation script for these may be in a separate file. All the other architecture evaluation commands may be 
stored in a single script.
Creating these scripts by one-self is prone-to-error, hence, requires automation. A `script template` may be stored in `scripts>script_templates` and contain '$var' placeholders .
 A `script_generator.sh` in `scripts>script_generator>` will be used to write `bash_scripts`. The `script_generator.sh` calls `generate_script.py` for each architectures. 
