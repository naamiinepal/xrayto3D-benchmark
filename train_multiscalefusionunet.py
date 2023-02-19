import torch
from XrayTo3DShape import get_dataset,get_nonkasten_transforms,TwoDPermuteConcatMultiScale,BiplanarAsInputExperiment,parse_training_arguments,NiftiPredictionWriter
from torch.utils.data import DataLoader
from monai.losses.dice import DiceCELoss,DiceLoss
from monai.utils.misc import set_determinism
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':

    args = parse_training_arguments()

    SEED = 12345
    EXPERIMENT = 'MultiScaleFusionUNet'
    lr = 1e-2
    NUM_EPOCHS = args.epochs
    IMG_SIZE = args.size
    IMG_RESOLUTION = args.res
    WANDB_ON = False
    TEST_ZERO_INPUT = False
    BATCH_SIZE = args.batch_size
    WANDB_PROJECT = 'pipeline-test-01'
    model_config = {
        "in_shape": (1,IMG_SIZE,IMG_SIZE),
        "kernel_size":3,
        "act":'RELU',
        "norm":"BATCH",
        # "deep_supr_num" : 2,

        "encoder":{
            "kernel_size":(3,)*4,
            'deep_supervision':False,
            'filters':[8,16,32,64],
            "strides":(1,2,2,2),   # keep the first element of the strides 1 so the input and output shape match

        },
        "decoder": {
            "out_channel":2,
            "kernel_size":3
        }
    }

    set_determinism(seed=SEED)
    seed_everything(seed=SEED)    

    train_transforms = get_nonkasten_transforms()
    train_loader = DataLoader(get_dataset(args.trainpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20)
    val_loader = DataLoader(get_dataset(args.valpaths,transforms=train_transforms),batch_size=BATCH_SIZE,num_workers=20)



    model = TwoDPermuteConcatMultiScale(model_config)
    x_ray_img = torch.zeros(1,1,64, 64)

    ap_out = model.ap_encoder(x_ray_img)
    lat_out = model.lat_encoder(x_ray_img)
    [print(a.shape) for a in ap_out]
    [print(a.shape) for a in lat_out]
    # print(f'decoders {len(model.decoder)}')
    # fused_cube_0 = torch.stack((ap_out[-1],lat_out[-1]),dim=1)
    # print(f'fused cube 0 {fused_cube_0.shape}')
    # dec_out_0 = model.decoder[0](fused_cube_0)
    # print(dec_out_0.shape)
    # fused_cube_1 = torch.stack((ap_out[-2],lat_out[-2]),dim=1)
    # dec_out_1 = model.decoder[1](torch.cat((dec_out_0,fused_cube_1),dim=1))
    # print(f'fused cube 1 {fused_cube_1.shape}')
    # print(dec_out_1.shape)
    # fused_cube_2 = torch.stack((ap_out[-3],lat_out[-3]),dim=1)
    # dec_out_2 = model.segmentation_head(torch.cat((dec_out_1,fused_cube_2),dim=1))
    # print('fused cube 2',fused_cube_2.shape)
    # print(dec_out_2.shape)
    # out = model(x_ray_img,x_ray_img)
    # print(f'output {out.shape}')
    out = model(x_ray_img,x_ray_img)
    print(out.shape)
    loss_function = DiceLoss(sigmoid=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr)

    experiment = BiplanarAsInputExperiment(model,optimizer,loss_function,BATCH_SIZE)

    if args.evaluate and args.save_predictions:
        nifti_saver = NiftiPredictionWriter(output_dir=args.output_dir,write_interval='batch')
        trainer = pl.Trainer(callbacks=[nifti_saver])
        trainer.predict(model=experiment,ckpt_path=args.checkpoint_path,dataloaders=val_loader,return_predictions=False)
    else:
        # loggers
        wandb_logger = WandbLogger(save_dir='runs/',project=WANDB_PROJECT,group=EXPERIMENT,tags=[EXPERIMENT,'model_selection',*args.tags])
        wandb_logger.log_hyperparams({'model':model_config})
        trainer = pl.Trainer(accelerator=args.accelerator,precision=args.precision,max_epochs=-1,gpus=[args.gpu],deterministic=False,log_every_n_steps=1,auto_select_gpus=True,logger=[wandb_logger],enable_progress_bar=True,enable_checkpointing=True)
        
        trainer.fit(experiment,train_loader,val_loader)