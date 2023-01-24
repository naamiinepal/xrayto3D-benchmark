import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from XrayTo3DShape import get_nonkasten_transforms,BaseDataset,get_deformation_transforms,DeformationDataset
from XrayTo3DShape import create_figure
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
import einops
from torch.distributions.normal import Normal
from torch import nn

def get_rotation_matrix(angle_in_degrees:float,tx=0.0,ty=0.0):
    # 2D rotation matrix
    theta = np.radians(angle_in_degrees)
    c,s = np.cos(theta),np.sin(theta)
    R = np.array(((c,-s,tx),(s,c,ty)),dtype=np.float32)
    return R

paths_location = 'configs/test/LIDC-DRR-test.csv'
paths = pd.read_csv(paths_location,index_col=0).to_numpy()
paths = [ {'ap':ap,'lat':lat,'seg':seg} for ap,lat,seg in paths] 

ds = BaseDataset(data=paths, transforms= get_nonkasten_transforms(size=64,resolution=1.5))
data_loader = DataLoader(ds,batch_size=1)
ap,lat,seg = next(iter(data_loader))
ap_tensor, lat_tensor, seg_tensor = ap['ap'], lat['lat'], seg['seg']

# print(ap_tensor.shape, lat_tensor.shape, seg_tensor.shape)

# theta = torch.tensor(get_rotation_matrix(15),dtype=torch.float32)
# theta = theta.view(1,2,3)

# grid = F.affine_grid(theta,ap_tensor.size(),align_corners=True)
# x_moving = F.grid_sample(ap_tensor,grid,align_corners=True)
# print(x_moving.shape,ap_tensor.shape,grid.shape)

# im = Image.fromarray(x_moving.squeeze().numpy()*255).convert('L')
# im.save('moving.png')

# im = Image.fromarray(ap_tensor.squeeze().numpy()*255).convert('L')
# im.save('fixed.png')

from scipy.spatial.transform import Rotation as R

r = R.from_euler(seq='zyx', angles = [5,20,5],degrees=True).as_matrix()
#3x4 matrix
theta_3d = torch.column_stack((torch.tensor(r,dtype=torch.float),torch.zeros(1,3,dtype=torch.float).T))
grid_3d = F.affine_grid(theta_3d.unsqueeze(0),seg_tensor.size(),align_corners=True)
seg_moving = F.grid_sample(seg_tensor,grid_3d,align_corners=True)
print(seg_moving.shape)

im = Image.fromarray(seg_moving.squeeze()[:,32,:].numpy()*255).convert('L')
im.save('seg_moving.png')

im = Image.fromarray(seg_tensor.squeeze()[:,32,:].numpy()*255).convert('L')
im.save('seg_fixed.png')

# fig, ax = create_figure(ap_tensor.squeeze(),x_moving.squeeze())

# ax[0].imshow(ap_tensor.squeeze(),cmap='gray')
# ax[1].imshow(x_moving.squeeze(),cmap='gray')
# # plt.show()

# fig = plt.figure()
# plt.scatter(grid.squeeze()[:,:,0],grid.squeeze()[:,:,1])
# plt.show()

paths_location = 'affine_transform_segpaths.csv'
paths = pd.read_csv(paths_location,index_col=0).to_numpy()
paths = [ {'fixed':f,'moving':m} for f,m in paths] 

ds = DeformationDataset(data=paths, transforms= get_deformation_transforms(size=64,resolution=1.5))
train_loader = DataLoader(ds,batch_size=1)
fixed,moving = next(iter(ds))
print(fixed['fixed'].shape,moving['moving'].shape)

import pytorch_lightning as pl
from torch import nn
from pytorch_lightning import seed_everything

SEED = 12345
seed_everything(SEED)
from monai.networks.blocks.convolutions import Convolution
class AffineRegistration(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            # feature extractor
            nn.Conv2d(1,8,kernel_size=7,padding=3),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(),
            nn.Conv2d(8,10,kernel_size=5,padding=2),
            nn.MaxPool2d(2,stride=2),
            nn.ReLU(),

        )
        
        self.deformation_field_generator = nn.Sequential(
            nn.Conv2d(10,14,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.Conv2d(14,16,kernel_size=5,padding=2),
            nn.ReLU(),
            # nn.Conv2d(16,16,kernel_size=3,padding=1),
            # nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ConvTranspose2d(16,16,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),
            # nn.Conv2d(16,16,kernel_size=3,padding=1),
            # nn.UpsamplingBilinear2d(scale_factor=2),            
            nn.ConvTranspose2d(16,16,kernel_size=4,stride=2,padding=1),
            nn.ReLU(),        
            nn.Conv2d(16,2,kernel_size=3,bias=False,padding=1),
        )

        self.deformation_field_generator[-1].weight.data.zero_() # type: ignore set tp zero deformable field initially
        # init flow layer with small weights and bias
        # self.deformation_field_generator[-1].weight = torch.nn.Parameter(Normal(0, 1e-5).sample(self.deformation_field_generator[-1].weight.size()))
        # self.deformation_field_generator[-1].bias = torch.nn.Parameter(torch.zeros(self.deformation_field_generator[-1].bias.size()))

        self.affine_matrix_generator = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2560,32),
            nn.ReLU(),
            nn.Linear(32,2*3) # learn only rotation angles no translation 
        )
        # initialize the weights /bias with identity transformation
        # self.affine_matrix_generator[-1].weight = torch.nn.Parameter(Normal(0,1e-5).sample(self.affine_matrix_generator[-1].weight.size()))
        self.affine_matrix_generator[-1].weight.data.zero_() # type: ignore
        self.affine_matrix_generator[-1].bias.data.copy_(torch.tensor([1,0,0,0,1,0],dtype=torch.float)) # type: ignore

        self.loss_function = torch.nn.MSELoss()
        

    def training_step(self,batch,batch_idx):
        fixed,moving = batch
        fixed_tensor = fixed['fixed']
        moving_tensor = moving['moving']
        out = self.encoder(moving_tensor)
        
        # affine decoder
        theta = self.affine_matrix_generator(out)
        theta = theta.view(-1,2,3)
        affine_grid = F.affine_grid(theta,moving_tensor.size())
        

        # deformable decoder
        deformable_field = self.deformation_field_generator(out)
        deformable_field = einops.rearrange(deformable_field,'b c h w -> b h w c')
        out = F.grid_sample(moving_tensor,affine_grid + deformable_field)

        loss = self.loss_function(out,fixed_tensor)
        self.log('train_loss',loss,on_step=True,on_epoch=True,batch_size=1)

        with torch.no_grad():
            deformable_field_mag = torch.complex(deformable_field.squeeze()[:,:,0],deformable_field.squeeze()[:,:,1]).abs()
            affine_field_mag = torch.complex(affine_grid.squeeze()[:,:,0],affine_grid.squeeze()[:,:,1]).abs()
            fig, ax = create_figure(out.squeeze(),fixed_tensor.squeeze(),deformable_field_mag,affine_field_mag)

            ax[0].imshow(out.squeeze(),cmap='gray')
            ax[0].title.set_text('moving')
            ax[1].imshow(fixed_tensor.squeeze(),cmap='gray')
            ax[1].title.set_text('fixed')
            ax[2].imshow(deformable_field_mag)
            ax[3].scatter(affine_grid.squeeze()[:,:,0],affine_grid.squeeze()[:,:,1])
            plt.savefig(f'transform_viz/epoch_{self.current_epoch}.png')
            plt.close()
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),lr=0.005) # this optimizer will optimize theta, choose learning rate



trainer = pl.Trainer(max_epochs=200,log_every_n_steps=1)
affine_registration = AffineRegistration()
trainer.fit(model=affine_registration,train_dataloaders= train_loader)

