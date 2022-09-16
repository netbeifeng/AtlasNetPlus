import os

from torch.utils.tensorboard import SummaryWriter   

from model.network import BackboneNetwork
from modelnet40 import ModelNet40AtlasNet
from hparams import HyperParameter
from thirdparty.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D import chamfer_3DDist
from thirdparty.ChamferDistancePytorch.fscore import fscore

from datetime import datetime
import torch

# Train Pipeline:
# 1. Train SVCNN / MVCNN to get a pre-trained encoder
# 2. Train MVCNN (pre-trained) + AtlasNet decoder

class Trainer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = BackboneNetwork(hparams)
        self.loss = chamfer_3DDist()
        
        self.optimized_parameters = [
            {"params": self.model.encoder.parameters()},
            {"params": self.model.decoder.parameters()}
        ]
                
        self.optimizer = torch.optim.Adam(self.optimized_parameters, self.hparams.lr, \
                                eps = 1e-8, betas=([0.9, 0.999]))
        
        self.train_set = ModelNet40AtlasNet('train', class_choice=hparams.class_choice, shuffle=True, 
                    img_normalization=True, pcd_normalization="UnitBall", num_point=hparams.num_points, mvcnn=hparams.mvr)
        
        self.val_set = ModelNet40AtlasNet('test', class_choice=hparams.class_choice, shuffle=False, 
                    img_normalization=True, pcd_normalization="UnitBall", num_point=hparams.num_points_eval, mvcnn=hparams.mvr)

        self.train_loader = torch.utils.data.DataLoader(self.train_set, 
                                                        batch_size=self.hparams.batch_size, 
                                                        shuffle=True)
        
        self.val_loader = torch.utils.data.DataLoader(self.val_set, 
                                                    batch_size=self.hparams.batch_size, 
                                                    shuffle=False)
    
    def setup(self):
        hparams = self.hparams
        time = datetime.now()
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minute = time.minute
        
        self.output_path = f'output/run_{year}_{month}_{day}_{hour}_{minute}'
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            os.makedirs(self.output_path + "/log")
        
        f = open(f'{self.output_path}/hparams.yaml','a')
        f.write(hparams.toString())
        f.close()
        
        self.writer = SummaryWriter(self.output_path + "/log")
        
        if hparams.model_path is not None:
            self.model.load_state_dict(torch.load(f'output/{hparams.model_path}',map_location='cpu'))
            self.model = self.model.to(hparams.device)
            
    def fit(self):
        print("Training Start ... ")
        print("Training Dataset Length: ", self.train_set.__len__())
        print("Validation Dataset Length: ", self.val_set.__len__())
        
        ratio = (self.train_set.__len__() / self.val_set.__len__())
        hparams = self.hparams
        device = hparams.device
        loss = self.loss.to(device)
        model = self.model
        optimizer = self.optimizer
        train_loader = self.train_loader
        val_loader = self.val_loader
        
        best_loss = 10000
        model.train()
        validating = True
        train_loss_running = 0.
        
        for epoch in range(hparams.max_epochs):
            
            # learning rate decay
            if epoch == hparams.lr_decay_1 or epoch == hparams.lr_decay_2 \
                    or epoch == hparams.lr_decay_3 or epoch == hparams.lr_decay_4:
                hparams.lr = hparams.lr / 10.0
                print(f"First learning rate decay {hparams.lr}")
                self.optimized_parameters = [
                    {"params": self.model.encoder.parameters()},
                    {"params": self.model.decoder.parameters()}
                ]
                optimizer = torch.optim.Adam(self.optimized_parameters, lr=hparams.lr, eps = 1e-8, betas=([0.9, 0.999]))
            
            train_epoch_loss = 0.
            print(f"Epoch {epoch}")
            for i, batch in enumerate(train_loader):
                train_loss_running = 0.
                
                # for image input
                if hparams.forImg:
                    
                    # for multiple views images
                    if hparams.mvr:
                        all_renderings = batch['all_renderings'].squeeze().float().to(device)   
                    # for single view image
                    else:
                        rendering = batch['rendering'].squeeze().float().to(device)   
                    pointcloud = batch['pointcloud'].squeeze().float().to(device)
                    gtPointcloud = pointcloud.contiguous()
                    
                    optimizer.zero_grad()
                    
                    if hparams.mvr:
                        predPointcloud = model(all_renderings, mvcnn=True).transpose(2, 3).contiguous()
                    else:
                        predPointcloud = model(rendering).transpose(2, 3).contiguous()
                        
                    predPointcloud = predPointcloud.view(predPointcloud.shape[0], -1, 3).contiguous()
                    
                    # print(predPointcloud.shape) # 16, 1, 2000, 3
                    
                    dist1, dist2, _, _  = loss(gtPointcloud, predPointcloud)
                    loss_total = ((torch.mean(dist1)) + (torch.mean(dist2))).to(device)
                    
                    
                    loss_total.backward()
                    optimizer.step()
                    train_epoch_loss += loss_total.item()
                    train_loss_running += loss_total.item()
                    iteration = epoch * len(train_loader) + i
                    if iteration % hparams.print_every_n == (hparams.print_every_n - 1):
                        print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running:.5f}')
                        
                # for pointcloud input
                else:
                    pointcloud = batch['pointcloud'].squeeze().float().to(device)
                    gtPointcloud = pointcloud.contiguous()
                    optimizer.zero_grad()
                    
                    predPointcloud = model(pointcloud.view(pointcloud.shape[0], 3, -1))
                    predPointcloud = predPointcloud.transpose(2, 3).contiguous().squeeze()
                    predPointcloud = predPointcloud.view(predPointcloud.shape[0], -1, 3).contiguous()
                    
                    dist1, dist2, _, _  = loss(gtPointcloud, predPointcloud)
                    loss_total = ((torch.mean(dist1)) + (torch.mean(dist2))).to(device)
                    
                    loss_total.backward()
                    optimizer.step()
                    train_epoch_loss += loss_total.item()
                    train_loss_running += loss_total.item()
                    iteration = epoch * len(train_loader) + i
                    if iteration % hparams.print_every_n == (hparams.print_every_n - 1):
                        print(f'[{epoch:03d}/{i:05d}] train_loss: {train_loss_running:.5f}')
                    
            # record epoch loss
            print(f"Epoch {epoch} finished, we have training loss {train_epoch_loss / ratio}")
            self.writer.add_scalar(f"loss/train_loss", train_epoch_loss / ratio, global_step=epoch)
                
            # validation evaluation and logging
            if validating is True:
                if epoch % hparams.validate_every_n_epoch == (hparams.validate_every_n_epoch - 1):
                    print("Validating ... ")
                    
                    # set model to eval, important if your network has e.g. dropout or batchnorm layers
                    model.eval()
                    loss_total_val = 0
                    loss_total_fscore = 0
                    total= 0
                    
                    # forward pass and evaluation for entire validation set
                    for i, batch_val in enumerate(val_loader):
                        
                        if hparams.forImg:
                            if hparams.mvr:
                                val_all_renderings = batch_val['all_renderings'].squeeze().float().to(device)   
                            else:
                                val_rendering = batch_val['rendering'].squeeze().float().to(device)   
                            val_pointcloud = batch_val['pointcloud'].squeeze().float().to(device)
                            gtPointcloud = val_pointcloud.contiguous()
                        
                            with torch.no_grad():
                                optimizer.zero_grad()
                                if hparams.mvr:
                                    predPointcloud = model(val_all_renderings, train=False, mvcnn=True).transpose(2, 3).contiguous()
                                else:
                                    predPointcloud = model(val_rendering, train=False).transpose(2, 3).contiguous()
                                
                                predPointcloud = predPointcloud.view(predPointcloud.shape[0], -1, 3).contiguous()

                                dist1, dist2, _, _  = loss(gtPointcloud, predPointcloud)
                                loss_val_per = ((torch.mean(dist1)) + (torch.mean(dist2))).to(device)
                                loss_fscore, _, _ = fscore(dist1, dist2)
                                loss_fscore_mean = loss_fscore.mean()
                            
                            loss_total_fscore += loss_fscore_mean.item()
                            loss_total_val += loss_val_per.item()
                            total += val_pointcloud.shape[0]
                            print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val_per:.5f}')
                            print(f'[{epoch:03d}/{i:05d}] fscore: {loss_total_fscore:.5f}')
                            
                        else:
                            val_pointcloud = batch_val['pointcloud'].squeeze().float().to(device)
                            gtPointcloud = val_pointcloud.contiguous()

                            with torch.no_grad():
                                optimizer.zero_grad()
                                
                                predPointcloud = model(val_pointcloud.view(val_pointcloud.shape[0], 3, -1), False)
                                predPointcloud = predPointcloud.transpose(2, 3).contiguous().squeeze()
                                predPointcloud = predPointcloud.view(predPointcloud.shape[0], -1, 3).contiguous()

                                dist1, dist2, _, _  = loss(gtPointcloud, predPointcloud)
                                loss_val_per = ((torch.mean(dist1)) + (torch.mean(dist2))).to(device)
                                loss_fscore, _, _ = fscore(dist1, dist2)
                                loss_fscore_mean = loss_fscore.mean()
                            
                            loss_total_fscore += loss_fscore_mean.item()
                            loss_total_val += loss_val_per.item()
                            total += val_pointcloud.shape[0]
                            print(f'[{epoch:03d}/{i:05d}] val_loss: {loss_val_per:.5f}')
                            
                    print(f"Epoch {epoch} finished, we have val loss {loss_total_val}")
                        
                    self.writer.add_scalar(f"loss/val_loss", loss_total_val, epoch)
                    self.writer.add_scalar(f"loss/val_fscore", loss_total_fscore, epoch)

                    if loss_total_val < best_loss:
                        print('better loss, model saved.')
                        torch.save(model.state_dict(), f'{self.output_path}/model_best_{epoch}.ckpt') 
                        best_loss = loss_total_val

                    if hparams.saveEveryValidation:
                        torch.save(model.state_dict(), f'{self.output_path}/model_epoch_{epoch}.ckpt') 
                    
                    model.train()
        
        # Final saving
        print("Model saving ... ")                
        torch.save(model.state_dict(), f'{self.output_path}/model_final.ckpt')
                    
if __name__ == "__main__":
    print("Trainer initialized.")
    hparams = HyperParameter()
    trainer = Trainer(hparams)
    trainer.setup()
    trainer.fit()
    