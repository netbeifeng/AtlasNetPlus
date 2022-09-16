import os
import numpy

from torch.utils.tensorboard import SummaryWriter   

from model.network import BackboneNetwork
from modelnet40 import ModelNet40AtlasNet
from hparams import HyperParameter

from datetime import datetime
import torch

class PointNetTrainer:
    def __init__(self, hparams):
        self.hparams = hparams
        self.model = BackboneNetwork(hparams)
        self.loss = torch.nn.CrossEntropyLoss()
        
        self.optimized_parameters = [
            {"params": self.model.encoder.parameters()},
            {"params": self.model.classifier.parameters()}
        ]
                
        self.optimizer = torch.optim.Adam(self.optimized_parameters, self.hparams.lr, \
                                eps = 1e-8, betas=([0.9, 0.999]))
        
        self.train_set = ModelNet40AtlasNet('train', class_choice=hparams.class_choice, shuffle=True, 
                    img_normalization=True, pcd_normalization="UnitBall", num_point=hparams.num_points)
        
        self.val_set = ModelNet40AtlasNet('test', class_choice=hparams.class_choice, shuffle=False, 
                    img_normalization=True, pcd_normalization="UnitBall", num_point=hparams.num_points_eval)

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
        
        self.output_path = f'output/pointnet_{year}_{month}_{day}_{hour}_{minute}'
        
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
        hparams = self.hparams
        device = hparams.device
        loss = self.loss.to(device)
        model = self.model
        train_loader = self.train_loader
        val_loader = self.val_loader
        
        model.train()
        validating = True
        best_acc = 0
        i_acc = 0
        
        for epoch in range(hparams.max_epochs):
            if epoch == hparams.lr_decay_1 or epoch == hparams.lr_decay_2 \
                    or epoch == hparams.lr_decay_3 or epoch == hparams.lr_decay_4:
                hparams.lr = hparams.lr / 10.0
                print(f"First learning rate decay {hparams.lr}")
                self.optimized_parameters = [
                    {"params": self.model.encoder.parameters()},
                    {"params": self.model.decoder.parameters()}
                ]
                self.optimizer = torch.optim.Adam(self.optimized_parameters, lr=hparams.lr, eps = 1e-8, betas=([0.9, 0.999]))
            
            print(f"Epoch {epoch + 1} Start...")
            for i, batch in enumerate(train_loader):
                pointcloud = batch['pointcloud'].squeeze().float().to(device)
                pointcloud = pointcloud.transpose(1,2)
                label = torch.autograd.Variable(batch['class_id']).long().to(device)

                self.optimizer.zero_grad()

                out_data = self.model(pointcloud, classification=True)

                loss = self.loss(out_data, label)
                
                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == label
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)
                    
            i_acc += i                

            # validation evaluation and logging
            if validating is True:
                with torch.no_grad():
                    if epoch % hparams.validate_every_n_epoch == (hparams.validate_every_n_epoch - 1):
                        print("Validating ... ")
                        # set model to eval, important if your network has e.g. dropout or batchnorm layers
                        model.eval()
                        all_correct_points = 0
                        all_points = 0

                        wrong_class = numpy.zeros(40)
                        samples_class = numpy.zeros(40)
                        all_loss = 0
                        
                        # forward pass and evaluation for entire validation set
                        for i, batch_val in enumerate(val_loader):
                            val_pointcloud = batch_val['pointcloud'].squeeze().float().to(device)
                            val_pointcloud = val_pointcloud.transpose(1,2)
                            val_label = batch_val['class_id'].long().to(device)
                            
                            out_data = self.model(val_pointcloud, classification= True)
                            pred = torch.max(out_data, 1)[1]
                            all_loss += self.loss(out_data, val_label).cpu().data.numpy()
                            results = pred == val_label

                            for i in range(results.size()[0]):
                                if not bool(results[i].cpu().data.numpy()):
                                    wrong_class[val_label.cpu().data.numpy().astype('int')[i]] += 1
                                samples_class[val_label.cpu().data.numpy().astype('int')[i]] += 1
                            correct_points = torch.sum(results.long())

                            all_correct_points += correct_points
                            all_points += results.size()[0]

                        print ('Total # of val samples: ', all_points)
                        val_mean_class_acc = numpy.mean((samples_class-wrong_class)/samples_class)
                        acc = all_correct_points.float() / all_points
                        val_overall_acc = acc.cpu().data.numpy()
                        loss = all_loss / len(self.val_loader)

                        print ('val mean class acc. : ', val_mean_class_acc)
                        print ('val overall acc. : ', val_overall_acc)
                        print ('val loss : ', loss)
                        self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                        self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                        self.writer.add_scalar('val/val_loss', loss, epoch+1)

                        if val_overall_acc > best_acc:
                            print('better loss, model saved.')
                            best_acc = val_overall_acc
                            torch.save(self.model.state_dict(), f'{self.output_path}/pointnet_best_{epoch}.pth')

                        # set model back to train
                        model.train()
        # Final saving
        print("Model saving ... ")                
        torch.save(model.encoder.state_dict(), f'{self.output_path}/pointnet.ckpt')
                    
if __name__ == "__main__":
    print("Trainer initialized.")
    hparams = HyperParameter()
    hparams.forImg = False
    trainer = PointNetTrainer(hparams)
    trainer.setup()
    trainer.fit()
    