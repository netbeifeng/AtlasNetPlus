import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.tensorboard import SummaryWriter   
from datetime import datetime
import os

from hparams import HyperParameterMVCNN
from model.mvcnn import MVCNN, SVCNN
from modelnet40 import ModelNet40MultiView, ModelNet40SingleView

class MVCNNTrainer(object):

    def __init__(self, model, train_loader, val_loader, optimizer, loss_fn, model_name, hparams):
        self.hparams = hparams
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.model_name = model_name
        self.num_views = hparams.num_views
        self.model.cuda()


    def train(self, n_epochs):

        best_acc = 0
        i_acc = 0
        self.model.train()
        for epoch in range(n_epochs):
            # plot learning rate
            lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            self.writer.add_scalar('params/lr', lr, epoch)

            # train one epoch
            out_data = None
            in_data = None
            for i, data in enumerate(self.train_loader):

                if self.model_name == 'mvcnn':
                    N,V,C,H,W = data[1].size()
                    in_data = Variable(data[1]).view(-1,C,H,W).cuda()
                else:
                    in_data = Variable(data[1].cuda())
                target = Variable(data[0]).cuda().long()

                self.optimizer.zero_grad()

                out_data = self.model(in_data)

                loss = self.loss_fn(out_data, target)
                
                self.writer.add_scalar('train/train_loss', loss, i_acc+i+1)

                pred = torch.max(out_data, 1)[1]
                results = pred == target
                correct_points = torch.sum(results.long())

                acc = correct_points.float()/results.size()[0]
                self.writer.add_scalar('train/train_overall_acc', acc, i_acc+i+1)

                loss.backward()
                self.optimizer.step()
                
                log_str = 'epoch %d, step %d: train_loss %.3f; train_acc %.3f' % (epoch+1, i+1, loss, acc)
                if (i+1)%1==0:
                    print(log_str)
            i_acc += i

            # evaluation
            if (epoch+1)%1==0:
                with torch.no_grad():
                    loss, val_overall_acc, val_mean_class_acc = self.update_validation_accuracy(epoch)
                self.writer.add_scalar('val/val_mean_class_acc', val_mean_class_acc, epoch+1)
                self.writer.add_scalar('val/val_overall_acc', val_overall_acc, epoch+1)
                self.writer.add_scalar('val/val_loss', loss, epoch+1)

            # save best model
            if val_overall_acc > best_acc:
                print('better loss, model saved.')
                best_acc = val_overall_acc
                torch.save(self.model.state_dict(), f'{self.output_path}/{self.model_name}_best_{epoch}.pth') # model_best.ckpt
                # self.model.save(self.log_dir, epoch)
 
            # adjust learning rate manually
            if epoch > 0 and (epoch+1) % 10 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr']*0.5

            if hparams.saveEveryValidation:
                torch.save(self.model.state_dict(), f'{self.output_path}/{self.model_name}_epoch_{epoch}.pth') 
                
        print("Model saving ... ")                
        torch.save(self.model.state_dict(), f'{self.output_path}/{self.model_name}_final.pth') 


    def update_validation_accuracy(self, epoch):
        all_correct_points = 0
        all_points = 0

        wrong_class = np.zeros(40)
        samples_class = np.zeros(40)
        all_loss = 0

        self.model.eval()

        print("Validating ... ")
        for _, data in enumerate(self.val_loader, 0):

            if self.model_name == 'mvcnn':
                N,V,C,H,W = data[1].size()
                in_data = Variable(data[1]).view(-1,C,H,W).cuda()
            else:#'svcnn'
                in_data = Variable(data[1]).cuda()
            target = Variable(data[0]).cuda()

            out_data = self.model(in_data)
            pred = torch.max(out_data, 1)[1]
            all_loss += self.loss_fn(out_data, target).cpu().data.numpy()
            results = pred == target

            for i in range(results.size()[0]):
                if not bool(results[i].cpu().data.numpy()):
                    wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
                samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
            correct_points = torch.sum(results.long())

            all_correct_points += correct_points
            all_points += results.size()[0]

        print ('Total # of val samples: ', all_points)
        val_mean_class_acc = np.mean((samples_class-wrong_class)/samples_class)
        acc = all_correct_points.float() / all_points
        val_overall_acc = acc.cpu().data.numpy()
        loss = all_loss / len(self.val_loader)

        print ('val mean class acc. : ', val_mean_class_acc)
        print ('val overall acc. : ', val_overall_acc)
        print ('val loss : ', loss)

        self.model.train()

        return loss, val_overall_acc, val_mean_class_acc
    
    def setup(self):
        hparams = self.hparams
        time = datetime.now()
        year = time.year
        month = time.month
        day = time.day
        hour = time.hour
        minute = time.minute
        
        self.output_path = f'output/{self.model_name}_run_{year}_{month}_{day}_{hour}_{minute}'
        
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            os.makedirs(self.output_path + "/log")
        
        f = open(f'{self.output_path}/hparams.yaml','a')
        f.write(hparams.toString())
        f.close()
        
        self.writer = SummaryWriter(self.output_path + "/log")
        
if __name__ == "__main__":
    print("Trainer initialized.")
    hparams = HyperParameterMVCNN()
    
    if hparams.skip_svcnn is False:
        print("Start training SVCNN ... ")
        svcnn = SVCNN(preTraining=hparams.pretraining_imagenet, bottleneck_size=hparams.bottleneck_size)
        optimizer = torch.optim.Adam(svcnn.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay)
        
        train_dataset = ModelNet40SingleView("train", scale_aug=False, rot_aug=False,  num_views=hparams.num_views, shuffle=True)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.svcnn_batch_size, shuffle=True, num_workers=0)
        
        val_dataset = ModelNet40SingleView("test", scale_aug=False, rot_aug=False, num_views=hparams.num_views, shuffle=False)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.svcnn_batch_size, shuffle=False, num_workers=0)
        print('num_train_files: ', train_dataset.__len__())
        print('train epochs ', train_dataset.__len__()/hparams.svcnn_batch_size)
        print('num_val_files: ', val_dataset.__len__())
        print('val epochs ', val_dataset.__len__()/hparams.svcnn_batch_size)
        
        trainer = MVCNNTrainer(svcnn, train_loader, val_loader, optimizer, torch.nn.CrossEntropyLoss(), "svcnn", hparams)
        trainer.setup()
        trainer.train(hparams.max_epochs)
        print("SVCNN Trained.")
    else:
        svcnn = SVCNN(preTraining=False, bottleneck_size=hparams.bottleneck_size)
        svcnn.load_state_dict(torch.load())
        print("SVCNN Loaded.")
    
    print("SVCNN Done.")
        
    print("Start training MVCNN ... ")
    mvcnn = MVCNN(svcnn, nclasses=40,num_views=hparams.num_views)
    del svcnn
    
    optimizer = torch.optim.Adam(mvcnn.parameters(), lr=hparams.lr, weight_decay=hparams.weight_decay, betas=(0.9, 0.999), eps=1e-8)
    
    train_dataset = ModelNet40MultiView("train", scale_aug=False, rot_aug=False, num_views=hparams.num_views, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hparams.mvcnn_batch_size, shuffle=True, num_workers=0)
    
    val_dataset = ModelNet40MultiView("test", scale_aug=False, rot_aug=False, num_views=hparams.num_views, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hparams.mvcnn_batch_size, shuffle=False, num_workers=0)
    
    print('num_train_files: ', train_dataset.__len__())
    print('train epochs ', train_dataset.__len__()/hparams.mvcnn_batch_size)
    
    print('num_val_files: ', val_dataset.__len__())
    print('val epochs ', val_dataset.__len__()/hparams.mvcnn_batch_size)
    
    trainer = MVCNNTrainer(mvcnn, train_loader, val_loader, optimizer, torch.nn.CrossEntropyLoss(), 'mvcnn', hparams)
    trainer.setup()
    trainer.train(hparams.max_epochs)
    
    print("MVCNN Trained.")
    print("MVCNN Done.")
    