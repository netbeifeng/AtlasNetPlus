from copy import deepcopy

import torch
from model.atlasnet import Atlasnet 
from model.resnet import resnet18
from model.pointnet import PointNet
from model.mvcnn import MVCNN, SVCNN

class BackboneNetwork(torch.nn.Module):
    def __init__(self, hparams):
        super(BackboneNetwork, self).__init__()
        self.hparams = hparams
        self.encoder = resnet18(pretrained=False, num_classes = hparams.bottleneck_size)
        self.mvcnn = MVCNN()
        self.classifier = torch.nn.Linear(hparams.bottleneck_size, 40)
        
        if hparams.forImg:
            if hparams.use_pre_trained_encoder:
                if hparams.mvr or hparams.mvcnn:
                    self.mvcnn.load_state_dict(torch.load(self.hparams.encoder_path))
                    print("MVCNN pretrained encoder loaded.")
                    self.encoder = deepcopy(self.mvcnn.net_1)
                    self.classifier = self.mvcnn.net_2
                else:
                    svcnn = SVCNN()
                    svcnn.load_state_dict(torch.load("pretrained/svcnn.pth"))
                    print("SVCNN pretrained encoder loaded.")
                    self.encoder = deepcopy(svcnn.net)
                    self.classifier = svcnn.classifer
                print("Encoder-MVCNN/SVCNN loaded.")
            else:
                self.encoder = resnet18(pretrained=False, num_classes = hparams.bottleneck_size)
                print("Encoder-ResNet-18 initialized.")
        else:
            self.encoder = PointNet(nlatent = hparams.bottleneck_size)
            print("Encoder-PointNet initialized.")
            
        self.decoder = Atlasnet(hparams)
            
        if hparams.use_pre_trained_decoder:
            self.decoder.load_state_dict(torch.load(hparams.decoder_path))
            print("Decoder-AtlasNet loaded.")
        else:
            print("Decoder-AtlasNet initialized.")
            
        self.to(hparams.device)

        self.eval()

    def forward(self, x, train=True, classification=False, mvcnn = False):
        # For training classification Network
        if classification:
            return self.classifier(self.encoder(x))
        else:
        # For training AtlasNet decoder
            if mvcnn:
                _,_,C,H,W = x.size()
                x = x.view(-1,C,H,W).to(self.hparams.device)
                x = self.encoder(x)
                size = int(x.shape[0] / 12)
                x = x.view((size, 12, x.shape[-1]))
                viewpooling = torch.max(x,1)[0]
                return self.decoder(viewpooling, train=train)
            else:
                return self.decoder(self.encoder(x), train=train)
    
    # For generating final mesh output
    def generate_mesh(self, x, mvcnn = False):
        self.decoder.eval()
        # if multiple view images as input
        if mvcnn:
            _,_,C,H,W = x.size()
            x = x.view(-1,C,H,W)
            x = self.encoder(x)
            size = int(x.shape[0] / 12)
            x = x.view((size, 12, x.shape[-1]))
            viewpooling = torch.max(x,1)[0]
            return self.decoder.generate_mesh(viewpooling)     
        else:
            return self.decoder.generate_mesh(self.encoder(x))     
    
    
    # For classification inference
    def class_inference(self, input, mvcnn = False):
        if self.hparams.forImg:
            if mvcnn:
                mvcnn = MVCNN()
                mvcnn.load_state_dict(torch.load("pretrained/mvcnn.pth"))
                mvcnn.eval()
                inference = mvcnn(input)
            else:
                svcnn = SVCNN()
                svcnn.load_state_dict(torch.load("pretrained/svcnn.pth"))
                svcnn.eval()
                inference = svcnn(input)
        else:
            pointnet = PointNet(nlatent = self.hparams.bottleneck_size)
            pointnet.load_state_dict(torch.load("pretrained/pointnet_encoder.pth"))
            pointnet.eval()
            
            classifier = torch.nn.Linear(self.hparams.bottleneck_size, 40)
            classifier.load_state_dict(torch.load("pretrained/pointnet_classifier.pth"))
            classifier.eval()
            
            encoding = pointnet(input)
            inference = classifier(encoding)
        return inference
            