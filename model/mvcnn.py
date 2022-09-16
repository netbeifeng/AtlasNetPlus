import torch
import torch.nn as nn
from .resnet import resnet18
from torch.autograd import Variable

mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1, 
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class SVCNN(torch.nn.Module):

    def __init__(self, nclasses=40, preTraining=True, bottleneck_size = 1024):
        super(SVCNN, self).__init__()

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        if preTraining:
            self.net = resnet18(pretrained= True,num_classes=1000)
            self.net.fc = nn.Linear(512,bottleneck_size)
        else:
            self.net = resnet18(pretrained= False,num_classes=bottleneck_size)

        self.classifer = nn.Linear(bottleneck_size, 40)

        self.model = nn.Sequential(
            self.net,
            self.classifer 
        )

    def forward(self, x):
        return self.model(x)



class MVCNN(torch.nn.Module):

    def __init__(self, model = SVCNN(), nclasses=40, num_views=12):
        super(MVCNN, self).__init__()

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.num_views = num_views
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).cuda()
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).cuda()

        self.net_1 = model.net
        self.net_2 = model.classifer


    def forward(self, x):
        y = self.net_1(x) # 12, 1024
        size = int(x.shape[0] / self.num_views) # 12 / 12 = 1
        y = y.view((size, self.num_views, y.shape[-1])) # 1, 12, 1024
        
        viewpooling = torch.max(y,1)[0] # max on 1 channel => all views 
        
        y = self.net_2(viewpooling.view(size,-1))
        return y

