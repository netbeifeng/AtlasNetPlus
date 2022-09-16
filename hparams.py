import torch
import yaml

class HyperParameter(object):
    encoder_path = "pretrained/mvcnn.pth"
    decoder_path = "pretrained/atlasnet.ckpt"
    use_pre_trained_encoder = True      
    use_pre_trained_decoder = True
    
    mvr = False                         # multiple view reconstruction
    mvcnn = False                       # use MVCNN as encoder 
    forImg = True                       # input is image
    
    model_path = None                   # for continue training pre-trained model  

    template_type = "SPHERE"            # template type
    class_choice = None                 # which class will be used for training like ['airplane', 'toilet']
                                        # None is for all classes        
                                        
    bottleneck_size = 1024              # bottleneck size of autoencoder
    
    num_points = 2000                   # num of sampled points 
    num_points_eval = 2000              
    
    num_layers = 2                      # number of layers                    
    hidden_neurons = 512                # number of hidden neurons for each layer
    num_prims = 1                       # number of primitives 
    
    batch_size = 16                        
    print_every_n = 1
    validate_every_n_epoch = 5
    
    max_epochs = 150
    
    activation = 'relu'
    
    dim_template_dict = {
        "SQUARE": 2,
        "SPHERE": 3,
    }
    
    lr = 0.001
    lr_decay_1 = 100
    lr_decay_2 = 200
    lr_decay_3 = 250
    lr_decay_4 = 280
    saveEveryValidation = False
    
    def __init__(self):
        self.dim_template = self.dim_template_dict[self.template_type]
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:0")
        else:
            self.device = torch.device(f"cpu")
    
    # for loading history hyperparameters yaml
    def load(self, path):
        with open(path, 'r') as f:
            doc = yaml.load(f, Loader=yaml.FullLoader)
            self.activation = doc['activation']
            self.lr = doc['lr']
            self.batch_size = doc['batch_size']
            self.bottleneck_size = doc['bottleneck_size']
            self.forImg = doc['forImg']
            self.num_layers = doc['num_layers']
            self.num_prims = doc['num_prims']
            self.use_pre_trained_decoder = doc['use_pre_trained_decoder']
            self.use_pre_trained_encoder = doc['use_pre_trained_encoder']
            self.dim_template = doc['dim_template']
            self.template_type = doc['template_type']
            self.mvcnn = doc['mvcnn']
            self.mvr = doc['mvr']
            self.device = torch.device(f"cpu")
            
    # for saving history hyperparameters yaml
    def toString(self):
        lr_str = f'lr: {self.lr.__str__()}\n'
        lr_decay_1_str = f'lr_decay_1: {self.lr_decay_1.__str__()}\n'
        lr_decay_2_str = f'lr_decay_2: {self.lr_decay_2.__str__()}\n'
        lr_decay_3_str = f'lr_decay_3: {self.lr_decay_3.__str__()}\n'
        lr_decay_4_str = f'lr_decay_4: {self.lr_decay_4.__str__()}\n'

        forImg_str = f'forImg: {self.forImg.__str__()}\n'
        mvr_str = f'mvr: {self.mvr.__str__()}\n'
        mvcnn_str = f'mvcnn: {self.mvcnn.__str__()}\n'
        
        template_type_str = f'template_type: {self.template_type.__str__()}\n'
        dim_template_str = f'dim_template: {self.dim_template.__str__()}\n'
        
        num_layers_str = f'num_layers: {self.num_layers.__str__()}\n'
        num_prims_str = f'num_prims: {self.num_prims.__str__()}\n'
        batch_size_str = f'batch_size: {self.batch_size.__str__()}\n'
        bottleneck_size_str = f'bottleneck_size: {self.bottleneck_size.__str__()}\n'
        max_epochs_str = f'max_epochs: {self.max_epochs.__str__()}\n'
        hidden_neurons_str = f'hidden_neurons: {self.hidden_neurons.__str__()}\n'
        activation_str = f'activation: {self.activation.__str__()}\n'
        saveEveryValidation_str = f'saveEveryValidation: {self.saveEveryValidation.__str__()}\n'
        
        class_choice_str = f'class_choice: {self.class_choice.__str__()}\n'

        print_every_n_str = f'print_every_n: {self.print_every_n.__str__()}\n'
        validate_every_n_epoch_str = f'validate_every_n_epoch: {self.validate_every_n_epoch.__str__()}\n'
        
        device_str = f'device: {self.device.__str__()}\n'
        num_points_str = f'num_points: {self.num_points.__str__()}\n'
        num_points_eval_str = f'num_points_eval: {self.num_points_eval.__str__()}\n'
        use_pre_trained_encoder_str = f'use_pre_trained_encoder: {self.use_pre_trained_encoder.__str__()}\n'
        use_pre_trained_decoder_str = f'use_pre_trained_decoder: {self.use_pre_trained_decoder.__str__()}\n'
        
        output = lr_str + forImg_str + template_type_str + dim_template_str + mvr_str + mvcnn_str \
            + num_layers_str + num_prims_str + lr_decay_1_str + lr_decay_2_str + lr_decay_3_str + lr_decay_4_str \
            + batch_size_str + bottleneck_size_str + max_epochs_str + hidden_neurons_str + activation_str \
            + class_choice_str + print_every_n_str + validate_every_n_epoch_str + device_str + num_points_str \
            + num_points_eval_str + saveEveryValidation_str + use_pre_trained_encoder_str + use_pre_trained_decoder_str
            
        return output

# Hyperparameter for training MVCNN
class HyperParameterMVCNN(object):
    svcnn_batch_size = 64
    mvcnn_batch_size = 8
    bottleneck_size = 1024
    lr = 5e-5
    weight_decay = 0.0
    pretraining_imagenet = True
    saveEveryValidation = False
    num_views = 12
    max_epochs = 30
    skip_svcnn = False
    
    def setTest(self):
        self.device = torch.device(f"cpu")
    
    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:0")
        else:
            self.device = torch.device(f"cpu")
            
    def toString(self):
        lr_str = f'lr: {self.lr.__str__()}\n'
        svcnn_batch_size_str = f'svcnn_batch_size: {self.svcnn_batch_size.__str__()}\n'
        mvcnn_batch_size_str = f'mvcnn_batch_size: {self.mvcnn_batch_size.__str__()}\n'
        weight_decay_str = f'weight_decay: {self.weight_decay.__str__()}\n'
        bottleneck_size_str = f'bottleneck_size: {self.bottleneck_size.__str__()}\n'
        
        max_epochs_str = f'max_epochs: {self.max_epochs.__str__()}\n'
        pretraining_imagenet_str = f'pretraining_imagenet: {self.pretraining_imagenet.__str__()}\n'
        num_views_str = f'num_views: {self.num_views.__str__()}\n'
        saveEveryValidation_str = f'saveEveryValidation: {self.saveEveryValidation.__str__()}\n'
        skip_svcnn_str = f'skip_svcnn: {self.skip_svcnn.__str__()}\n'
        
        output = lr_str + svcnn_batch_size_str + mvcnn_batch_size_str + skip_svcnn_str \
            + weight_decay_str + max_epochs_str + pretraining_imagenet_str + num_views_str \
            + saveEveryValidation_str + bottleneck_size_str
            
        return output