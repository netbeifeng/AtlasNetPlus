import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from model.template import get_template
import torch.nn.functional as F
import open3d
import numpy as np

class MLPCore(nn.Module):
    """
    Core Atlasnet Function.
    Takes batched points as input and run them through an MLP.
    Note : the MLP is implemented as a torch.nn.Conv1d with kernels of size 1 for speed.
    Note : The latent vector is added as a bias after the first layer. Note that this is strictly identical
    as concatenating each input point with the latent vector but saves memory and speeed.
    Author : Thibault Groueix 01.11.2019
    """

    def __init__(self, hparams):

        self.bottleneck_size = hparams.bottleneck_size
        self.input_size = hparams.dim_template
        self.dim_output = 3
        self.hidden_neurons = hparams.hidden_neurons
        self.num_layers = hparams.num_layers
        super(MLPCore, self).__init__()
        print(
            f"New MLP decoder : hidden size {hparams.hidden_neurons}, num_layers {hparams.num_layers}, activation {hparams.activation}")

        self.conv1 = torch.nn.Conv1d(self.input_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, self.hidden_neurons, 1)

        self.conv_list = nn.ModuleList(
            [torch.nn.Conv1d(self.hidden_neurons, self.hidden_neurons, 1) for i in range(self.num_layers)])

        self.last_conv = torch.nn.Conv1d(self.hidden_neurons, self.dim_output, 1)

        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.hidden_neurons)

        self.bn_list = nn.ModuleList([torch.nn.BatchNorm1d(self.hidden_neurons) for i in range(self.num_layers)])

        self.activation = get_activation(hparams.activation)

    def forward(self, x, latent):
        x = self.conv1(x) + latent
        x = self.activation(self.bn1(x))
        x = self.activation(self.bn2(self.conv2(x)))
        for i in range(self.num_layers):
            x = self.activation(self.bn_list[i](self.conv_list[i](x)))
        return self.last_conv(x)
    
class Atlasnet(torch.nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        super(Atlasnet, self).__init__()
        self.device = hparams.device

        # Define number of points per primitives
        self.nb_pts_in_primitive = hparams.num_points // hparams.num_prims
        self.nb_pts_in_primitive_eval = hparams.num_points // hparams.num_prims

        print(f"Every prim has {self.nb_pts_in_primitive_eval} points")
        
        # Initialize templates
        # Patches or Spheres, each one represent one part of 3d 
        self.template = [get_template(hparams.template_type, device=self.device) for i in range(0, hparams.num_prims)]

        # Intialize deformation networks
        self.decoder = nn.ModuleList(
            [MLPCore(hparams) for i in range(0, hparams.num_prims)]
        )

    def forward(self, latent_vector, train=True):
        """
        Deform points from self.template using the embedding latent_vector
        :param latent_vector: an hparams.bottleneck size vector encoding a 3D shape or an image. size : batch, bottleneck
        :return: A deformed pointcloud os size : batch, nb_prim, num_point, 3
        """
        if train:
            input_points = [self.template[i].get_random_points(
                torch.Size((1, self.template[i].dim, self.nb_pts_in_primitive)),
                latent_vector.device) for i in range(self.hparams.num_prims)]
        else:
            input_points = [self.template[i].get_regular_points(self.nb_pts_in_primitive_eval,
                                                                device=latent_vector.device)
                            for i in range(self.hparams.num_prims)]

        # Deform each patch
        output_points = torch.cat([self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).unsqueeze(1) 
                                   for i in range(self.hparams.num_prims)], dim=1)
        
        # print(output_points.shape) # N, npoints, 3, num_prims => 2000, 3, 1
        # Return the deformed pointcloud    
        return output_points.contiguous()  # batch, nb_prim, nb_pts_in_primitive, 3

    def generate_mesh(self, latent_vector):
        assert latent_vector.size(0)==1, "input should have batch size 1!"

        input_points = []
        for i in range(self.hparams.num_prims):
            sampled_points = self.template[i].get_regular_points(self.nb_pts_in_primitive, latent_vector.device)
            input_points.append(sampled_points)

        input_points = [input_points[i] for i in range(self.hparams.num_prims)]

        # Deform each patch
        output_points = []
        for i in range(0, self.hparams.num_prims):
            # Decoder to forward
            self.decoder[i].eval()
            learned_points = self.decoder[i](input_points[i], latent_vector.unsqueeze(2)).squeeze()
            output_points.append(learned_points)

        output_meshes = open3d.geometry.TriangleMesh()
        for i in range(self.hparams.num_prims):
            nd_vertices = output_points[i].transpose(1, 0).contiguous().cpu().detach().numpy()
            nd_faces = np.asarray(self.template[i].mesh.triangles).astype(np.int32)
            mesh = open3d.geometry.TriangleMesh()
            mesh.vertices = open3d.utility.Vector3dVector(nd_vertices)
            mesh.triangles = open3d.utility.Vector3iVector(nd_faces)
            output_meshes += mesh

        # Deform return the deformed pointcloud
        mesh = output_meshes
        
        return mesh
    
def get_activation(argument):
    getter = {
        "relu": F.relu,
        "sigmoid": F.sigmoid,
        "softplus": F.softplus,
        "logsigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
    }
    return getter.get(argument, "Invalid activation")
