# import numpy as np
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class STN2D(nn.Module):

#     def __init__(self, input_size, input_channels, device):
#         super(STN2D, self).__init__()
#         num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2)
#         self.device = device
#         self.conv1 = nn.Conv2d(input_channels, 8, kernel_size=5)
#         self.conv2 = nn.Conv2d(8, 16, kernel_size=5)
#         self.conv3 = nn.Conv2d(16, 32, kernel_size=5)
#         self.fc = nn.Linear(int(32 * num_features), 32)

#         self.theta = torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float).view(2, 3)

#         # Regressor for the 2 * 3 affine matrix
#         # self.affine_regressor = nn.Linear(32, 2 * 3)

#         # initialize the weights/bias with identity transformation
#         # self.affine_regressor.weight.data.zero_()
#         # self.affine_regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

#         # Regressor for individual parameters
#         self.translation = nn.Linear(32, 2)
#         self.rotation = nn.Linear(32, 1)
#         self.scaling = nn.Linear(32, 2)
#         self.shearing = nn.Linear(32, 1)

#         # initialize the weights/bias with identity transformation
#         self.translation.weight.data.zero_()
#         self.translation.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
#         self.rotation.weight.data.zero_()
#         self.rotation.bias.data.copy_(torch.tensor([0], dtype=torch.float))
#         self.scaling.weight.data.zero_()
#         self.scaling.bias.data.copy_(torch.tensor([0, 0], dtype=torch.float))
#         self.shearing.weight.data.zero_()
#         self.shearing.bias.data.copy_(torch.tensor([0], dtype=torch.float))

#     def forward(self, x):
#         xs = F.avg_pool2d(F.relu(self.conv1(x)), 2)
#         xs = F.avg_pool2d(F.relu(self.conv2(xs)), 2)
#         xs = F.avg_pool2d(F.relu(self.conv3(xs)), 2)
#         xs = xs.view(xs.size(0), -1)
#         xs = F.relu(self.fc(xs))
#         # theta = self.affine_regressor(xs).view(-1, 2, 3)
#         self.theta = self.affine_matrix(xs)

#         # extract first channel for warping
#         img = x.narrow(dim=1, start=0, length=1)

#         # warp image
#         return self.warp_image(img)

#     def warp_image(self, img):
#         grid = F.affine_grid(self.theta, img.size()).to(self.device)
#         wrp = F.grid_sample(img, grid)

#         return wrp

#     def affine_matrix(self, x):
#         b = x.size(0)

#         # trans = self.translation(x)
#         trans = torch.tanh(self.translation(x)) * 0.1
#         translation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
#         translation_matrix[:, 0, 0] = 1.0
#         translation_matrix[:, 1, 1] = 1.0
#         translation_matrix[:, 0, 2] = trans[:, 0].view(-1)
#         translation_matrix[:, 1, 2] = trans[:, 1].view(-1)
#         translation_matrix[:, 2, 2] = 1.0

#         # rot = self.rotation(x)
#         rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
#         rotation_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
#         rotation_matrix[:, 0, 0] = torch.cos(rot.view(-1))
#         rotation_matrix[:, 0, 1] = -torch.sin(rot.view(-1))
#         rotation_matrix[:, 1, 0] = torch.sin(rot.view(-1))
#         rotation_matrix[:, 1, 1] = torch.cos(rot.view(-1))
#         rotation_matrix[:, 2, 2] = 1.0

#         # scale = F.softplus(self.scaling(x), beta=np.log(2.0))
#         # scale = self.scaling(x)
#         scale = torch.tanh(self.scaling(x)) * 0.2
#         scaling_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
#         # scaling_matrix[:, 0, 0] = scale[:, 0].view(-1)
#         # scaling_matrix[:, 1, 1] = scale[:, 1].view(-1)
#         scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
#         scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
#         scaling_matrix[:, 2, 2] = 1.0

#         # shear = self.shearing(x)
#         shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)
#         shearing_matrix = torch.zeros([b, 3, 3], dtype=torch.float)
#         shearing_matrix[:, 0, 0] = torch.cos(shear.view(-1))
#         shearing_matrix[:, 0, 1] = -torch.sin(shear.view(-1))
#         shearing_matrix[:, 1, 0] = torch.sin(shear.view(-1))
#         shearing_matrix[:, 1, 1] = torch.cos(shear.view(-1))
#         shearing_matrix[:, 2, 2] = 1.0

#         # Affine transform
#         matrix = torch.bmm(shearing_matrix, scaling_matrix)
#         matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
#         matrix = torch.bmm(matrix, rotation_matrix)
#         matrix = torch.bmm(matrix, translation_matrix)

#         # matrix = torch.bmm(translation_matrix, rotation_matrix)
#         # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
#         # matrix = torch.bmm(matrix, scaling_matrix)
#         # matrix = torch.bmm(matrix, shearing_matrix)

#         # No-shear transform
#         # matrix = torch.bmm(scaling_matrix, rotation_matrix)
#         # matrix = torch.bmm(matrix, translation_matrix)

#         # Rigid-body transform
#         # matrix = torch.bmm(rotation_matrix, translation_matrix)

#         return matrix[:, 0:2, :]












import torch
import torch.nn as nn
import torch.nn.functional as F


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 3 * 3, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, xs.shape[0] * xs.shape[1] * xs.shape[2] * xs.shape[3])
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)