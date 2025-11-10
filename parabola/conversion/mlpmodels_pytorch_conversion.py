# import tensorflow as tf
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Running on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")


class PINNPyTorch(nn.Module):
    """Physics-informed neural networks for 2d poisson in PyTorch"""

    def __init__(self, eps=1, lr=1e-3, name="pinn"):
        super(PINNPyTorch, self).__init__()
#         self.activation = activation
#         self.layers = nn.ModuleList()
#         self.layer1 = nn.Linear(2, 100)
#         self.layer2 = nn.Linear(100, 100)
#         self.layer3 = nn.Linear(100, 100)
#         self.layer4 = nn.Linear(100, 1)
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(1, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 1)
        )

    
        
#         for i in range(len(layers) - 2):
#             self.layers.append(nn.Linear(layers[i], layers[i + 1]))
#             self.layers.append(activation)
#         self.layers.append(nn.Linear(layers[-2], layers
#         [-1]))
        # self.opt = optim.Adam(self.parameters(), lr=lr)
        self.name = name

    def forward(self, train_dataset):
#         x = train_dataset[:,0].reshape(-1,1)
#         y = train_dataset[:,1].reshape(-1,1)
#         x = torch.tensor(x, dtype=torch.float32, requires_grad=True)
#         y = torch.tensor(y, dtype=torch.float32, requires_grad=True)
#         mesh = torch.cat([x, y], dim=1)
#         mesh = train_dataset
#         for layer in self.layers:
#             mesh = layer(mesh)
#         mesh = self.layer1(mesh)
#         mesh = torch.tanh(mesh)
#         mesh = self.layer2(mesh)
#         mesh = torch.tanh(mesh)
#         mesh = self.layer3(mesh)
#         mesh = torch.tanh(mesh)
#         mesh = self.layer4(mesh)

        

        return self.layers(train_dataset)
            

    def poisson(self, train_dataset):
        x = train_dataset[:,0].reshape(-1,1).clone().detach().requires_grad_(True)

        u = x*(1-x)*(self.forward(x))
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        eqn = -u_xx - 2*((torch.pi)**2)*(torch.sin((np.pi)*x))
        return eqn


    def loss_function(self, train_dataset):
        x = train_dataset[:,0].reshape(-1,1).clone().detach().requires_grad_(True)
        # a = torch.zeros((20849,1))
        # x_bc1 = torch.ones_like(a).cuda()
        # x_bc2 = -torch.ones_like(a).cuda()
        # y_bc1 = torch.ones_like(a).cuda()
        # y_bc2 = -torch.ones_like(a).cuda()

        # loss_pde = torch.mean(self.poisson(train_dataset) ** 2)
        label = x**2
        loss = torch.mean((self.forward(x)-label)**2)

        # loss_bc = (torch.mean(self.forward(torch.cat((x_bc1, y), dim=1)) ** 2) +
        #          torch.mean(self.forward(torch.cat((x_bc2, y), dim=1)) ** 2) +
        #         torch.mean(self.forward(torch.cat((x, y_bc1), dim=1)) ** 2) +
        #         torch.mean(self.forward(torch.cat((x, y_bc2), dim=1)) ** 2))
        return loss

# def train(self, x, y, x_bc1, x_bc2, y_bc1, y_bc2, niter=50000):
#     x = torch.tensor(x, dtype=torch.float32)
#     y = torch.tensor(y, dtype=torch.float32)
#     x_bc1 = torch.tensor(x_bc1, dtype=torch.float32)
#     x_bc2 = torch.tensor(x_bc2, dtype=torch.float32)
#     y_bc1 = torch.tensor(y_bc1, dtype=torch.float32)
#     y_bc2 = torch.tensor(y_bc2, dtype=torch.float32)

#     for it in range(niter):
#         self.opt.zero_grad()
#         loss = self.loss_function(x, y, x_bc1, x_bc2, y_bc1, y_bc2)
#         loss.backward()
#         self.opt.step()

#         if it % 1000 == 0:
#             print(f"Iteration {it}: Loss = {loss.item()}") 