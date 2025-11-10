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

def s_x(x):  # [-1,1] → [0,1], C1-flat at ±1
    return (1 - x**2)**2

def s_y(y):
    return (1 - y**2)**2

def S_xy(x, y):
    return s_x(x) * s_y(y)

# def chi_t(t, T=1.0):
#     # 0 at t=0, ≈1 for moderate t; C1 and numerically gentle
#     # you can also use chi_t = t if you prefer
#     eps = 1e-6
#     tau = 0.2  # tune if you like
#     z = jnp.clip(t / T, 0.0, 1.0)
#     return z / (z + tau + eps)

def chi_t(t, T=1.0):
    # return np.clip(t, 0.0, 1.0)           # exact 0 at t=0, 1 at t=1
    return torch.clamp(t, 0.0, 1.0)

def U_exact(x, y, t):
    u = -torch.cos(x) * torch.sin(y) * torch.exp(-2*t)
    v =  torch.sin(x) * torch.cos(y) * torch.exp(-2*t)
    p = -0.25 * (np.cos(2*x) + np.cos(2*y)) * np.exp(-4*t)
    return u, v, p

def tfi_dirichlet_u(x, y, t):
    # edge data from the exact solution (or from your prescribed BC functions)
    UL = U_exact(-1.0, y, t)[0]  # u at left edge
    UR = U_exact(+1.0, y, t)[0]  # u at right edge
    UB = U_exact(x, -1.0, t)[0]  # u at bottom edge
    UT = U_exact(x, +1.0, t)[0]  # u at top edge

    # corner consistency
    UBL = U_exact(-1.0, -1.0, t)[0]
    UBR = U_exact(+1.0, -1.0, t)[0]
    UTL = U_exact(-1.0, +1.0, t)[0]
    UTR = U_exact(+1.0, +1.0, t)[0]

    a = 0.5*(x + 1.0)   # α(x)
    b = 0.5*(y + 1.0)   # β(y)

    # Gordon–Hall TFI on a rectangle
    E  = (1-a)*UL + a*UR + (1-b)*UB + b*UT \
         - ((1-a)*(1-b)*UBL + a*(1-b)*UBR + (1-a)*b*UTL + a*b*UTR)
    return E

def tfi_dirichlet_v(x, y, t):
    # identical structure for v
    VL = U_exact(-1.0, y, t)[1]
    VR = U_exact(+1.0, y, t)[1]
    VB = U_exact(x, -1.0, t)[1]
    VT = U_exact(x, +1.0, t)[1]
    VBL = U_exact(-1.0, -1.0, t)[1]
    VBR = U_exact(+1.0, -1.0, t)[1]
    VTL = U_exact(-1.0, +1.0, t)[1]
    VTR = U_exact(+1.0, +1.0, t)[1]
    a = 0.5*(x + 1.0); b = 0.5*(y + 1.0)
    E  = (1-a)*VL + a*VR + (1-b)*VB + b*VT \
         - ((1-a)*(1-b)*VBL + a*(1-b)*VBR + (1-a)*b*VTL + a*b*VTR)
    return E

def U_exact(x, y, t):
    # print(type(t))
    u = -torch.cos(x) * torch.sin(y) * torch.exp(-2*t)
    v =  torch.sin(x) * torch.cos(y) * torch.exp(-2*t)
    p = -0.25 * (torch.cos(2*x) + torch.cos(2*y)) * torch.exp(-4*t)
    return u, v, p

def tfi_dirichlet_u(x, y, t):
    # edge data from the exact solution (or from your prescribed BC functions)
    UL = U_exact(torch.tensor([-1.0]).cuda(), y, t)[0]  # u at left edge
    UR = U_exact(torch.tensor([+1.0]).cuda(), y, t)[0]  # u at right edge
    UB = U_exact(x, torch.tensor([-1.0]).cuda(), t)[0]  # u at bottom edge
    UT = U_exact(x, torch.tensor([+1.0]).cuda(), t)[0]  # u at top edge

    # corner consistency
    UBL = U_exact(torch.tensor([-1.0]).cuda(), torch.tensor([-1.0]).cuda(), t)[0]
    UBR = U_exact(torch.tensor([+1.0]).cuda(), torch.tensor([-1.0]).cuda(), t)[0]
    UTL = U_exact(torch.tensor([-1.0]).cuda(), torch.tensor([+1.0]).cuda(), t)[0]
    UTR = U_exact(torch.tensor([+1.0]).cuda(), torch.tensor([+1.0]).cuda(), t)[0]

    a = 0.5*(x + 1.0)   # α(x)
    b = 0.5*(y + 1.0)   # β(y)

    # Gordon–Hall TFI on a rectangle
    E  = (1-a)*UL + a*UR + (1-b)*UB + b*UT \
         - ((1-a)*(1-b)*UBL + a*(1-b)*UBR + (1-a)*b*UTL + a*b*UTR)
    return E

def tfi_dirichlet_v(x, y, t):
    # identical structure for v
    VL = U_exact(torch.tensor([-1.0]).cuda(), y, t)[1]
    VR = U_exact(torch.tensor([+1.0]).cuda(), y, t)[1]
    VB = U_exact(x, torch.tensor([-1.0]).cuda(), t)[1]
    VT = U_exact(x, torch.tensor([+1.0]).cuda(), t)[1]
    VBL = U_exact(torch.tensor([-1.0]).cuda(), torch.tensor([-1.0]).cuda(), t)[1]
    VBR = U_exact(torch.tensor([+1.0]).cuda(), torch.tensor([-1.0]).cuda(), t)[1]
    VTL = U_exact(torch.tensor([-1.0]).cuda(), torch.tensor([+1.0]).cuda(), t)[1]
    VTR = U_exact(torch.tensor([+1.0]).cuda(), torch.tensor([+1.0]).cuda(), t)[1]
    a = 0.5*(x + 1.0); b = 0.5*(y + 1.0)
    E  = (1-a)*VL + a*VR + (1-b)*VB + b*VT \
         - ((1-a)*(1-b)*VBL + a*(1-b)*VBR + (1-a)*b*VTL + a*b*VTR)
    return E

def hard_enforce_uv(x, y, t, u_raw, v_raw):
    u0, v0, _ = U_exact(x, y, torch.tensor([0.0]).cuda())
    Eu = tfi_dirichlet_u(x, y, t)
    Ev = tfi_dirichlet_v(x, y, t)
    Eu0 = tfi_dirichlet_u(x, y, torch.tensor([0.0]).cuda())
    Ev0 = tfi_dirichlet_v(x, y, torch.tensor([0.0]).cuda())

    S   = S_xy(x, y)
    c   = chi_t(t, T=1.0)  # or T=config_physical_T

    u = Eu + S * ((1.0 - c) * (u0 - Eu0) + c * u_raw)
    v = Ev + S * ((1.0 - c) * (v0 - Ev0) + c * v_raw)
    return u, v

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
            torch.nn.Linear(3, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 3)
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
            

    def ns(self, train_dataset):
        x = train_dataset[:,0].reshape(-1,1).clone().detach().requires_grad_(True)
        y = train_dataset[:,1].reshape(-1,1).clone().detach().requires_grad_(True)
        t = train_dataset[:,2].reshape(-1,1).clone().detach().requires_grad_(True)
        # print(type(t))

        out_raw = self.forward(torch.cat((x, y, t), dim=1))
        # print('out',out_raw.shape)

        u_raw, v_raw, p = out_raw[:, 0:1], out_raw[:, 1:2], out_raw[:, 2:3]
        # print(type(t))
        u, v = hard_enforce_uv(x, y, t, u_raw, v_raw)
        # print('u', u.shape)
   
        
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

        v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]
        v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_y), create_graph=True)[0]


        p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        p_y = torch.autograd.grad(p, y, grad_outputs=torch.ones_like(p), create_graph=True)[0]
      
        f_u = u_t + (u*u_x + v*u_y) + p_x - (u_xx + u_yy)
        f_v = v_t + (u*v_x + v*v_y) + p_y - (v_xx + v_yy)
        f_c = u_x + v_y  # Continuity
        
        eqn = f_u**2 + f_v**2 + f_c**2

       
        return eqn


    def loss_function(self, train_dataset):
        x = train_dataset[:,0].reshape(-1,1).clone().detach().requires_grad_(True)
        y = train_dataset[:,1].reshape(-1,1).clone().detach().requires_grad_(True)
        t = train_dataset[:,2].reshape(-1,1).clone().detach().requires_grad_(True)
        # a = torch.zeros((20849,1))
        # x_bc1 = torch.ones_like(a).cuda()
        # x_bc2 = -torch.ones_like(a).cuda()
        # y_bc1 = torch.ones_like(a).cuda()
        # y_bc2 = -torch.ones_like(a).cuda()

        loss_pde = torch.mean(self.ns(train_dataset) ** 2)

        # loss_bc = (torch.mean(self.forward(torch.cat((x_bc1, y), dim=1)) ** 2) +
        #          torch.mean(self.forward(torch.cat((x_bc2, y), dim=1)) ** 2) +
        #         torch.mean(self.forward(torch.cat((x, y_bc1), dim=1)) ** 2) +
        #         torch.mean(self.forward(torch.cat((x, y_bc2), dim=1)) ** 2))
        return loss_pde

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