import torch
import torch.optim as optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import time

import UNO_conversion.PINNmodels_pytorch_conversion2d as PINNmodels_pytorch_conversion
import logging
import os
import unittest
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import LambdaLR
import math
from torch.optim.lr_scheduler import MultiStepLR
import jax


# from pytorch.src.conversion.convert import convert_ann_to_snn
# from pytorch.src.conversion.convert import save_converted_model

from UNO_conversion.pytorch.src.data import TrainDataset
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Running on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

import gc

# Add this at the start of your script and between training runs
torch.cuda.empty_cache()
gc.collect()

TEST_FOLDER = (
    "/users/rwan5/reproduce_poisson/UNO-conversion/pytorch/tests"
)

# logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s, %(levelname)s:     %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S",
)
logger = logging.getLogger()
logger.setLevel(15)

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
    return np.clip(t, 0.0, 1.0)           # exact 0 at t=0, 1 at t=1

def U_exact(x, y, t):
    u = -jnp.cos(x) * jnp.sin(y) * jnp.exp(-2*t)
    v =  jnp.sin(x) * jnp.cos(y) * jnp.exp(-2*t)
    p = -0.25 * (jnp.cos(2*x) + jnp.cos(2*y)) * jnp.exp(-4*t)
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
    u = -np.cos(x) * np.sin(y) * np.exp(-2*t)
    v =  np.sin(x) * np.cos(y) * np.exp(-2*t)
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

def hard_enforce_uv(x, y, t, u_raw, v_raw):
    u0, v0, _ = U_exact(x, y, 0.0)
    Eu = tfi_dirichlet_u(x, y, t)
    Ev = tfi_dirichlet_v(x, y, t)
    Eu0 = tfi_dirichlet_u(x, y, 0.0)
    Ev0 = tfi_dirichlet_v(x, y, 0.0)

    S   = S_xy(x, y)
    c   = chi_t(t, T=1.0)  # or T=config_physical_T

    u = Eu + S * ((1.0 - c) * (u0 - Eu0) + c * u_raw)
    v = Ev + S * ((1.0 - c) * (v0 - Ev0) + c * v_raw)
    return u, v

    
nx, ny, nt = 30,30,30

key = jax.random.PRNGKey(42)
keys = jax.random.split(key, 13)

# Collocation points
t = jax.random.uniform(keys[0], (nt, 1), minval=0., maxval=1.)
x = jax.random.uniform(keys[1], (nx, 1), minval=-1., maxval=1.)
y = jax.random.uniform(keys[2], (ny, 1), minval=-1., maxval=1.)

# Create 3D meshgrid
X, Y, T = np.meshgrid(x, y, t, indexing='ij')

# Flatten to get all points
x_train = X.flatten().reshape(-1, 1)
y_train = Y.flatten().reshape(-1, 1)
t_train = T.flatten().reshape(-1, 1)

u_train = -np.cos(x_train)*np.sin(y_train)*np.exp(-2*t_train)
v_train = np.sin(x_train)*np.cos(y_train)*np.exp(-2*t_train)
p_train = -0.25*(np.cos(2*x_train)+np.cos(2*y_train))*np.exp(-4*t_train)

# Test data
nx_test, ny_test, nt_test = 40, 40, 40

x_test = np.linspace(-1, 1, nx_test)
y_test = np.linspace(-1, 1, ny_test)
t_test = np.linspace(0, 1, nt_test)

X_test, Y_test, T_test = np.meshgrid(x_test, y_test, t_test, indexing='ij')

x_test = X_test.flatten().reshape(-1, 1)
y_test = Y_test.flatten().reshape(-1, 1)
t_test = T_test.flatten().reshape(-1, 1)

u_test = -np.cos(x_test)*np.sin(y_test)*np.exp(-2*t_test)
v_test = np.sin(x_test)*np.cos(y_test)*np.exp(-2*t_test)
p_test = -0.25*(np.cos(2*x_test)+np.cos(2*y_test))*np.exp(-4*t_test)




test_input = torch.cat((torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32), torch.tensor(t_test, dtype=torch.float32)), dim=1)
test_label = torch.cat((torch.tensor(u_test, dtype=torch.float32), torch.tensor(v_test, dtype=torch.float32), torch.tensor(p_test, dtype=torch.float32)), dim=1)


x = torch.tensor(x_train, dtype=torch.float32)
y = torch.tensor(y_train, dtype=torch.float32)
t = torch.tensor(t_train, dtype=torch.float32)
u = torch.tensor(u_train, dtype=torch.float32)
v = torch.tensor(v_train, dtype=torch.float32)
p = torch.tensor(p_train, dtype=torch.float32)

train_input = torch.cat((x, y, t), dim=1)
train_label = torch.cat((u, v, p), dim=1)




# Create a TensorDataset and DataLoader for minibatching
train_dataset = TensorDataset(train_input, train_label)
batch_size = 900
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = PINNmodels_pytorch_conversion.PINNPyTorch().to(device)  # Move the model to GPU if available

optimizer = optim.Adam(model.parameters(), lr=1e-3)



init_lr = 1e-3  # Your initial learning rate

# optimizer = AdaBelief(
#     model.parameters(),
#     lr=init_lr,
#     eps=1e-16,
#     betas=(0.9, 0.999),
#     weight_decay=0,
#     weight_decouple=True,
#     rectify=True,
#     print_change_log=False
# )

# Create scheduler
scheduler = MultiStepLR(
    optimizer,
    milestones=[30*350],  # Steps where LR changes: [45000, 90000]
    gamma=0.1  # Multiply by 0.1 at each milestone
)

# def exponential_decay(step):
#     """
#     Mimics optax.exponential_decay(init_lr, 2000, 1/e)
#     lr(step) = init_lr * (1/e)^(step/2000)
#     """
#     return init_lr * (decay_rate ** (step / transition_steps))

# scheduler = LambdaLR(optimizer, lr_lambda=exponential_decay)

# Training loop
niter = 1500
# timing
times=[]
train_loss = []

global_step = 0
for epoch in range(niter):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_x, batch_u in train_loader:
        batch_x = batch_x.to(device)
        batch_u = batch_u.to(device)
        
        st=time.time()
        optimizer.zero_grad()
        loss = model.loss_function(batch_x)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update LR after each batch
        et=time.time()
        times.append(et-st)
        global_step += 1
        
        epoch_loss += loss.item()
        num_batches += 1
    
    avg_loss = epoch_loss / num_batches
    train_loss.append(loss.detach().cpu().numpy())
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Average Loss {avg_loss:.6f}")



# Evaluation
model.eval()
with torch.no_grad():
    
    x_full = test_input.cuda()
    out_raw = model(x_full).cpu().numpy()
    u_raw, v_raw, p = out_raw[:, 0:1], out_raw[:, 1:2], out_raw[:, 2:3]
    u_pred, v_pred = hard_enforce_uv(x_test, y_test, t_test, u_raw, v_raw)
    
    



PINN_result = {
    'x': np.linspace(-1, 1, nx_test),
    'y': np.linspace(-1, 1, ny_test),
    't': np.linspace(0, 1, nt_test),
    'u_true': u_test.reshape(40,40,40),
    'u_pred': u_pred.reshape(40,40,40),
    'v_true': v_test.reshape(40,40,40),
    'v_pred': v_pred.reshape(40,40,40),
    'time': times,
    'loss': train_loss
}



# print(u_pred,shape)

# Save the data to a .mat file
sio.savemat('conversion_ann_pinn_ns.mat', PINN_result)

torch.save(model.state_dict(), 'annmodel.pth')
print(model.state_dict())
print(u_pred.shape)
ann = PINNmodels_pytorch_conversion.PINNPyTorch()
annmodel = ann.load_state_dict(torch.load('annmodel.pth'))
ann.eval()

from UNO_conversion.pytorch.src.conversion.convert import convert_ann_to_snn
from UNO_conversion.pytorch.src.conversion.convert import save_converted_model

ann.layers[-1].is_output = True
# train_dataset = np.linspace(0, 1, 1000).reshape(-1, 1)
train_dataset = torch.tensor(train_input, dtype = torch.float32)


cfg_conversion = {
            "activation": "Tanh", 
            "calib": None,
            "calib_arb_act": True,
            "usebn": False,
            "seed": 0,
            "batch_size":900,
            "T": 128,
        }
snn = convert_ann_to_snn(cfg_conversion, ann.layers.cpu(), train_dataset.cpu())

# x_test = torch.tensor(np.linspace(0, 1, 5000).reshape(-1,1), dtype=torch.float32)
x_full = test_input.cuda()
# u_raw = snn(x_full).cpu().numpy()
# CORRECT - Process in chunks
chunk_size = 5000  # Adjust based on your memory

# # Keep test_input on CPU
if not isinstance(test_input, torch.Tensor):
    test_input = torch.tensor(test_input, dtype=torch.float32)

n_samples = test_input.shape[0]
predictions_list = []

snn.eval()
with torch.no_grad():
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        
        # Load only this chunk to GPU
        x_chunk = test_input[i:end_idx].cuda()
        
        # Forward pass
        pred_chunk = snn(x_chunk)
        
        # Move to CPU immediately
        predictions_list.append(pred_chunk.cpu())
        
        # Critical: delete GPU tensors
        del x_chunk, pred_chunk
        torch.cuda.empty_cache()

# Concatenate results on CPU
u_raw = torch.cat(predictions_list, dim=0).numpy()

# print(f"u_raw shape: {u_raw.shape}")
u_raw, v_raw, p = out_raw[:, 0:1], out_raw[:, 1:2], out_raw[:, 2:3]
u_pred, v_pred = hard_enforce_uv(x_test, y_test, t_test, u_raw, v_raw)

print(u_pred.shape)


torch.save(snn.state_dict(), 'snnmodel.pth')
sio.savemat('conversion_snn_pinn_ns.mat', {'u_pred': u_pred.reshape(40,40,40), 'v_pred': v_pred.reshape(40,40,40)})

from UNO_conversion.pytorch.src.conversion.convert import convert_ann_to_snn
from UNO_conversion.pytorch.src.conversion.convert import save_converted_model

ann.layers[-1].is_output = True
cfg_conversion = {
            "activation": "Tanh",
            "calib": "advanced",
            "calib_arb_act": True,
            "usebn": False,
            "seed": 0,
            "batch_size": 900,
            "T": 128,
            "wd": 1e-4
        }
snn = convert_ann_to_snn(cfg_conversion, ann.layers.cpu(), train_dataset.cpu())

# x_test = torch.tensor(np.linspace(0, 1, 5000).reshape(-1,1), dtype=torch.float32)
# x_full = x_test.cuda()
x_full = test_input.cuda()
# u_raw = snn(x_full).cpu().numpy()

# CORRECT - Process in chunks
chunk_size = 5000  # Adjust based on your memory

# # Keep test_input on CPU
if not isinstance(test_input, torch.Tensor):
    test_input = torch.tensor(test_input, dtype=torch.float32)

n_samples = test_input.shape[0]
predictions_list = []

snn.eval()
with torch.no_grad():
    for i in range(0, n_samples, chunk_size):
        end_idx = min(i + chunk_size, n_samples)
        
        # Load only this chunk to GPU
        x_chunk = test_input[i:end_idx].cuda()
        
        # Forward pass
        pred_chunk = snn(x_chunk)
        
        # Move to CPU immediately
        predictions_list.append(pred_chunk.cpu())
        
        # Critical: delete GPU tensors
        del x_chunk, pred_chunk
        torch.cuda.empty_cache()

# Concatenate results on CPU
u_raw = torch.cat(predictions_list, dim=0).numpy()

u_raw, v_raw, p = out_raw[:, 0:1], out_raw[:, 1:2], out_raw[:, 2:3]
u_pred, v_pred = hard_enforce_uv(x_test, y_test, t_test, u_raw, v_raw)
print(u_pred.shape)


torch.save(snn.state_dict(), 'snnmodel_cali.pth')
sio.savemat('conversion_cali_snn_pinn_ns.mat', {'u_pred': u_pred.reshape(40,40,40), 'v_pred': v_pred.reshape(40,40,40)})


# output = (x_full*(1-x_full)*snn(x_full.cuda())).cpu().numpy()
# print(output.max())




# # Evaluation
# model.eval()
# with torch.no_grad():
#     x_full = test_input.cuda()
#     u_raw = model(x_full)
#     time_mult = torch.exp(-3.0*x_full[:, 1]) 
#     boundary_mult = (-1-x_full[:, 0]) * (1-x_full[:, 0])
#     u_ic = -torch.sin(torch.pi * x_full[:, 0])
#     u_pred = u_ic.reshape(-1,1) * time_mult.reshape(-1,1) + x_full[:, 1].reshape(-1,1) * boundary_mult.reshape(-1,1) * u_raw



# PINN_result = {
#     'x': x_full.detach().cpu().numpy(),
#     'u_true': u_test,
#     'u_pred': u_pred.cpu().numpy(),
#     'time': times,
#     'loss': train_loss
    
# }


# print(u_pred)

# # Save the data to a .mat file
# sio.savemat('conversion_ann_pinn_burger.mat', PINN_result)


# print(model)
# print(model.layers)