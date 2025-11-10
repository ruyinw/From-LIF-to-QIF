import torch
import torch.optim as optim
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

import scipy.io as sio
import time

import UNO_conversion.mlpmodels_pytorch_conversion as PINNmodels_pytorch_conversion
import logging
import os
import unittest
from torch.utils.data import DataLoader, TensorDataset
from adabelief_pytorch import AdaBelief
from torch.optim.lr_scheduler import LambdaLR
import math

# from pytorch.src.conversion.convert import convert_ann_to_snn
# from pytorch.src.conversion.convert import save_converted_model

from UNO_conversion.pytorch.src.data import TrainDataset
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Running on GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available. Running on CPU.")

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

# data = sio.loadmat("PDE_solution_data_20849.mat")
x = np.linspace(-1, 1, 100)
u = x**2
x = x.reshape(-1, 1)
u = u.reshape(-1, 1)

x = torch.tensor(x, dtype=torch.float32)
u = torch.tensor(u, dtype=torch.float32)


# Create a TensorDataset and DataLoader for minibatching
train_dataset = TensorDataset(x, u)
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = PINNmodels_pytorch_conversion.PINNPyTorch().to(device)  # Move the model to GPU if available
# optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Parameters matching optax.exponential_decay(lr, 2000, 1/e)
init_lr = 1e-3  # Your initial learning rate
transition_steps = 2000
decay_rate = 1 / math.e  # approximately 0.3679

optimizer = AdaBelief(
    model.parameters(),
    lr=init_lr,
    eps=1e-16,
    betas=(0.9, 0.999),
    weight_decay=0,
    weight_decouple=True,
    rectify=True,
    print_change_log=False
)

def exponential_decay(step):
    """
    Mimics optax.exponential_decay(init_lr, 2000, 1/e)
    lr(step) = init_lr * (1/e)^(step/2000)
    """
    return init_lr * (decay_rate ** (step / transition_steps))

scheduler = LambdaLR(optimizer, lr_lambda=exponential_decay)

# Training loop
niter = 10000
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
        # scheduler.step()  # Update LR after each batch
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
    x_test = torch.tensor(np.linspace(-1, 1, 1000).reshape(-1,1), dtype=torch.float32)
    x_full = x_test.cuda()
    u_pred = model(x_full)
    u = x_test**2
  


PINN_result = {
    'x': x_full.detach().cpu().numpy(),
    'u_ref': u.detach().cpu().numpy(),
    'u_pred': u_pred.cpu().numpy(),
    'time': times,
    'loss': train_loss
}



print(u_pred)

# Save the data to a .mat file
sio.savemat('conversion_ann_parabola.mat', PINN_result)

torch.save(model.state_dict(), 'annmodel.pth')
# print(model.state_dict())
# print(u_pred.shape)
ann = PINNmodels_pytorch_conversion.PINNPyTorch()
annmodel = ann.load_state_dict(torch.load('annmodel.pth'))
ann.eval()

from UNO_conversion.pytorch.src.conversion.convert import convert_ann_to_snn
from UNO_conversion.pytorch.src.conversion.convert import save_converted_model

ann.layers[-1].is_output = True
train_dataset = np.linspace(-1, 1, 1000).reshape(-1, 1)
train_dataset = torch.tensor(train_dataset, dtype = torch.float32)

cfg_conversion = {
            "activation": "Tanh", 
            "calib": None,
            "calib_arb_act": True,
            "usebn": False,
            "seed": 0,
            "batch_size":100,
            "T": 32,
        }
snn = convert_ann_to_snn(cfg_conversion, ann.layers.cpu(), train_dataset.cpu())

x_test = torch.tensor(np.linspace(-1, 1, 1000).reshape(-1,1), dtype=torch.float32)
x_full = x_test.cuda()
# print((x_full*(1-x_full)*snn(x_full.cuda())).shape)


torch.save(snn.state_dict(), 'snnmodel.pth')
sio.savemat('conversion_snn_parabola.mat', {'u_pred': snn(x_full.cuda()).cpu().numpy()})

from UNO_conversion.pytorch.src.conversion.convert import convert_ann_to_snn
from UNO_conversion.pytorch.src.conversion.convert import save_converted_model

ann.layers[-1].is_output = True
cfg_conversion = {
            "activation": "Tanh",
            "calib": "advanced",
            "calib_arb_act": True,
            "usebn": False,
            "seed": 0,
            "batch_size": 100,
            "T": 32,
            "wd": 1e-4
        }
snn = convert_ann_to_snn(cfg_conversion, ann.layers.cpu(), train_dataset.cpu())


torch.save(snn.state_dict(), 'snnmodel_cali.pth')
sio.savemat('conversion_cali_snn_parabola.mat', {'u_pred': snn(x_full.cuda()).cpu().numpy()})


# output = (x_full*(1-x_full)*snn(x_full.cuda())).cpu().numpy()
# print(output.max())




# Evaluation
model.eval()
with torch.no_grad():
    x_test = torch.tensor(np.linspace(-1, 1, 1000).reshape(-1,1), dtype=torch.float32)
    x_full = x_test.cuda()
    u_pred = model(x_full)
    u = x_full**2
   


PINN_result = {
    'x': x_full.detach().cpu().numpy(),
    'u_true': u.detach().cpu().numpy(),
    'u_pred': u_pred.cpu().numpy(),
    'time': times,
    'loss': train_loss
    
}


print(u_pred)

# Save the data to a .mat file
sio.savemat('conversion_ann_parabola.mat', PINN_result)


print(model)
print(model.layers)