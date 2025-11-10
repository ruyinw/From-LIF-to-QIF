from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import torch
from jax import jit, random, value_and_grad, vmap, vjp, jvp
from jaxtyping import Array, ArrayLike, Float, Int, UInt8
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange as trange_script
import time
import scipy.io as sio
import numpy as np
import os
import pickle
import equinox as eqx
from jax.scipy.optimize import minimize
from jaxopt import LBFGS
from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron
from spikegd.utils.plotting import formatter, petroff10

os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

# %%
############################
### Batched Derivatives
############################

def s_x(x):  # [-1,1] → [0,1], C1-flat at ±1
    return (1 - x**2)**2

def s_y(y):
    return (1 - y**2)**2

def S_xy(x, y):
    return s_x(x) * s_y(y)

def chi_t(t, T=1.0):
    return jnp.clip(t, 0.0, 1.0)           # exact 0 at t=0, 1 at t=1

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


def hvp_fwdrev(f, primals, tangents, return_primals=False):
    """Hessian-vector product using forward-over-reverse mode"""
    g = lambda primals: vjp(f, primals)[1](tangents[0])[0]
    primals_out, tangents_out = jvp(g, primals, tangents)
    if return_primals:
        return primals_out, tangents_out
    else:
        return tangents_out


def compute_derivatives_batched(neuron, p, config, inputs_batch):
    """
    Compute derivatives for a BATCH of points using JVP/HVP.
    
    Args:
        inputs_batch: shape (N, 3) where N is batch size
    
    Returns:
        All outputs have shape (N,)
    """
    
    T = config["T"]
    scale_x = -T / 2
    scale_y = -T / 2
    scale_t = -T
    
    x = inputs_batch[:, 0]
    y = inputs_batch[:, 1]
    t = inputs_batch[:, 2]
    
    vv_x = jnp.ones_like(x)
    vv_y = jnp.ones_like(y)
    vv_t = jnp.ones_like(t)
    
    def apply_fn_batch(x_vals, y_vals, t_vals):
        """Apply boundary conditions to batch of points"""
        def single_apply(x_val, y_val, t_val):
            xyt_vec = jnp.array([x_val, y_val, t_val])
            return apply_boundary_conditions(neuron, p, config, xyt_vec)
        
        u_batch, v_batch, pre_batch = vmap(single_apply)(x_vals, y_vals, t_vals)
        return u_batch, v_batch, pre_batch
    
    u, v, pre = apply_fn_batch(x, y, t)
    
    # First derivatives of u
    def u_fn_x(x_vals):
        u_batch, _, _ = apply_fn_batch(x_vals, y, t)
        return u_batch
    
    u_x = jvp(u_fn_x, (x,), (vv_x,))[1] * scale_x
    
    def u_fn_y(y_vals):
        u_batch, _, _ = apply_fn_batch(x, y_vals, t)
        return u_batch
    
    u_y = jvp(u_fn_y, (y,), (vv_y,))[1] * scale_y
    
    def u_fn_t(t_vals):
        u_batch, _, _ = apply_fn_batch(x, y, t_vals)
        return u_batch
    
    u_t = jvp(u_fn_t, (t,), (vv_t,))[1] * scale_t
    
    # Second derivatives of u
    u_xx = hvp_fwdrev(u_fn_x, (x,), (vv_x,)) * (scale_x ** 2)
    u_yy = hvp_fwdrev(u_fn_y, (y,), (vv_y,)) * (scale_y ** 2)
    
    # First derivatives of v
    def v_fn_x(x_vals):
        _, v_batch, _ = apply_fn_batch(x_vals, y, t)
        return v_batch
    
    v_x = jvp(v_fn_x, (x,), (vv_x,))[1] * scale_x
    
    def v_fn_y(y_vals):
        _, v_batch, _ = apply_fn_batch(x, y_vals, t)
        return v_batch
    
    v_y = jvp(v_fn_y, (y,), (vv_y,))[1] * scale_y
    
    def v_fn_t(t_vals):
        _, v_batch, _ = apply_fn_batch(x, y, t_vals)
        return v_batch
    
    v_t = jvp(v_fn_t, (t,), (vv_t,))[1] * scale_t
    
    # Second derivatives of v
    v_xx = hvp_fwdrev(v_fn_x, (x,), (vv_x,)) * (scale_x ** 2)
    v_yy = hvp_fwdrev(v_fn_y, (y,), (vv_y,)) * (scale_y ** 2)
    
    # Pressure derivatives
    def p_fn_x(x_vals):
        _, _, pre_batch = apply_fn_batch(x_vals, y, t)
        return pre_batch
    
    p_x = jvp(p_fn_x, (x,), (vv_x,))[1] * scale_x
    
    def p_fn_y(y_vals):
        _, _, pre_batch = apply_fn_batch(x, y_vals, t)
        return pre_batch
    
    p_y = jvp(p_fn_y, (y,), (vv_y,))[1] * scale_y
    
    return u, v, u_t, v_t, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, pre, p_x, p_y


# %%
############################
### Data loading
############################

def normalize(x1,T):
    normalized_x1 = (x1 - jnp.min(x1))/(jnp.max(x1)-jnp.min(x1))
    normalized_x1=(1-normalized_x1)*T
    normalized_x1 = normalized_x1.reshape(-1,1)
    return normalized_x1


def generate_regression_data(num_samples: int, T_snn):
    """Generate regression data with Gaussian receptive field encoding."""
    
    nx, ny, nt = 30,30,30
    # nx, ny, nt = 5, 5, 5

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

    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    t_min = 0
    t_max = 1

    # Normalize
    normalized_x_train = (1-((x_train-x_min)/(x_max-x_min)))*T_snn
    normalized_y_train = (1-((y_train-y_min)/(y_max-y_min)))*T_snn
    normalized_t_train = (1-((t_train-t_min)/(t_max-t_min)))*T_snn

    normalized_x_test = (1-((x_test-x_min)/(x_max-x_min)))*T_snn
    normalized_y_test = (1-((y_test-y_min)/(y_max-y_min)))*T_snn
    normalized_t_test = (1-((t_test-t_min)/(t_max-t_min)))*T_snn
    
    train_input = jnp.concat((normalized_x_train, normalized_y_train, normalized_t_train), axis=1)
    test_input = jnp.concat((normalized_x_test, normalized_y_test, normalized_t_test), axis=1)

    train_input = np.array(train_input)
    test_input = np.array(test_input)

    train_label = jnp.concat((u_train, v_train, p_train), axis=1)
    test_label = jnp.concat((u_test, v_test, p_test), axis=1)

    train_label = np.array(train_label)
    test_label = np.array(test_label)

    return train_input, train_label, test_input, test_label


T_snn = 2.0
num_train = 30*30*30
train_inputs, train_targets, test_inputs, test_targets = generate_regression_data(num_train, T_snn)


def create_batches(X, Y, config):
    batch_size: int = config["Nbatch"]
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], Y[batch_indices]


def load_data(data: callable, root: str, config: dict) -> tuple[DataLoader, DataLoader]:
    """Creates DataLoaders for regression data."""
    Nbatch: int = config["Nbatch"]
    T: float = config["T"]

    num_train = config.get("num_train", 5000)
    train_inputs, train_targets, test_inputs, test_targets = generate_regression_data(num_train, T)
    
    train_set = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_targets))
    train_loader = DataLoader(train_set, batch_size=Nbatch, shuffle=True)

    num_test = config.get("num_test", 5000)
    test_set = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_targets))
    test_loader = DataLoader(test_set, batch_size=5000, shuffle=False)

    return train_loader, test_loader


def safe_save_params(params, filepath):
    """Save JAX parameters safely using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Parameters saved to {filepath}")


def safe_load_params(filepath):
    """Load JAX parameters safely using pickle"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    print(f"Parameters loaded from {filepath}")
    return params


# %%
############################
### Initialization
############################

def init_weights(key: Array, config: dict) -> tuple[Array, list]:
    """Initializes input and network weights."""
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    w_scale: float = config["w_scale"]

    weights = []
    width = w_scale / jnp.sqrt(Nin)

    key, subkey = random.split(key)
    weights_in = random.uniform(subkey, (Nhidden, Nin), minval=-width, maxval=width)
    weights.append(weights_in)
    
    width = w_scale / jnp.sqrt(Nhidden)
    for _ in range(1, Nlayer - 1):
        key, subkey = random.split(key)
        weights_hidden = random.uniform(
            subkey, (Nhidden, Nhidden), minval=-width, maxval=width
        )
        weights.append(weights_hidden)
        
    key, subkey = random.split(key)
    weights_out = random.uniform(subkey, (Nout, Nhidden), minval=-width, maxval=width)
    weights.append(weights_out)

    return key, weights


def init_phi0(neuron: AbstractPhaseOscNeuron, config: dict) -> Array:
    """Initializes initial phase of neurons."""
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    theta = neuron.Theta()

    phi0 = theta / 2 * jnp.ones(N)
    return phi0


def init_lambda(config: dict):
    lambda_1 = jnp.array([0.0])
    lambda_2 = jnp.array([0.0])
    return lambda_1, lambda_2


# %%
############################
### Model
############################

def eventffwd(
    neuron: AbstractPhaseOscNeuron, p: list, input: Float[Array, " Nin"], config: dict
) -> tuple:
    """Simulates a feedforward network with time-to-first-spike input encoding."""
    Nin_virtual: int = config["Nin_virtual"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    T: float = config["T"]
    weights: list = p[0]
    phi0: Array = p[1]
    lambda_1 = p[2]
    lambda_2 = p[3]
    x0 = phi0[jnp.newaxis]

    neurons_in = jnp.arange(Nin_virtual)
    times_in = input
    spikes_in = (times_in, neurons_in)

    # Input weights
    weights_in = weights[0]
    weights_in_virtual = jnp.zeros((N, Nin_virtual))
    weights_in_virtual = weights_in_virtual.at[:Nhidden, :].set(weights_in)

    # Network weights
    weights_net = jnp.zeros((N, N))
    for i in range(Nlayer - 2):
        slice_in = slice(i * Nhidden, (i + 1) * Nhidden)
        slice_out = slice((i + 1) * Nhidden, (i + 2) * Nhidden)
        weights_net = weights_net.at[slice_out, slice_in].set(weights[i + 1])
    weights_net = weights_net.at[N - Nout :, N - Nout - Nhidden : N - Nout].set(
        weights[-1]
    )

    # Run simulation
    out = neuron.event(x0, weights_net, weights_in_virtual, spikes_in, config)

    return out


def outfn(
    neuron: AbstractPseudoPhaseOscNeuron, out: tuple, p: list, config: dict
) -> Array:
    """Computes output spike times given simulation results."""
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    weights = p[0]
    times: Array = out[0]
    spike_in: Array = out[1]
    neurons: Array = out[2]
    x: Array = out[3]

    # Run network as feedforward rate ANN
    Kord = jnp.sum(neurons >= 0)
    x_end = x[Kord]
    pseudo_rates = jnp.zeros(Nin)
    for i in range(Nlayer - 1):
        input = neuron.linear(pseudo_rates, weights[i])
        x_end_i = x_end[:, i * Nhidden : (i + 1) * Nhidden]
        pseudo_rates = neuron.construct_ratefn(x_end_i)(input)
    input = neuron.linear(pseudo_rates, weights[Nlayer - 1])

    # Spike times for each learned neuron
    def compute_tout(i: ArrayLike) -> Array:
        mask = (neurons == N - Nout + i) & (spike_in == False)
        Kout = jnp.sum(mask)
        t_out_ord = times[jnp.argmax(mask)]

        t_out_pseudo = neuron.t_pseudo(x_end[:, N - Nout + i], input[i], 1, config)

        t_out = jnp.where(0 < Kout, t_out_ord, t_out_pseudo)

        return t_out

    t_outs = vmap(compute_tout)(jnp.arange(Nout))
    return t_outs


def apply_boundary_conditions(neuron: AbstractPseudoPhaseOscNeuron, p: list, config: dict, input):
    """Apply boundary and initial conditions"""
    T = config["T"]
    
    # Raw network output
    outs = eventffwd(neuron, p, input, config)
    t_outs = outfn(neuron, outs, p, config)
    u_network = t_outs[1] - t_outs[0]
    v_network = t_outs[3] - t_outs[2]
    pred_p_raw = t_outs[5] - t_outs[4]
    
    # Convert to physical coordinates
    x = 1-((2*input[0])/T)
    y = 1-((2*input[1])/T)
    t = 1-(input[2]/T)

    # Initial conditions
    u_ic = -jnp.cos(x) * jnp.sin(y)
    v_ic = jnp.sin(x) * jnp.cos(y)
    p_ic = -0.25*(jnp.sin(x*2) + jnp.cos(y*2))
    
    # Normalized coordinates
    xi = (x + 1) / 2
    eta = (y + 1) / 2
    
    # Boundary conditions
    u_left = -jnp.cos(-1.0) * jnp.sin(y) * jnp.exp(-2 * t)
    u_right = -jnp.cos(1.0) * jnp.sin(y) * jnp.exp(-2 * t)
    u_bottom = -jnp.cos(x) * jnp.sin(-1.0) * jnp.exp(-2 * t)
    u_top = -jnp.cos(x) * jnp.sin(1.0) * jnp.exp(-2 * t)
    
    v_left = jnp.sin(-1.0) * jnp.cos(y) * jnp.exp(-2 * t)
    v_right = jnp.sin(1.0) * jnp.cos(y) * jnp.exp(-2 * t)
    v_bottom = jnp.sin(x) * jnp.cos(-1.0) * jnp.exp(-2 * t)
    v_top = jnp.sin(x) * jnp.cos(1.0) * jnp.exp(-2 * t)

    p_left = -0.25*(jnp.sin(-1.0*2) + jnp.cos(y*2)) * jnp.exp(-4 * t)
    p_right = -0.25*(jnp.sin(1.0*2) + jnp.cos(y*2)) * jnp.exp(-4 * t)
    p_bottom = -0.25*(jnp.sin(x*2) + jnp.cos(-1.0*2)) * jnp.exp(-4 * t)
    p_top = -0.25*(jnp.sin(x*2) + jnp.cos(1.0*2)) * jnp.exp(-4 * t)
    
    # Transfinite interpolation
    u_bc_x = u_left * (1 - xi) + u_right * xi
    u_bc_y = u_bottom * (1 - eta) + u_top * eta
    
    v_bc_x = v_left * (1 - xi) + v_right * xi
    v_bc_y = v_bottom * (1 - eta) + v_top * eta

    p_bc_x = p_left * (1 - xi) + p_right * xi
    p_bc_y = p_bottom * (1 - eta) + p_top * eta
    
    # Corner contributions
    u_ll = -jnp.cos(-1.0) * jnp.sin(-1.0) * jnp.exp(-2 * t)
    u_lr = -jnp.cos(1.0) * jnp.sin(-1.0) * jnp.exp(-2 * t)
    u_ul = -jnp.cos(-1.0) * jnp.sin(1.0) * jnp.exp(-2 * t)
    u_ur = -jnp.cos(1.0) * jnp.sin(1.0) * jnp.exp(-2 * t)
    u_corners = ((1-xi)*(1-eta)*u_ll + xi*(1-eta)*u_lr + (1-xi)*eta*u_ul + xi*eta*u_ur)
    
    v_ll = jnp.sin(-1.0) * jnp.cos(-1.0) * jnp.exp(-2 * t)
    v_lr = jnp.sin(1.0) * jnp.cos(-1.0) * jnp.exp(-2 * t)
    v_ul = jnp.sin(-1.0) * jnp.cos(1.0) * jnp.exp(-2 * t)
    v_ur = jnp.sin(1.0) * jnp.cos(1.0) * jnp.exp(-2 * t)
    v_corners = ((1-xi)*(1-eta)*v_ll + xi*(1-eta)*v_lr + (1-xi)*eta*v_ul + xi*eta*v_ur)

    p_ll = -0.25*(jnp.sin(-1.0*2) + jnp.cos(-1.0*2)) * jnp.exp(-4 * t)
    p_lr = -0.25*(jnp.sin(1.0*2) + jnp.cos(-1.0*2)) * jnp.exp(-4 * t)
    p_ul = -0.25*(jnp.sin(-1.0*2) + jnp.cos(1.0*2)) * jnp.exp(-4 * t)
    p_ur = -0.25*(jnp.sin(1.0*2) + jnp.cos(1.0*2)) * jnp.exp(-4 * t)
    p_corners = ((1-xi)*(1-eta)*p_ll + xi*(1-eta)*p_lr + (1-xi)*eta*p_ul + xi*eta*p_ur)
    
    # Complete transfinite interpolation
    u_bc = u_bc_x + u_bc_y - u_corners
    v_bc = v_bc_x + v_bc_y - v_corners
    p_bc = p_bc_x + p_bc_y - p_corners
    
    # Distance functions
    d_spatial = (1 - x**2) * (1 - y**2)
    d_temporal = 1-jnp.exp(-5.0*t)
    
    # Apply IC and BC
    u = (1 - d_temporal) * u_ic + d_temporal * u_bc + t * d_spatial * u_network
    v = (1 - d_temporal) * v_ic + d_temporal * v_bc + t * d_spatial * v_network
    pre = (1 - d_temporal) * p_ic + d_temporal * p_bc + t * d_spatial * pred_p_raw

    # return u, v, pred_p_raw

    u_h, v_h = hard_enforce_uv(x, y, t, u_network, v_network)
    return u_h, v_h, pred_p_raw  # usually keep pressure soft



# %%
############################
### Loss Functions (BATCHED)
############################

def lossfn_batched(neuron, p, targets, config, inputs_batch):
    """
    Compute physics-informed loss for BATCH of points.
    """
    
    # Compute derivatives for ALL points at once
    u, v, u_t, v_t, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, pre, p_x, p_y = \
        compute_derivatives_batched(neuron, p, config, inputs_batch)
    
    # print('u', u.shape)
    
    # PDE residuals (Navier-Stokes)
    f_u = u_t + (u*u_x + v*u_y) + p_x - (u_xx + u_yy)
    f_v = v_t + (u*v_x + v*v_y) + p_y - (v_xx + v_yy)
    f_c = u_x + v_y  # Continuity
    
    physics_loss = f_u**2 + f_v**2 + f_c**2
    
    loss = physics_loss
    
    threshold = config.get("reg_threshold", 0.1)
    correct = jnp.abs(targets[:, 0] - u) < threshold
    
    return loss, correct


def simulatefn_batched(neuron, p, inputs_batch, labels_batch, config):
    """Simulate network for entire batch at once"""
    loss, correct = lossfn_batched(neuron, p, labels_batch, config, inputs_batch)
    mean_loss = jnp.mean(loss)
    accuracy = jnp.mean(correct)
    return mean_loss, accuracy


def create_gradfn(neuron, config):
    """Create gradient function with neuron and config baked in"""
    
    @jit
    @partial(value_and_grad, has_aux=True)
    def _gradfn(p, inputs_batch, labels_batch):
        loss, acc = simulatefn_batched(neuron, p, inputs_batch, labels_batch, config)
        return loss, acc
    
    return _gradfn


# %%
############################
### Probe Function
############################

def probefn(neuron, p, input, labels, config):
    """Computes several metrics using batched computation."""
    
    T = config["T"]
    Nhidden = config["Nhidden"]
    Nlayer = config["Nlayer"]
    Nout = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    Nbatch = input.shape[0]
    
    # Run network (batched)
    @vmap
    def batch_eventffwd(input_single):
        return eventffwd(neuron, p, input_single, config)
    
    @vmap
    def batch_outfn(outs):
        return outfn(neuron, outs, p, config)
    
    outs = batch_eventffwd(input)
    times = outs[0]
    spike_in = outs[1]
    neurons = outs[2]
    t_outs = batch_outfn(outs)
    
    # Loss and accuracy with pseudospikes
    loss, correct = lossfn_batched(neuron, p, labels, config, input)
    mean_loss = jnp.mean(loss)
    acc = jnp.mean(correct)
    
    # Loss and accuracy without pseudospikes
    t_out_ord = jnp.where(t_outs < T, t_outs, T)
    
    def lossfn_ord_single(t_out, label):
        pred = t_out[1] - t_out[0]
        loss = (pred - label[0]) ** 2
        threshold = config.get("reg_threshold", 0.1)
        correct = jnp.abs(pred - label[0]) < threshold
        return loss, correct
    
    loss_ord, correct_ord = vmap(lossfn_ord_single)(t_out_ord, labels)
    mean_loss_ord = jnp.mean(loss_ord)
    acc_ord = jnp.mean(correct_ord)
    
    # Activity and silent neurons
    mask = (spike_in == False) & (neurons < N - Nout) & (neurons >= 0)
    activity = jnp.sum(mask) / (Nbatch * (N - Nout))
    silent_neurons = jnp.isin(
        jnp.arange(N - Nout), jnp.where(mask, neurons, -1), invert=True
    )
    
    # Activity until first output spike
    t_out_first = jnp.min(t_out_ord, axis=1)
    mask = (
        (spike_in == False)
        & (neurons < N - Nout)
        & (neurons >= 0)
        & (times < t_out_first[:, jnp.newaxis])
    )
    activity_first = jnp.sum(mask) / (Nbatch * (N - Nout))
    silent_neurons_first = jnp.isin(
        jnp.arange(N - Nout), jnp.where(mask, neurons, -1), invert=True
    )
    
    # Pack results
    metrics = {
        "loss": mean_loss,
        "acc": acc,
        "loss_ord": mean_loss_ord,
        "acc_ord": acc_ord,
        "activity": activity,
        "activity_first": activity_first,
    }
    silents = {
        "silent_neurons": silent_neurons,
        "silent_neurons_first": silent_neurons_first,
    }
    
    return metrics, silents


# %%
############################
### Training
############################

def run(
    neuron: AbstractPseudoPhaseOscNeuron,
    config: dict,
    progress_bar: str | None = None
) -> dict:
    """
    Trains a feedforward network with batched derivative computation for PINNs.
    """

    # Unpack arguments
    seed: int = config["seed"]
    Nin_virtual: int = config["Nin_virtual"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    Nepochs: int = config["Nepochs"]
    lr: float = config["lr"]
    theta = neuron.Theta()
    
    if progress_bar == "script":
        trange = trange_script
    else:
        trange = range

    # Initialize parameters
    key = random.PRNGKey(seed)
    key, weights = init_weights(key, config)
    phi0 = init_phi0(neuron, config)
    lambda_1, lambda_2 = init_lambda(config)
    p = [weights, phi0, lambda_1, lambda_2]
    p_init = [weights, phi0, lambda_1, lambda_2]

    # Optimizer
    optim = optax.adam(lr)
    opt_state = optim.init(p)
    
    # Create gradient function ONCE with neuron and config baked in
    gradfn = create_gradfn(neuron, config)

    # Load data
    torch.manual_seed(seed)
    train_loader, test_loader = load_data(datasets.MNIST, "data", config)

    # JIT compiled probe function
    @jit
    def jprobefn(p, input, labels):
        return probefn(neuron, p, input, labels, config)

    # Probe function to evaluate on entire dataset
    def probe(p):
        metrics_agg = {
            "loss": 0.0,
            "acc": 0.0,
            "loss_ord": 0.0,
            "acc_ord": 0.0,
            "activity": 0.0,
            "activity_first": 0.0,
        }
        silents = {
            "silent_neurons": jnp.ones(N - Nout, dtype=bool),
            "silent_neurons_first": jnp.ones(N - Nout, dtype=bool),
        }
        
        steps = 0
        for input, labels in create_batches(train_inputs, train_targets, config):
            metric, silent = jprobefn(p, input, labels)
            for k in metrics_agg:
                metrics_agg[k] += metric[k]
            for k in silents:
                silents[k] = silents[k] & silent[k]
            steps += 1
        
        # Average metrics
        for k in metrics_agg:
            metrics_agg[k] /= steps
        
        # Convert silent neuron masks to fractions
        for k, v in silents.items():
            metrics_agg[k] = jnp.mean(v).item()
        
        return metrics_agg

    # Initial probe
    initial_metrics = probe(p)
    
    # Metrics tracking - initialize with probe results
    metrics = {k: [v] for k, v in initial_metrics.items()}
    
    # Timing
    times = []
    
    # 350 1e-3, 300+850 1e-4
    # Load saved parameters if available
    if os.path.exists('ns_pinn_params_hard.pkl'):
        print(f"Loading parameters from 'ns_pinn_params_hard.pkl'")
        p = safe_load_params('ns_pinn_params_hard.pkl')
    else:
        print(f"No saved parameters found. Training from scratch.")

    # Training loop
    pre_loss = 10000000
    for epoch in trange(Nepochs):
        epoch_loss = 0
        batch_count = 0
        
        for input, labels in create_batches(train_inputs, train_targets, config):
            st = time.time()
            
            # Compute gradients and update (BATCHED!)
            (loss, acc), grad = gradfn(p, input, labels)
            
            updates, opt_state = optim.update(grad, opt_state)
            p = optax.apply_updates(p, updates)
            
            # Clip phases
            p[1] = jnp.clip(p[1], 0, theta)
            
            et = time.time()
            times.append(et - st)
            
            epoch_loss += loss
            batch_count += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        
        # Probe after each epoch
        epoch_metrics = probe(p)
        for k, v in epoch_metrics.items():
            metrics[k].append(v)
        
        if epoch % 50 == 0:
            # Test on a sample point
            u, v, u_t, v_t, u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy, pre, p_x, p_y = \
                compute_derivatives_batched(neuron, p, config, train_inputs[2000:2001, :])
            print(f'Epoch {epoch}: Loss = {avg_epoch_loss:.6e}')
            print(f'  u={u[0]:.4f}, v={v[0]:.4f}, p={pre[0]:.4f}')
            print(f'  True: u={train_targets[2000, 0]:.4f}, v={train_targets[2000, 1]:.4f}, p={train_targets[2000, 2]:.4f}')
            
            if avg_epoch_loss <= pre_loss:
                safe_save_params(p, 'ns_pinn_params_hard.pkl')
                pre_loss = avg_epoch_loss

    # Test evaluation
    T_snn = config['T']
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

    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    t_min = 0
    t_max = 1

    normalized_x_test = (1-((x_test-x_min)/(x_max-x_min)))*T_snn
    normalized_y_test = (1-((y_test-y_min)/(y_max-y_min)))*T_snn
    normalized_t_test = (1-((t_test-t_min)/(t_max-t_min)))*T_snn
    
    encoded_inputs_test = jnp.concat((normalized_x_test, normalized_y_test, normalized_t_test), axis=1)
    encoded_inputs_test = np.array(encoded_inputs_test)
    test_label = jnp.concat((u_test, v_test, p_test), axis=1)
    test_label = np.array(test_label)

    @jit
    def test_forward(p, inputs):
        outs = vmap(eventffwd, in_axes=(None, None, 0, None))(
            neuron, p, inputs, config
        )
        t_outs = vmap(outfn, in_axes=(None, 0, None, None))(
            neuron, outs, p, config
        )
        return t_outs

    t_outs_test = test_forward(p, encoded_inputs_test)

    preds_raw_u_test = jnp.squeeze(t_outs_test[:,1]-t_outs_test[:,0])
    preds_raw_v_test = jnp.squeeze(t_outs_test[:,3]-t_outs_test[:,2])
    preds_raw_pre_test = jnp.squeeze(t_outs_test[:,5]-t_outs_test[:,4])

    x_test = 1-((2*encoded_inputs_test[:, 0])/T_snn)
    y_test = 1-((2*encoded_inputs_test[:, 1])/T_snn)
    t_test = 1-(encoded_inputs_test[:, 2]/T_snn)

    # Apply boundary conditions to test predictions
    u_ic_test = -jnp.cos(x_test) * jnp.sin(y_test)
    v_ic_test = jnp.sin(x_test) * jnp.cos(y_test)
    p_ic_test = -0.25*(jnp.sin(x_test*2) + jnp.cos(y_test*2))
    
    xi_test = (x_test + 1) / 2
    eta_test = (y_test + 1) / 2
    
    u_left_test = -jnp.cos(-1.0) * jnp.sin(y_test) * jnp.exp(-2 * t_test)
    u_right_test = -jnp.cos(1.0) * jnp.sin(y_test) * jnp.exp(-2 * t_test)
    u_bottom_test = -jnp.cos(x_test) * jnp.sin(-1.0) * jnp.exp(-2 * t_test)
    u_top_test = -jnp.cos(x_test) * jnp.sin(1.0) * jnp.exp(-2 * t_test)
    
    v_left_test = jnp.sin(-1.0) * jnp.cos(y_test) * jnp.exp(-2 * t_test)
    v_right_test = jnp.sin(1.0) * jnp.cos(y_test) * jnp.exp(-2 * t_test)
    v_bottom_test = jnp.sin(x_test) * jnp.cos(-1.0) * jnp.exp(-2 * t_test)
    v_top_test = jnp.sin(x_test) * jnp.cos(1.0) * jnp.exp(-2 * t_test)

    p_left_test = -0.25*(jnp.sin(-1.0*2) + jnp.cos(y_test*2)) * jnp.exp(-4 * t_test)
    p_right_test = -0.25*(jnp.sin(1.0*2) + jnp.cos(y_test*2)) * jnp.exp(-4 * t_test)
    p_bottom_test = -0.25*(jnp.sin(x_test*2) + jnp.cos(-1.0*2)) * jnp.exp(-4 * t_test)
    p_top_test = -0.25*(jnp.sin(x_test*2) + jnp.cos(1.0*2)) * jnp.exp(-4 * t_test)
    
    u_bc_x_test = u_left_test * (1 - xi_test) + u_right_test * xi_test
    u_bc_y_test = u_bottom_test * (1 - eta_test) + u_top_test * eta_test
    
    v_bc_x_test = v_left_test * (1 - xi_test) + v_right_test * xi_test
    v_bc_y_test = v_bottom_test * (1 - eta_test) + v_top_test * eta_test

    p_bc_x_test = p_left_test * (1 - xi_test) + p_right_test * xi_test
    p_bc_y_test = p_bottom_test * (1 - eta_test) + p_top_test * eta_test
    
    u_ll_test = -jnp.cos(-1.0) * jnp.sin(-1.0) * jnp.exp(-2 * t_test)
    u_lr_test = -jnp.cos(1.0) * jnp.sin(-1.0) * jnp.exp(-2 * t_test)
    u_ul_test = -jnp.cos(-1.0) * jnp.sin(1.0) * jnp.exp(-2 * t_test)
    u_ur_test = -jnp.cos(1.0) * jnp.sin(1.0) * jnp.exp(-2 * t_test)
    u_corners_test = ((1-xi_test)*(1-eta_test)*u_ll_test + xi_test*(1-eta_test)*u_lr_test + 
                      (1-xi_test)*eta_test*u_ul_test + xi_test*eta_test*u_ur_test)
    
    v_ll_test = jnp.sin(-1.0) * jnp.cos(-1.0) * jnp.exp(-2 * t_test)
    v_lr_test = jnp.sin(1.0) * jnp.cos(-1.0) * jnp.exp(-2 * t_test)
    v_ul_test = jnp.sin(-1.0) * jnp.cos(1.0) * jnp.exp(-2 * t_test)
    v_ur_test = jnp.sin(1.0) * jnp.cos(1.0) * jnp.exp(-2 * t_test)
    v_corners_test = ((1-xi_test)*(1-eta_test)*v_ll_test + xi_test*(1-eta_test)*v_lr_test + 
                      (1-xi_test)*eta_test*v_ul_test + xi_test*eta_test*v_ur_test)

    p_ll_test = -0.25*(jnp.sin(-1.0*2) + jnp.cos(-1.0*2)) * jnp.exp(-4 * t_test)
    p_lr_test = -0.25*(jnp.sin(1.0*2) + jnp.cos(-1.0*2)) * jnp.exp(-4 * t_test)
    p_ul_test = -0.25*(jnp.sin(-1.0*2) + jnp.cos(1.0*2)) * jnp.exp(-4 * t_test)
    p_ur_test = -0.25*(jnp.sin(1.0*2) + jnp.cos(1.0*2)) * jnp.exp(-4 * t_test)
    p_corners_test = ((1-xi_test)*(1-eta_test)*p_ll_test + xi_test*(1-eta_test)*p_lr_test + 
                      (1-xi_test)*eta_test*p_ul_test + xi_test*eta_test*p_ur_test)
    
    u_bc_test = u_bc_x_test + u_bc_y_test - u_corners_test
    v_bc_test = v_bc_x_test + v_bc_y_test - v_corners_test
    p_bc_test = p_bc_x_test + p_bc_y_test - p_corners_test
    
    d_spatial_test = (1 - x_test**2) * (1 - y_test**2)
    d_temporal_test = 1-jnp.exp(-3.0*t_test)
    
    # u_pred_test = (1 - d_temporal_test) * u_ic_test + d_temporal_test * u_bc_test + t_test * d_spatial_test * preds_raw_u_test
    # v_pred_test = (1 - d_temporal_test) * v_ic_test + d_temporal_test * v_bc_test + t_test * d_spatial_test * preds_raw_v_test
    pre_pred_test = (1 - d_temporal_test) * p_ic_test + d_temporal_test * p_bc_test + t_test * d_spatial_test * preds_raw_pre_test

    u_pred_test, v_pred_test = hard_enforce_uv(x_test, y_test, t_test, preds_raw_u_test, preds_raw_v_test)

    # Save results
    sio.savemat('experiments/pinnst/qif_pinn_ns_hard.mat',{
        'x': np.linspace(-1, 1, nx_test),
        'y': np.linspace(-1, 1, ny_test),
        't': np.linspace(0, 1, nt_test),
        'u_true': u_test.reshape(nx_test, ny_test, nt_test),
        'v_true': v_test.reshape(nx_test, ny_test, nt_test),
        'p_true': p_test.reshape(nx_test, ny_test, nt_test),
        'u_pred': u_pred_test.reshape(nx_test, ny_test, nt_test),
        'v_pred': v_pred_test.reshape(nx_test, ny_test, nt_test),
        'p_pred': pre_pred_test.reshape(nx_test, ny_test, nt_test),
        'time': times,
        'loss': [m for m in metrics["loss"]]
    })

    total_time = jnp.sum(jnp.array(times))
    print(f'Total training time: {total_time:.2f}s')

    if jnp.any(jnp.isnan(jnp.array(metrics["loss"]))):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )
    
    # Convert metrics to arrays
    metrics = {k: jnp.array(v) for k, v in metrics.items()}
    metrics["p_init"] = p_init
    metrics["p_end"] = p

    return metrics


# %%
############################
### Examples (Single Point Evaluation)
############################

def run_example(p: list, neuron: AbstractPseudoPhaseOscNeuron, config: dict) -> dict:
    """
    Simulates the network for a single example input given the parameters `p`.
    """

    seed: int = config["seed"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout

    @jit
    def jeventffwd(p, input):
        return eventffwd(neuron, p, input, config)

    @jit
    def joutfn(out, p):
        return outfn(neuron, out, p, config)

    # Data
    torch.manual_seed(seed)
    _, test_loader = load_data(datasets.MNIST, "data", config)

    input, label = next(iter(test_loader))
    input, label = jnp.array(input[60]), jnp.array(label[60])
    out = jeventffwd(p, input)
    t_outs = joutfn(out, p)

    # Prepare results
    times: Array = out[0]
    spike_in: Array = out[1]
    neurons: Array = out[2]

    trace_ts, trace_xs = neuron.traces(p[1][jnp.newaxis], out, config)
    trace_phis = trace_xs[:, 0]
    trace_Vs = neuron.iPhi(trace_phis)

    spiketimes = []
    for i in range(N):
        times_i = times[~spike_in & (neurons == i)]
        spiketimes.append(times_i)
    
    predicted = t_outs[1]-t_outs[0]

    # Pack results
    results = {
        "input": input,
        "label": label,
        "predicted": predicted,
        "trace_ts": trace_ts,
        "trace_phis": trace_phis,
        "trace_Vs": trace_Vs,
        "spiketimes": spiketimes,
    }
    
    print('input', results['input'])
    print('label', results['label'])
    print('out spike times', t_outs)
    print('pred', predicted)

    return results


# %%
############################
### Plotting
############################

def plot_spikes(ax: Axes, example: dict, config: dict) -> None:
    """Plot spike raster"""
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout
    spiketimes: Array = example["spiketimes"]

    tick_len = 2
    ax.eventplot(spiketimes, colors="k", linewidths=0.5, linelengths=tick_len)
    patch = Rectangle((0, Nhidden - 1 / 2), T, Nhidden, color="k", alpha=0.2, zorder=0)
    ax.add_patch(patch)
    ax.text(
        T,
        0,
        r"$1^\mathrm{st}$ hidden",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )
    ax.text(
        T,
        Nhidden - 1 / 2,
        r"$2^\mathrm{nd}$ hidden",
        ha="right",
        va="bottom",
        color="white",
        zorder=1,
    )
    ax.text(
        T,
        2 * Nhidden - 1 / 2,
        "Output",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )

    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks(
        [0, Nhidden - 1, 2 * Nhidden - 1, N - 1],
        [str(1), str(Nhidden), str(2 * Nhidden), str(N)],
    )
    ax.set_ylim(-tick_len / 2, N - 1 + tick_len / 2)
    ax.set_ylabel("Neuron", labelpad=-0.1)


def plot_error(ax: Axes, metrics: dict, config: dict) -> None:
    """Plot classification error"""
    Nepochs: int = config["Nepochs"]
    acc: Array = metrics["acc"]
    mean_acc = jnp.mean(acc, 0)
    std_acc = jnp.std(acc, 0)
    acc_ord: Array = metrics["acc_ord"]
    mean_acc_ord = jnp.mean(acc_ord, 0)
    std_acc_ord = jnp.std(acc_ord, 0)
    epochs = jnp.arange(1, Nepochs + 2)

    ax.plot(epochs, 1 - mean_acc_ord, label="Excl. pseudo", c="C0", zorder=1)
    ax.fill_between(
        epochs,
        1 - mean_acc_ord - std_acc_ord,
        1 - mean_acc_ord + std_acc_ord,
        alpha=0.3,
        color="C0",
    )
    ax.plot(epochs, 1 - mean_acc, label="Incl. pseudo", c="C1", zorder=0)
    ax.fill_between(
        epochs,
        1 - mean_acc - std_acc,
        1 - mean_acc + std_acc,
        alpha=0.3,
        color="C1",
    )
    ax.legend()

    ax.set_xlabel("Epochs + 1", labelpad=-1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim(0.01, 1)
    ax.set_ylabel("Test error", labelpad=-3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(formatter)


def plot_traces(ax: Axes, example: dict, config: dict) -> None:
    """Plot membrane potential traces"""
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout

    trace_ts: Array = example["trace_ts"]
    trace_Vs: Array = example["trace_Vs"]

    ax.axhline(0, c="gray", alpha=0.3, zorder=-1)
    ax.plot([-0.1, -0.1], [0, 1], c="k", clip_on=False)
    for i in range(10):
        ax.plot(trace_ts, trace_Vs[:, N - Nout + i], color=petroff10[i])
        ax.text((i % 5) * 0.15, -4 - (i // 5) * 3, str(i), color=petroff10[i])

    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks([])
    ax.set_ylim(-8, 8)
    ax.set_ylabel("Potential $V$")
    ax.spines["left"].set_visible(False)
