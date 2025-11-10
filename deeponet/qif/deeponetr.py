from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import torch
from jax import jit, random, value_and_grad, vmap
from jaxtyping import Array, ArrayLike, Float, Int, UInt8
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange as trange_script
# from tqdm.notebook import trange as trange_notebook
from tqdm import trange as trange_scipt  # Works in .py scripts
import time
import numpy as np
import os
import pickle

from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron
from spikegd.utils.plotting import formatter, petroff10
import scipy.io as sio

# %%
############################
### Data loading
############################


# Save parameters (works with any JAX parameter structure)
def safe_save_params(params, filepath):
    """Save JAX parameters safely using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Parameters saved to {filepath}")

# Load parameters
def safe_load_params(filepath):
    """Load JAX parameters safely using pickle"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    print(f"Parameters loaded from {filepath}")
    return params

def input_normalization(xs, T):
    normalized = (xs - jnp.min(xs))/(jnp.max(xs)-jnp.min(xs))
    normalized=(1-normalized)*T
    # normalized = normalized.reshape(-1,1)
    # encoded_inputs=normalized.reshape(-1,1)
    return normalized

# no encoding
def generate_regression_data(num_samples: int, T):
    """Generate regression data with Gaussian receptive field encoding."""
    # Generate raw inputs and targets
    train_data = sio.loadmat('experiments/deeponet/train_data.mat')
    branch_X_train = train_data['f_train'][:800, :]
    trunk_X_train = train_data['x_train'].reshape(-1, 1)
    Y = train_data['y_train'][:800, :]
    # Y = Y/jnp.max(Y)
    Y_min = jnp.min(Y)
    Y_max = jnp.max(Y)
    Y = (Y-Y_min)/(Y_max-Y_min)
    print(branch_X_train.shape, trunk_X_train.shape, Y.shape)
    # branch_X_test = train_data['f_train'][800:1600, :]
    # trunk_X_test = train_data['x_train'].reshape(-1, 1)
    # Y_test = train_data['y_train'][800:1600, :]
    test_data = sio.loadmat('experiments/deeponet/test_data.mat')
    branch_X_test = test_data['f_test'][:800, :]
    trunk_X_test = test_data['x_test'].reshape(-1, 1)[:800, :]
    # trunk_X_test = jnp.concatenate((trunk_X_test, jnp.sin(trunk_X_test), jnp.sin(2*trunk_X_test), jnp.sin(3*trunk_X_test), jnp.cos(trunk_X_test), jnp.cos(2*trunk_X_test), jnp.cos(3*trunk_X_test)), axis=1)
    Y_test = test_data['y_test'][:800, :]

    branch_min = min(jnp.min(branch_X_test), jnp.min(branch_X_train))
    branch_max = max(jnp.max(branch_X_train), jnp.max(branch_X_test))
    train_input_b = (branch_X_train-branch_min)/(branch_max-branch_min)
    test_input_b = (branch_X_test-branch_min)/(branch_max-branch_min)

    # Y_test = Y_test/jnp.max(Y_test)
    # Y_test = (Y_test-jnp.min(Y_test))/(jnp.max(Y_test)-jnp.min(Y_test))
    # Y_test = (Y_test-Y_min)/(Y_max-Y_min)
    # train_input_b = input_normalization(branch_X_train, T)
    train_input_t = input_normalization(trunk_X_train, T)
    # test_input_b = input_normalization(branch_X_test, T)
    test_input_t = input_normalization(trunk_X_test, T)
    # test_input_b = input_normalization(test_input_b, T)
    # train_input_b = input_normalization(train_input_b, T)

    # print(jnp.min(train_input_b), jnp.max(train_input_b), jnp.min(test_input_b), jnp.max(test_input_b) )
    train_input_b = (1 - train_input_b)*T
    test_input_b = (1 - test_input_b)*T
    

    # branch_min = min(jnp.min(train_input_b), jnp.min(test_input_b))
    # branch_max = max(jnp.max(train_input_b), jnp.max(test_input_b))
    # train_input_b = (train_input_b-branch_min)/(branch_max-branch_min)
    # test_input_b = (test_input_b-branch_min)/(branch_max-branch_min)

    # trunk_min = min(jnp.min(train_input_t), jnp.min(test_input_t))
    # trunk_max = max(jnp.max(train_input_t), jnp.max(test_input_t))
    # train_input_t = (train_input_t-trunk_min)/(trunk_max-branch_min)
    # test_input_t = (test_input_t-trunk_min)/(trunk_max-trunk_min)


    train_input_b = np.array(train_input_b)
    train_input_t = np.array(train_input_t)
    test_input_b = np.array(test_input_b)
    test_input_t = np.array(test_input_t)
    Y = np.array(Y)
    Y_test = np.array(Y_test)
    return train_input_b, train_input_t, Y, test_input_b, test_input_t, Y_test




def load_data(data: callable, root: str, config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for regression data.
    
    This function generates synthetic regression data (learning y = x^2) instead of
    loading MNIST images. The generated data is wrapped in a TensorDataset to mimic
    the original data-loading format.
    
    Args:
        data: Unused here; kept for compatibility with the original signature.
        root: Unused for regression data.
        config: Dictionary containing configuration parameters.
            Expected keys include:
              - "Nbatch": Batch size.
              - "num_train": Number of training samples.
              - "num_test": Number of test samples.
    
    Returns:
        A tuple (train_loader, test_loader) of PyTorch DataLoaders.
    """
    Nbatch: int = config["Nbatch"]
    T: float = config["T"]

    # Training set: Generate regression data
    num_train = config.get("num_train", 100)
    train_inputs_b, train_inputs_t, train_targets, test_inputs_b, test_inputs_t, test_targets = generate_regression_data(num_train, T)
    # print(train_inputs)
    # Convert JAX arrays to PyTorch tensors
    train_inputs_t = jnp.repeat(train_inputs_t.reshape(1, train_inputs_t.shape[0], train_inputs_t.shape[1]), train_inputs_b.shape[0], axis=0)
    train_inputs_t = np.array(train_inputs_t)
    # print(train_inputs_b.shape, train_inputs_t.shape, train_targets.shape)
    train_set = TensorDataset(torch.tensor(train_inputs_b), torch.tensor(train_inputs_t), torch.tensor(train_targets))
    train_loader = DataLoader(train_set, batch_size=Nbatch, shuffle=True)

    # Test set: Generate regression data
    # num_test = config.get("num_test", 100)
    # test_inputs, test_targets = generate_regression_data(num_test, T)
    test_inputs_t = jnp.repeat(test_inputs_t.reshape(1, test_inputs_t.shape[0], test_inputs_t.shape[1]), test_inputs_b.shape[0], axis=0)
    test_inputs_t = np.array(test_inputs_t)
    test_set = TensorDataset(torch.tensor(test_inputs_b), torch.tensor(test_inputs_t), torch.tensor(test_targets))
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    return train_loader, test_loader












# %%
############################
### Initialization
############################


def init_weights(key: Array, config: dict):
    """
    Initializes input and network weights.
    """
    ### Unpack arguments
    Nin_b: int = config["Nin_b"] # Input function dimensionality
    Nin_t: int = config["Nin_t"] # Dimension of evaluation points
    Nhidden_b: int = config["Nhidden_b"] # Width of branch network
    Nhidden_t: int = config["Nhidden_t"]  # Width of trunk network
    Nlayer_b: int = config["Nlayer_b"]  # Number of layers in branch network
    Nlayer_t: int = config["Nlayer_t"] # Number of layers in trunk network
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]
    w_scale: float = config["w_scale"]

    ### Initialize branch weights
    key, subkey = random.split(key)
    weights_b = []
    width_b = w_scale / jnp.sqrt(Nin_b)
    weights_in_b = random.uniform(subkey, (Nhidden_b, Nin_b), minval=-width_b, maxval=width_b)
    weights_b.append(weights_in_b)
    width_b = w_scale / jnp.sqrt(Nhidden_b)
    for _ in range(1, Nlayer_b - 1):
        key, subkey = random.split(key)
        weights_hidden_b = random.uniform(
            subkey, (Nhidden_b, Nhidden_b), minval=-width_b, maxval=width_b
        )
        weights_b.append(weights_hidden_b)
    # Final layer of branch network
    key, subkey = random.split(key)
    weights_out_b = random.uniform(subkey, (Nout_b, Nhidden_b), minval=-width_b, maxval=width_b)
    weights_b.append(weights_out_b)
    # print(weights_b[-1].shape)


    ### Initialize trunk weights
    key, subkey = random.split(key)
    weights_t = []
    width_t = w_scale / jnp.sqrt(Nin_t)
    weights_in_t = random.uniform(subkey, (Nhidden_t, Nin_t), minval=-width_t, maxval=width_t)
    weights_t.append(weights_in_t)
    width_t = w_scale / jnp.sqrt(Nhidden_t)
    for _ in range(1, Nlayer_t - 1):
        key, subkey = random.split(key)
        weights_hidden_t = random.uniform(
            subkey, (Nhidden_t, Nhidden_t), minval=-width_t, maxval=width_t
        )
        weights_t.append(weights_hidden_t)
    # Final layer of trunk network
    key, subkey = random.split(key)
    weights_out_t = random.uniform(subkey, (Nout_t, Nhidden_t), minval=-width_t, maxval=width_t)
    weights_t.append(weights_out_t)

    # Return all weights
    weights = [weights_b, weights_t]

    # return key, weights
    return key, weights_b, weights_t


def init_phi0(neuron_b: AbstractPhaseOscNeuron, neuron_t: AbstractPhaseOscNeuron, config: dict):
    """
    Initializes initial phase of neurons.
    """
    ### Unpack arguments
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]
    Num_points: int = config["Num_points"]


    # Total number of neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1) + Nout_b
    
    # Total number of neurons in trunk network
    N_trunk = Nhidden_t* (Nlayer_t-1) + Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk

    theta_b = neuron_b.Theta()
    theta_t = neuron_t.Theta()

    # ### Initialize initial phase
    # phi0 = theta / 2 * jnp.ones(N)
    # print(jnp.ones(N_branch).shape)

    ### Initialize initial phase (branch)
    phi0_b = theta_b / 2 * jnp.ones(N_branch)

    ### Initialize initial phase (trunk)
    phi0_t = theta_t / 2 * jnp.ones(N_trunk)
    # phi0_t = theta_t / 2 * jnp.ones((Num_points, N_trunk))

    # return phi0
    return phi0_b, phi0_t


# %%
############################
### Model
############################


def eventffwd(
    neuron_b: AbstractPhaseOscNeuron, neuron_t: AbstractPhaseOscNeuron, p_b: list, p_t: list, input_b: Float[Array, " Nin_b"], input_t: Float[Array, " Nin_t"], config: dict
):
    """
    Simulates a feedforward network with time-to-first-spike input encoding.
    """
    ### Unpack arguments
    Nin_virtual_b: int = config["Nin_virtual_b"]
    Nin_virtual_t: int = config["Nin_virtual_t"]
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]  
    Nlayer_t: int = config["Nlayer_t"]  
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b
    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk

    T: float = config["T"]
    weights_b: list = p_b[0]
    # print(weights_b[-1].shape)
    weights_t: list = p_t[0]
    phi0_b: Array = p_b[1]
    phi0_t: Array = p_t[1]
    x0_b = phi0_b[jnp.newaxis]
    x0_t = phi0_t[jnp.newaxis]

  
    # branch input
    neurons_in_b = jnp.arange(Nin_virtual_b)
    # print(input.shape)
    times_in_b= input_b
    # print('b', neurons_in_b.shape, times_in_b.shape)
    spikes_in_b = (times_in_b, neurons_in_b)

    # trunk input
    neurons_in_t = jnp.arange(Nin_virtual_t)
    times_in_t= input_t
    neurons_in_t = jnp.repeat(neurons_in_t.reshape(neurons_in_t.shape[0], 1), input_t.shape[0], axis=1)
    neurons_in_t = neurons_in_t.reshape(neurons_in_t.shape[1], -1)
    # print('t', neurons_in_t.shape, times_in_t.shape)
    spikes_in_t = (times_in_t, neurons_in_t)

    ### branch Input weights 
    weights_in_b = weights_b[0]
    weights_in_virtual_b = jnp.zeros((N_branch, Nin_virtual_b))
    weights_in_virtual_b = weights_in_virtual_b.at[:Nhidden_b, :].set(
            weights_in_b
        )
    
    ### trunk Input weights 
    weights_in_t = weights_t[0]
    weights_in_virtual_t = jnp.zeros((N_trunk, Nin_virtual_t))
    weights_in_virtual_t = weights_in_virtual_t.at[:Nhidden_t, :].set(
            weights_in_t
        )

    ### branch Network weights
    weights_net_b = jnp.zeros((N_branch, N_branch))
    # print(weights_net_b.shape)
    for i in range(Nlayer_b - 2):
        slice_in_b = slice(i * Nhidden_b, (i + 1) * Nhidden_b)
        slice_out_b = slice((i + 1) * Nhidden_b, (i + 2) * Nhidden_b)
        # print(slice_in_b, slice_out_b)
        weights_net_b = weights_net_b.at[slice_out_b, slice_in_b].set(weights_b[i + 1])
    # print(N_branch - Nout_b, N_branch - Nout_b - Nhidden_b)
    weights_net_b = weights_net_b.at[N_branch - Nout_b :, N_branch - Nout_b - Nhidden_b : N_branch - Nout_b].set(
        weights_b[-1]
    )

    ### trunk Network weights
    weights_net_t = jnp.zeros((N_trunk, N_trunk))
    for i in range(Nlayer_t - 2):
        slice_in_t = slice(i * Nhidden_t, (i + 1) * Nhidden_t)
        slice_out_t = slice((i + 1) * Nhidden_t, (i + 2) * Nhidden_t)
        weights_net_t = weights_net_t.at[slice_out_t, slice_in_t].set(weights_t[i + 1])
    weights_net_t = weights_net_t.at[N_trunk - Nout_t :, N_trunk - Nout_t - Nhidden_t : N_trunk - Nout_t].set(
        weights_t[-1]
    )

    # Run simulation (branch)
    out_b = neuron_b.event(x0_b, weights_net_b, weights_in_virtual_b, spikes_in_b, config)

    # Run simulation (trunk)
    out_t = neuron_t.event(x0_t, weights_net_t, weights_in_virtual_t, spikes_in_t, config)

    return out_b, out_t





def outfn_b(
    neuron_b: AbstractPseudoPhaseOscNeuron, out_b: tuple, p_b: list, config: dict
):
    """
    Computes output spike times given simulation results.
    """
    ### Unpack arguments
    Nin_b: int = config["Nin_b"]
    Nhidden_b: int = config["Nhidden_b"]
    Nlayer_b: int = config["Nlayer_b"]
    Nout_b: int = config["Nout_b"]
    Num_points: int = config["Num_points"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b


    weights_b = p_b[0]
    times_b: Array = out_b[0]
    spike_in_b: Array = out_b[1]
    neurons_b: Array = out_b[2]
    x_b: Array = out_b[3]
    # print(x_b.shape, x_t.shape)

    ### Run network as feedforward rate ANN (branch)
    Kord_b = jnp.sum(neurons_b >= 0)  # Number of ordinary spikes
    x_end_b = x_b[Kord_b]
    # print(x_end_b.shape)
    pseudo_rates_b = jnp.zeros(Nin_b)
    for i in range(Nlayer_b - 1):
        # print(pseudo_rates_b.shape, weights_b[i].shape)
        input_b = neuron_b.linear(pseudo_rates_b, weights_b[i])
        # print(input_b.shape)
        x_end_i_b = x_end_b[:, i * Nhidden_b : (i + 1) * Nhidden_b]
        # print(x_end_i_b.shape)
        pseudo_rates_b = neuron_b.construct_ratefn(x_end_i_b)(input_b)
        # print(pseudo_rates_b.shape)
    input_b = neuron_b.linear(pseudo_rates_b, weights_b[Nlayer_b - 1])

    

    ### Spike times for each learned neuron (branch)
    def compute_tout_b(i: ArrayLike) -> Array:
        ### Potential ordinary output spike times
        mask_b = (neurons_b == N_branch-Nout_b + i) & (spike_in_b == False)  # noqa: E712
        Kout_b = jnp.sum(mask_b)  # Number of ordinary output spikes
        t_out_ord_b = times_b[jnp.argmax(mask_b)]

        ### Pseudospike time
        t_out_pseudo_b = neuron_b.t_pseudo(x_end_b[:, N_branch-Nout_b + i], input_b[i], 1, config)

        ### Output spike time
        t_out_b = jnp.where(0 < Kout_b, t_out_ord_b, t_out_pseudo_b)

        return t_out_b
    
    t_outs_b = vmap(compute_tout_b)(jnp.arange(Nout_b))

    return t_outs_b

def outfn_t(
    neuron_t: AbstractPseudoPhaseOscNeuron, out_t: tuple, p_t: list, config: dict
):
    """
    Computes output spike times given simulation results.
    """
    ### Unpack arguments
    Nin_t: int = config["Nin_t"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_t: int = config["Nout_t"]
    Num_points: int = config["Num_points"]


    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t


    weights_t = p_t[0]
    times_t: Array = out_t[0]
    spike_in_t: Array = out_t[1]
    neurons_t: Array = out_t[2]
    x_t: Array = out_t[3]
    # print(x_b.shape, x_t.shape)

    

    ### Run network as feedforward rate ANN (trunk)
    Kord_t = jnp.sum(neurons_t >= 0)  # Number of ordinary spikes
    x_end_t = x_t[Kord_t]
    # print(x_end_t.shape)
    pseudo_rates_t = jnp.zeros((Nin_t, Num_points))
    for i in range(Nlayer_t - 1):
        # print(pseudo_rates_t.shape, weights_t[i].shape)
        input_t = neuron_t.linear(pseudo_rates_t, weights_t[i])
        # print(input_t.shape)
        x_end_i_t = x_end_t[:, i * Nhidden_t : (i + 1) * Nhidden_t]
        # x_end_i_t = jnp.repeat(x_end_i_t, Num_points, axis=0)
        x_end_i_t = jnp.repeat(x_end_i_t.reshape(x_end_i_t.shape[0],x_end_i_t.shape[1],1), Num_points, axis=-1)
        # print(x_end_i_t.shape)
        pseudo_rates_t = neuron_t.construct_ratefn(x_end_i_t)(input_t)
        # print(pseudo_rates_t.shape)
    input_t = neuron_t.linear(pseudo_rates_t, weights_t[Nlayer_t - 1])

   
    

    ### Spike times for each learned neuron (trunk)
    def compute_tout_t(i: ArrayLike) -> Array:
        ### Potential ordinary output spike times
        mask_t = (neurons_t == N_trunk-Nout_t + i) & (spike_in_t == False)  # noqa: E712
        Kout_t = jnp.sum(mask_t)  # Number of ordinary output spikes
        t_out_ord_t = times_t[jnp.argmax(mask_t)]

        ### Pseudospike time
        t_out_pseudo_t = neuron_t.t_pseudo(x_end_t[:, N_trunk-Nout_t + i], input_t[i], 1, config)

        ### Output spike time
        t_out_t = jnp.where(0 < Kout_t, t_out_ord_t, t_out_pseudo_t)

        return t_out_t

    t_outs_t = vmap(compute_tout_t)(jnp.arange(Nout_t))

    return t_outs_t


def lossfn(
    t_out: jnp.ndarray, target: jnp.ndarray, config: dict, p_b: list,
    p_t: list
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the mean squared error (MSE) loss for regression and a simple accuracy metric.

    Args:
        t_out: Array of predicted output spike times of shape (Nout,). For regression, Nout=1.
        target: Scalar (or single-element array) representing the true value (e.g. x^2).
        config: Configuration dictionary. An optional key 'reg_threshold' can be provided,
                which defines a threshold for considering the prediction "accurate".

    Returns:
        A tuple (loss, correct) where:
            - loss: The squared error (MSE) between the prediction and target.
            - correct: A boolean flag (as a JAX array) indicating whether the absolute error 
              is below the given threshold.
    """

   

    # subtract first
    loss = (t_out-target)**2

    # regularization
    # print((p_b[0][1])**2)
    # print(p_t)
    # for i in range(len(p_b[0])):

    # p_b[i] for in range(len(p_b))
    # b_reg = jnp.mean((jnp.array(p_b))**2)
    b_reg_l2 = sum([jnp.sum(bw**2) for bw in p_b[0]])
    t_reg_l2 = sum([jnp.sum(tw**2) for tw in p_t[0]])
    b_reg_l1 = sum([jnp.sum(jnp.abs(bw)) for bw in p_b[0]])
    t_reg_l1 = sum([jnp.sum(jnp.abs(tw)) for tw in p_t[0]])
    
    
    # t_reg = jnp.mean((jnp.array(p_t))**2)
    lam = 1e-7
    # loss = loss + lam * (b_reg_l2 + t_reg_l2)
    # loss = loss + lam * (b_reg_l1 + t_reg_l1)


    # loss = (jnp.tan(t_out[0]*jnp.pi/2-jnp.pi) - target) ** 2
 


    # Define a threshold for acceptable error (default threshold 0.1)
    threshold = config.get("reg_threshold", 0.1)
    # correct = jnp.abs(t_out[0] - target) < threshold
    correct = 0
    
    return loss, correct




def simulatefn(
    neuron_b: AbstractPseudoPhaseOscNeuron,
    neuron_t: AbstractPseudoPhaseOscNeuron,
    p_b: list,
    p_t: list,
    input_b: Float[Array, "Batch Nin_b"],
    input_t: Float[Array, "Batch Nin_t"],
    labels: Float[Array, " Batch"],
    config: dict,
) -> tuple[Array, Array]:
    """
    Simulates the network and computes the loss and accuracy for batched input.
    """
    outs_b, outs_t = vmap(eventffwd, in_axes=(None, None, None, None, 0, 0, None))(neuron_b, neuron_t, p_b, p_t, input_b, input_t, config)
    # print('p_b',p_b, 'p_t', p_t)
    t_outs_b = vmap(outfn_b, in_axes=(None, 0, None, None))(neuron_b, outs_b, p_b, config)
    t_outs_t = vmap(outfn_t, in_axes=(None, 0, None, None))(neuron_t, outs_t, p_t, config)
    

    
    # both (subtract first)
    t_outs_t_reshape = t_outs_t.reshape(t_outs_t.shape[0], t_outs_t.shape[2], -1, 2)
    t_outs_b_reshape = t_outs_b.reshape(t_outs_b.shape[0], -1, 2)
    t_outs_t_true = t_outs_t_reshape[:, :, :, 1] - t_outs_t_reshape[:, :, :, 0]
    t_outs_b_true = t_outs_b_reshape[:, :, 1] - t_outs_b_reshape[:, :, 0]
    t_final = jnp.einsum('bo,bto->bt', t_outs_b_true, t_outs_t_true)

    

    loss, correct = vmap(lossfn, in_axes=(0, 0, None, None, None))(t_final, labels, config, p_b, p_t)
    mean_loss = jnp.mean(loss)

    # # regularization
    # b_loss = jnp.mean((jnp.array(p_b))**2)
    # t_loss = jnp.mean((np.array(p_t))**2)

    accuracy = jnp.mean(correct)
    return mean_loss, accuracy


def probefn(
    neuron_b: AbstractPseudoPhaseOscNeuron,
    neuron_t: AbstractPseudoPhaseOscNeuron,
    p_b: list,
    p_t: list,
    input_b: Float[Array, "Batch Nin_b"],
    input_t: Float[Array, "Batch Nin_t"],
    labels: Float[Array, " Batch"],
    config: dict,
) -> tuple:
    """
    Computes several metrics.
    """

    ### Unpack arguments
    T: float = config["T"]
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b
    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk
    Nbatch: int = config["Nbatch"]

    ### Batched functions
    @vmap
    def batch_eventffwd(input_b, input_t):
        return eventffwd(neuron_b, neuron_t, p_b, p_t, input_b, input_t, config)


    @vmap
    def batch_outfn_b(outs_b):
        return outfn_b(neuron_b, outs_b, p_b, config)
    
    @vmap
    def batch_outfn_t(outs_t):
        return outfn_t(neuron_t, outs_t, p_t, config)

    @vmap
    def batch_lossfn(t_outs, labels):
        return lossfn(t_outs, labels, config, p_b, p_t)

    ### Run network
    outs_b, outs_t = batch_eventffwd(input_b, input_t)
    times_b: Array = outs_b[0]
    spike_in_b: Array = outs_b[1]
    neurons_b: Array = outs_b[2]
    times_t: Array = outs_t[0]
    spike_in_t: Array = outs_t[1]
    neurons_t: Array = outs_t[2]
    # t_outs_b, t_outs_t = batch_outfn(outs_b, outs_t)
    t_outs_b = batch_outfn_b(outs_b)
    t_outs_t = batch_outfn_t(outs_t)

  
   

    # both (subtract first)
    t_outs_t_reshape = t_outs_t.reshape(t_outs_t.shape[0], t_outs_t.shape[2], -1, 2)
    t_outs_b_reshape = t_outs_b.reshape(t_outs_b.shape[0], -1, 2)
    t_outs_t_true = t_outs_t_reshape[:, :, :, 1] - t_outs_t_reshape[:, :, :, 0]
    t_outs_b_true = t_outs_b_reshape[:, :, 1] - t_outs_b_reshape[:, :, 0]
    t_final = jnp.einsum('bo,bto->bt', t_outs_b_true, t_outs_t_true)

   
   

    loss, correct = batch_lossfn(t_final, labels)
    mean_loss = jnp.mean(loss)
    acc = jnp.mean(correct)

 

    mean_loss_ord = 0
    acc_ord = 0


    ### Activity and silent neurons (branch)
    mask_b = (spike_in_b == False) & (neurons_b < N_branch - Nout_b) & (neurons_b >= 0)  # noqa: E712
    activity_b = jnp.sum(mask_b) / (Nbatch * (N_branch - Nout_b))
    silent_neurons_b = jnp.isin(
        jnp.arange(N_branch - Nout_b), jnp.where(mask_b, neurons_b, -1), invert=True
    )

    ### Activity and silent neurons (trunk)
    mask_t = (spike_in_t == False) & (neurons_t < N_trunk - Nout_t) & (neurons_t >= 0)  # noqa: E712
    activity_t = jnp.sum(mask_t) / (Nbatch * (N_trunk - Nout_t))
    silent_neurons_t = jnp.isin(
        jnp.arange(N_trunk - Nout_t), jnp.where(mask_t, neurons_t, -1), invert=True
    )

   

    activity_first_b = 0
    activity_first_t = 0
    silent_neurons_first_b=0
    silent_neurons_first_t=0


    ### Pack results in dictionary
    metrics = {
        "loss": mean_loss,
        "acc": acc,
        "loss_ord": mean_loss_ord,
        "acc_ord": acc_ord,
        "activity_b": activity_b,
        "activity_t": activity_t,
        "activity_first_b": activity_first_b,
        "activity_first_t": activity_first_t,
    }
    silents = {
        "silent_neurons_b": silent_neurons_b,
        "silent_neurons_t": silent_neurons_t,
        "silent_neurons_first_b": silent_neurons_first_b,
        "silent_neurons_first_t": silent_neurons_first_t,
    }

    return metrics, silents


# %%
############################
### Training
############################


def run(
    neuron_b: AbstractPseudoPhaseOscNeuron,
    neuron_t: AbstractPseudoPhaseOscNeuron,
    config: dict,
    progress_bar: str | None = None,
) -> dict:
    """
    Trains a feedforward network with time-to-first-spike encoding on MNIST.

    The pixel values are binned into `Nin_virtual+1` bins, each corresponding to an
    input spike time except for the last bin, which is ignored. The effect of all inputs
    in each bin is captured by a virtual input neuron under the hood to speed up the
    simulation. See `transform_image` and `eventffwd` for details. The trained
    parameters `p` are the feedforward weights of the network and the initial phases of
    the neurons.

    Args:
        neuron:
            Phase oscillator model including pseudodynamics.
        config:
            Simulation configuration. Needs to contain the following items:
                `seed`: Random seed
                `Nin`: Number of input neurons, has to be 28*28 for MNIST
                `Nin_virtual`: Number of virtual input neurons
                `Nhidden`: Number of hidden neurons per layer
                `Nlayer`: Number of layers
                `Nout`: Number of output neurons, has to be 10 for MNIST
                `w_scale`: Scale of the initial weights
                `T`: Trial duration
                `K`: Maximal number of simulated ordinary spikes
                `dt`: Integration time step (for state traces)
                `gamma`: Regularization strength
                `Nbatch`: Batch size
                `lr`: Learning rate
                `tau_lr`: Learning rate decay time constant
                `beta1`: Adabelief parameter
                `beta2`: Adabelief parameter
                `p_flip`: Probability of flipping input pixels
                `Nepochs`: Number of epochs
        progress_bar:
            Whether to use 'notebook' or 'script' tqdm progress bar or `None`.
    Returns:
        A dictionary containing detailed learning dynamics.
    """

    ### Unpack arguments
    seed: int = config["seed"]
    Nin_virtual_b: int = config["Nin_virtual_b"]
    Nin_virtual_t: int = config["Nin_virtual_t"]
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b
    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk

    Nepochs: int = config["Nepochs"]
    p_flip: float = config["p_flip"]
    lr: float = config["lr"]
    tau_lr: float = config["tau_lr"]
    beta1: float = config["beta1"]
    beta2: float = config["beta2"]
    theta_b = neuron_b.Theta()
    theta_t = neuron_t.Theta()
    if progress_bar == "notebook":
        trange = trange_notebook
    elif progress_bar == "script":
        trange = trange_script
    else:
        trange = range

    ### Set up the simulation

    # Gradient
    @jit
    @partial(value_and_grad, has_aux=True)
    def gradfn(
        p, input_b: Float[Array, "Batch Nin_b"], input_t: Float[Array, "Batch Nin_t"], labels: Float[Array, " Batch"]
    ) -> tuple[Array, Array]:
        p_b = p[0]
        p_t = p[1]
        loss, acc = simulatefn(neuron_b, neuron_t, p_b, p_t, input_b, input_t, labels, config)
        return loss, acc


    # Optimization step
    @jit
    def trial(
        p,
        input_b: Float[Array, "Batch Nin_b"],
        input_t: Float[Array, "Batch Nin_t"],
        labels: Float[Array, " Batch"],
        opt_state: optax.OptState,
    ) -> tuple:
        (loss, acc), grad = gradfn(p, input_b, input_t, labels)
        updates, opt_state = optim.update(grad, opt_state)
        p = optax.apply_updates(p, updates)  # type: ignore
        theta_b = neuron_b.Theta()  
        theta_t = neuron_t.Theta()  
        p[0][1] = jnp.clip(p[0][1], 0, theta_b) 
        p[1][1] = jnp.clip(p[1][1], 0, theta_t) 
        return loss, acc, p, opt_state

    # Probe network
    @jit
    def jprobefn(p_b, p_t, input_b, input_t, labels):
        return probefn(neuron_b, neuron_t, p_b, p_t, input_b, input_t, labels, config)

    def probe(p_b: list, p_t: list) -> dict:
        metrics = {
            "loss": 0.0,
            "acc": 0.0,
            "loss_ord": 0.0,
            "acc_ord": 0.0,
            "activity_b": 0.0,
            "activity_t": 0.0,
            "activity_first_b": 0.0,
            "activity_first_t": 0.0,
        }
        silents = {
            "silent_neurons_b": jnp.ones(N_branch - Nout_b, dtype=bool),
            "silent_neurons_t": jnp.ones(N_trunk - Nout_t, dtype=bool),
            "silent_neurons_first_b": jnp.ones(N_branch - Nout_b, dtype=bool),
            "silent_neurons_first_t": jnp.ones(N_trunk - Nout_t, dtype=bool),
        }
        steps = len(test_loader)
        for data in test_loader:
            input_b, input_t, labels = jnp.array(data[0]), jnp.array(data[1]), jnp.array(data[2])
            metric, silent = jprobefn(p_b, p_t, input_b, input_t, labels)
            metrics = {k: metrics[k] + metric[k] / steps for k in metrics}
            silents = {k: silents[k] & silent[k] for k in silents}
        for k, v in silents.items():
            metrics[k] = jnp.mean(v).item()
        return metrics

    ### Simulation

    # Data
    torch.manual_seed(seed)
    train_loader, test_loader = load_data(datasets.MNIST, "data", config)

    # Parameters
    key = random.PRNGKey(seed)
    key, weights_b, weights_t = init_weights(key, config)
    phi0_b, phi0_t = init_phi0(neuron_b, neuron_t, config)
    p_b = [weights_b, phi0_b]
    p_t = [weights_t, phi0_t]
    p = [p_b, p_t]
    p_init_b = [weights_b, phi0_b]
    p_init_t = [weights_t, phi0_t]
    p_init = [p_init_b, p_init_t]

    # Optimizer
    # schedule = optax.exponential_decay(lr, int(tau_lr * len(train_loader)), 1 / jnp.e)
    # optim = optax.adabelief(schedule, b1=beta1, b2=beta2)
   
    optim = optax.adabelief(lr, b1=beta1, b2=beta2)
  
    opt_state = optim.init(p)

    # Metrics
    metrics: dict[str, Array | list] = {k: [v] for k, v in probe(p_b, p_t).items()}

    # timing
    times=[]
    train_loss = []
    test_loss_list=[]

    # if os.path.exists('experiments/deeponet/qif_deeponet.pkl'):
    #         print(f"Loading parameters from {'experiments/deeponet/qif_deeponet.pkl'}")
    #         p = safe_load_params('experiments/deeponet/qif_deeponet.pkl')
      
    # else:
    #     print(f"No saved parameters found. Training from scratch.")

    # Training
    pre_loss = 10000000
    for epoch in trange(Nepochs):
        epoch_loss = 0
        batch_count = 0
        for data in train_loader:
            input_b, input_t, labels = jnp.array(data[0]), jnp.array(data[1]), jnp.array(data[2])
            # key, input_b, input_t = flip(key, input_b, input_t)

            st=time.time()
            # print(p)
            loss, acc, p, opt_state = trial(p, input_b, input_t, labels, opt_state)

            et=time.time()
            times.append(et-st)
            p_b = p[0]
            p_t = p[1]
            epoch_loss += loss
            batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count
        train_loss.append(loss)

        for data_test in test_loader:
            input_b_test, input_t_test, labels_test = jnp.array(data_test[0]), jnp.array(data_test[1]), jnp.array(data_test[2])
            outs_b_test, outs_t_test = vmap(eventffwd, in_axes=(None, None, None, None, 0, 0, None))(neuron_b, neuron_t, p_b, p_t, input_b_test, input_t_test, config)
            t_outs_b_test = vmap(outfn_b, in_axes=(None, 0, None, None))(neuron_b, outs_b_test, p_b, config)
            t_outs_t_test = vmap(outfn_t, in_axes=(None, 0, None, None))(neuron_t, outs_t_test, p_t, config)
        
            # both (subtract first)
            t_outs_t_reshape_test = t_outs_t_test.reshape(t_outs_t_test.shape[0], t_outs_t_test.shape[2], -1, 2)
            t_outs_b_reshape_test = t_outs_b_test.reshape(t_outs_b_test.shape[0], -1, 2)
            t_outs_t_true_test = t_outs_t_reshape_test[:, :, :, 1] - t_outs_t_reshape_test[:, :, :, 0]
            t_outs_b_true_test = t_outs_b_reshape_test[:, :, 1] - t_outs_b_reshape_test[:, :, 0]
            t_final_test = jnp.einsum('bo,bto->bt', t_outs_b_true_test, t_outs_t_true_test)
            td = sio.loadmat('experiments/deeponet/train_data.mat')
            scale_use = td['y_train'][:800, :]
            scale_min = jnp.min(scale_use)
            scale_max = jnp.max(scale_use)
            t_final_test = t_final_test * (scale_max - scale_min) + scale_min
            # t_final_test = t_final_test * (jnp.max(labels_test) - jnp.min(labels_test)) + jnp.min(labels_test)
            test_loss = jnp.mean((t_final_test-labels_test)**2)
                
        test_loss_list.append(test_loss)


        if epoch%50==0:
            print('loss:', loss, 'test loss', test_loss)
            if avg_epoch_loss <= pre_loss:
                safe_save_params(p, 'experiments/deeponet/qif_deeponet.pkl')
                pre_loss = avg_epoch_loss
                

        # Probe network
        metric = probe(p_b, p_t)
        metrics = {k: v + [metric[k]] for k, v in metrics.items()}
    # total_time = jnp.sum(jnp.array(times))
    # print('time:', total_time)
    # sio.savemat('loss_his.mat', {'train_loss':train_loss, 'test_loss': test_loss_list})
    # import matplotlib.pyplot as plt
    # Nepochs = epoch + 1
    # plt.loglog(jnp.linspace(1, Nepochs, Nepochs), train_loss, label="train", color="red")
    # plt.loglog(jnp.linspace(1, Nepochs, Nepochs), test_loss_list, label="test", color="blue")
    # plt.xlabel("epoch")
    # plt.ylabel("MSE")
    # plt.title("Loss history")
    # plt.legend()
    # plt.show()
    p_b = p[0]
    p_t = p[1]
    test_data = sio.loadmat('experiments/deeponet/test_data.mat')
    branch_X_test = test_data['f_test'][:800, :]
    trunk_X_test = test_data['x_test'].reshape(-1, 1)[:800, :]
    # trunk_X_test = jnp.concatenate((trunk_X_test, jnp.sin(trunk_X_test), jnp.sin(2*trunk_X_test), jnp.sin(3*trunk_X_test), jnp.cos(trunk_X_test), jnp.cos(2*trunk_X_test), jnp.cos(3*trunk_X_test)), axis=1)
    Y_test = test_data['y_test'][:800, :]

    train_data = sio.loadmat('experiments/deeponet/train_data.mat')
    branch_X_train = train_data['f_train'][:800, :]
    trunk_X_train = train_data['x_train'].reshape(-1, 1)[:800, :]
    Y = train_data['y_train'][:800, :]
    Y_min = jnp.min(Y)
    Y_max = jnp.max(Y)
   
    T = config["T"] 
   
    branch_min = min(jnp.min(branch_X_test), jnp.min(branch_X_train))
    branch_max = max(jnp.max(branch_X_test), jnp.max(branch_X_train))
    test_input_b = (branch_X_test-branch_min)/(branch_max-branch_min)
    test_input_b = (1 - test_input_b)*T
    test_input_t = input_normalization(trunk_X_test, T)
    test_input_t = jnp.repeat(test_input_t.reshape(1, test_input_t.shape[0], test_input_t.shape[1]), test_input_b.shape[0], axis=0)

    outs_b, outs_t = vmap(eventffwd, in_axes=(None, None, None, None, 0, 0, None))(neuron_b, neuron_t, p_b, p_t, test_input_b, test_input_t, config)
    # print('p_b',p_b, 'p_t', p_t)
    t_outs_b = vmap(outfn_b, in_axes=(None, 0, None, None))(neuron_b, outs_b, p_b, config)
    t_outs_t = vmap(outfn_t, in_axes=(None, 0, None, None))(neuron_t, outs_t, p_t, config)
  
    t_outs_t_reshape = t_outs_t.reshape(t_outs_t.shape[0], t_outs_t.shape[2], -1, 2)
    t_outs_b_reshape = t_outs_b.reshape(t_outs_b.shape[0], -1, 2)
    t_outs_t_true = t_outs_t_reshape[:, :, :, 1] - t_outs_t_reshape[:, :, :, 0]
    t_outs_b_true = t_outs_b_reshape[:, :, 1] - t_outs_b_reshape[:, :, 0]
    t_final = jnp.einsum('bo,bto->bt', t_outs_b_true, t_outs_t_true)
    preds = t_final

   
    preds = preds * (Y_max - Y_min) + Y_min

    sio.savemat('experiments/deeponet/qif_deeponet_result.mat', {'y_pred': preds,
                                        'y_true': Y_test,
                                        'loss': np.array(train_loss),
                                        'time': np.array(times)})


    if jnp.any(jnp.isnan(jnp.array(metrics["loss"]))):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )
    metrics = {k: jnp.array(v) for k, v in metrics.items()}
    p_end = p
    metrics["p_init"] = p_init
    metrics["p_end"] = p_end

    return metrics


# %%
############################
### Examples
############################


def run_example(p_b: list, p_t: list, neuron_b: AbstractPseudoPhaseOscNeuron, neuron_t: AbstractPseudoPhaseOscNeuron, config: dict) -> dict:
    """
    Simulates the network for a single example input given the parameters `p`.
    """

    ### Unpack arguments
    seed: int = config["seed"]
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b
    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk

    ### Set up the simulation
    @jit
    def jeventffwd(p_b, p_t, input_b, input_t):
        return eventffwd(neuron_b, neuron_t, p_b, p_t, input_b, input_t, config)

    # @jit
    # def joutfn(out_b, out_t, p_b, p_t):
    #     return outfn(neuron_b, neuron_t, out_b, out_t, p_b, p_t, config)

    @jit
    def joutfn_b(out_b, p_b):
        return outfn_b(neuron_b, out_b, p_b, config)
    
    @jit
    def joutfn_t(out_t, p_t):
        return outfn_t(neuron_t, out_t, p_t, config)

    ### Run simulation

    # Data
    torch.manual_seed(seed)
    _, test_loader = load_data(datasets.MNIST, "data", config)

    input_b, input_t, label = next(iter(test_loader))
    # input, label = jnp.array(input[2]), jnp.array(label[2])
    input_b, input_t, label = jnp.array(input_b[60]), jnp.array(input_t[60]), jnp.array(label[60])
    out_b, out_t = jeventffwd(p_b, p_t, input_b, input_t)
    # t_outs_b, t_outs_t = joutfn(out_b, out_t, p_b, p_t)
    t_outs_b = joutfn_b(out_b, p_b)
    t_outs_t = joutfn_t(out_t, p_t)
    


    ### Prepare results
    times_b: Array = out_b[0]
    spike_in_b: Array = out_b[1]
    neurons_b: Array = out_b[2]
    times_t: Array = out_t[0]
    spike_in_t: Array = out_t[1]
    neurons_t: Array = out_t[2]

    trace_ts_b, trace_xs_b = neuron_b.traces(p_b[1][jnp.newaxis], out_b, config)
    trace_ts_t, trace_xs_t = neuron_t.traces(p_t[1][jnp.newaxis], out_t, config)
    trace_phis_b = trace_xs_b[:, 0]
    trace_phis_t = trace_xs_t[:, 0]
    trace_Vs_b = neuron_b.iPhi(trace_phis_b)
    trace_Vs_t = neuron_t.iPhi(trace_phis_t)

    # branch
    spiketimes_b = []
    for i in range(N_branch):
        times_i_b = times_b[~spike_in_b & (neurons_b == i)]
        spiketimes_b.append(times_i_b)
    # trunk
    spiketimes_t = []
    for i in range(N_trunk):
        times_i_t = times_t[~spike_in_t & (neurons_t == i)]
        spiketimes_t.append(times_i_t)

   
    
    # both (subtract first)
    t_outs_t_reshape = t_outs_t.reshape(t_outs_t.shape[1], -1, 2)
    t_outs_b_reshape = t_outs_b.reshape(-1, 2)
    t_outs_t_true = t_outs_t_reshape[:, :, 1] - t_outs_t_reshape[:, :, 0]
    t_outs_b_true = t_outs_b_reshape[:, 1] - t_outs_b_reshape[:, 0]
    t_final = jnp.einsum('o,to->t', t_outs_b_true, t_outs_t_true)

   
    predicted = t_final[0]

    

    ### Pack results in dictionary
    results = {
        "input_b": input_b,
        "input_t": input_t,
        "label": label,
        "predicted": predicted,
        "trace_ts_b": trace_ts_b,
        "trace_ts_t": trace_ts_t,
        "trace_phis_b": trace_phis_b,
        "trace_phis_t": trace_phis_t,
        "trace_Vs_b": trace_Vs_b,
        "trace_Vs_t": trace_Vs_t,
        "spiketimes_b": spiketimes_b,
        "spiketimes_t": spiketimes_t,
    }
    # print('input',results['input'])
    # print('label', results['label'])
    # print('out spike times', t_outs)
    # print('pred', predicted)

    return results


# %%
############################
### Plotting
############################


def plot_spikes(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b
    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk

    spiketimes_b: Array = example["spiketimes_b"]
    spiketimes_t: Array = example["spiketimes_t"]

    ### Plot spikes
    tick_len = 2
    ax.eventplot(spiketimes_b, colors="k", linewidths=0.5, linelengths=tick_len)
    ax.eventplot(spiketimes_t, colors="r", linewidths=0.5, linelengths=tick_len)
    patch_b = Rectangle((0, Nhidden_b - 1 / 2), T, Nhidden_b, color="k", alpha=0.2, zorder=0)
    patch_t = Rectangle((0, Nhidden_t - 1 / 2), T, Nhidden_t, color="k", alpha=0.2, zorder=0)
    ax.add_patch(patch_b)
    ax.add_patch(patch_t)
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
        Nhidden_b - 1 / 2,
        r"$2^\mathrm{nd}$ hidden",
        ha="right",
        va="bottom",
        color="white",
        zorder=1,
    )
    ax.text(
        T,
        Nhidden_t - 1 / 2,
        r"$2^\mathrm{nd}$ hidden",
        ha="right",
        va="bottom",
        color="white",
        zorder=1,
    )
    ax.text(
        T,
        2 * Nhidden_b - 1 / 2,
        "Output",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )

    ax.text(
        T,
        2 * Nhidden_t - 1 / 2,
        "Output",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )

    ### Formatting
    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks(
        [0, Nhidden_b - 1, 2 * Nhidden_b - 1, N_branch - 1],
        [str(1), str(Nhidden_b), str(2 * Nhidden_b), str(N_branch)],
    )
    ax.set_yticks(
        [0, Nhidden_t - 1, 2 * Nhidden_t - 1, N_trunk - 1],
        [str(1), str(Nhidden_t), str(2 * Nhidden_t), str(N_trunk)],
    )
    ax.set_ylim(-tick_len / 2, N_branch - 1 + tick_len / 2)
    ax.set_ylim(-tick_len / 2, N_trunk - 1 + tick_len / 2)
    ax.set_ylabel("Neuron", labelpad=-0.1)


def plot_error(ax: Axes, metrics: dict, config: dict) -> None:
    ### Unpack arguments
    Nepochs: int = config["Nepochs"]
    acc: Array = metrics["acc"]
    mean_acc = jnp.mean(acc, 0)
    std_acc = jnp.std(acc, 0)
    acc_ord: Array = metrics["acc_ord"]
    mean_acc_ord = jnp.mean(acc_ord, 0)
    std_acc_ord = jnp.std(acc_ord, 0)
    epochs = jnp.arange(1, Nepochs + 2)

    ### Plot classification error
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

    ### Formatting
    ax.set_xlabel("Epochs + 1", labelpad=-1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim(0.01, 1)
    ax.set_ylabel("Test error", labelpad=-3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(formatter)


def plot_traces(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden_b: int = config["Nhidden_b"]
    Nhidden_t: int = config["Nhidden_t"]
    Nlayer_b: int = config["Nlayer_b"]
    Nlayer_t: int = config["Nlayer_t"]
    Nout_b: int = config["Nout_b"]
    Nout_t: int = config["Nout_t"]

    # Total neurons in branch network
    N_branch = Nhidden_b * (Nlayer_b-1)+Nout_b
    
    # Total neurons in trunk network
    N_trunk = Nhidden_t * (Nlayer_t-1)+Nout_t
    
    # Total number of neurons
    N = N_branch + N_trunk

    ### Unpack example
    trace_ts_b: Array = example["trace_ts_b"]
    trace_ts_t: Array = example["trace_ts_t"]
    trace_Vs_b: Array = example["trace_Vs_b"]
    trace_Vs_t: Array = example["trace_Vs_t"]

    ### Plot
    ax.axhline(0, c="gray", alpha=0.3, zorder=-1)
    ax.plot([-0.1, -0.1], [0, 1], c="k", clip_on=False)
    for i in range(10):
        ax.plot(trace_ts_b, trace_Vs_b[:, N_branch - Nout_b + i], color=petroff10[i])
        ax.plot(trace_ts_t, trace_Vs_t[:, N_trunk - Nout_t + i], color=petroff10[i])
        ax.text((i % 5) * 0.15, -4 - (i // 5) * 3, str(i), color=petroff10[i])

    ### Formatting
    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks([])
    ax.set_ylim(-8, 8)
    ax.set_ylabel("Potential $V$")
    ax.spines["left"].set_visible(False)
