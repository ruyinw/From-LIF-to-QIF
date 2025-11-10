import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from jax import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from experiments.regression_parabola_test.regression import plot_error, plot_spikes, plot_traces, run, run_example, eventffwd, outfn, forwardfn
# from experiments.regression.regressionqif import plot_spikes, plot_traces, run, run_example, eventffwd, outfn
from spikegd.theta import ThetaNeuron
from spikegd.utils.plotting import (
    cm2inch,
    panel_label,
)

from spikegd.lif import LIFNeuron

from spikegd.olif import OscLIFNeuron

from spikegd.qif import QIFNeuron

plt.style.use("spikegd.utils.plotstyle")


config_theta = {
    "seed": 0,
    # Neuron
    "tau": 6 / jnp.pi,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin": 1,
    "Nin_virtual": 1,  # #Virtual input neurons = #Pixel value bins - 1
    "Nhidden": 64,
    "Nlayer": 4,  # Number of layers
    "Nout": 2,
    "w_scale": 0.9,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": 200,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 0,
    "Nbatch": 100,
    "lr": 1e-3,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 10000}


config_olif = {
    "seed": 0,
    # Neuron
    "tau": 6 / jnp.pi,
    "I0": 5 / 4,
    "V_th": 1.0,
    # Network
    "Nin": 1,
    "Nin_virtual": 1,  # #Virtual input neurons = #Pixel value bins - 1
    "Nhidden": 4,
    "Nlayer": 2,  # Number of layers
    "Nout": 2,
    "w_scale": 0.9,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": 200,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 0,
    "Nbatch": 100,
    "lr": 1e-2,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 500
}


config_qif = {
    "seed": 0,
    # Neuron
    "tau": 6 / jnp.pi,
    "eps": 1e-6,
    "alpha": 10.0,
    # Network
    "Nin": 32,
    "Nin_virtual": 1,  # #Virtual input neurons = #Pixel value bins - 1
    "Nhidden": 4,
    "Nlayer": 2,  # Number of layers
    "Nout": 2,
    "w_scale": 0.9,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": 200,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 0,
    "Nbatch": 100,
    "lr": 1e-2,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 500
}




def run_theta(config: dict) -> dict:
    """
    Wrapper to train a network of Theta neurons with the given configuration.

    See docstring of `run` and article for more information.
    """
    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    # metrics = run(neuron, config, progress_bar="notebook")
    metrics = run(neuron, config, progress_bar="script")
    return metrics


def run_olif(config: dict) -> tuple:
   
    tau, I0, V_th = config["tau"], config["I0"], config["V_th"]
    neuron = OscLIFNeuron(tau, I0, V_th)
    metrics = run(neuron, config, progress_bar="script")
    return metrics


def run_qif(config: dict) -> tuple:
    """
    Wrapper to train a QIF neuron with the given configuration.

    See docstring of `run` and article for more information.
    """
    tau, eps, alpha = config["tau"], config["eps"], config["alpha"]
    neuron = QIFNeuron(tau, eps, alpha)
    metrics = run(neuron, config, progress_bar="script")
    return metrics


seed = 0
samples = 1 # Number of network realizations, decrease to save simulation time
key = random.PRNGKey(seed)
seeds = random.randint(key, (samples,), 0, jnp.uint32(2**32 - 1), dtype=jnp.uint32)
metrics_list = []
for seed in seeds:
    config_theta["seed"] = seed
    metrics = run_theta(config_theta)
    # metrics = run_lif(config_lif)
    # metrics = run_qif(config_qif)
    # metrics = run_olif(config_olif)
    metrics_list.append(metrics)
metrics_example = metrics_list[0]
metrics = jax.tree.map(lambda *args: jnp.stack(args), *metrics_list)


def summarize_metrics(metrics: dict, epoch: int) -> None:
    """
    Print a summary of the metrics at the given epoch.
    """
    summary_metrics = {k: v for k, v in metrics.items() if k not in ["p_init", "p_end"]}
    summary_metrics = jax.tree.map(
        lambda x: jnp.array([jnp.mean(x[:, epoch]), jnp.std(x[:, epoch])]),
        summary_metrics,
    )
    for key, value in summary_metrics.items():
        print(f"{key:<25} {value[0]:.3f} ± {value[1]:.3f}")


print("**Results before training**")
summarize_metrics(metrics, 0)
print()
print("**Results after training**")
summarize_metrics(metrics, -1)


def run_example_theta(p: list, config: dict) -> dict:
    """
    Wrapper to run network on one example input.

    See docstring of `run_example` and article for more information.
    """
    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    metrics = run_example(p, neuron, config)
    return metrics



# def run_example_qif(p: list, config: dict) -> dict:
#     """
#     Wrapper to run network on one example input.

#     See docstring of `run_example` and article for more information.
#     """
#     tau, eps,alpha = config["tau"], config["eps"], config["alpha"]
#     neuron = QIFNeuron(tau, eps, alpha)
#     metrics = run_example(p, neuron, config)
#     return metrics

def run_example_olif(p: list, config: dict) -> dict:
    """
    Wrapper to run network on one example input.

    See docstring of `run_example` and article for more information.
    """
    tau, I0, V_th = config["tau"], config["I0"], config["V_th"]
    neuron = OscLIFNeuron(tau, I0, V_th)
    metrics = run_example(p, neuron, config)
    return metrics



example_init = run_example_theta(metrics_example["p_init"], config_theta)
example_end = run_example_theta(metrics_example["p_end"], config_theta)

# example_init = run_example_qif(metrics_example["p_init"], config_qif)
# example_end = run_example_qif(metrics_example["p_end"], config_qif)

# example_init = run_example_olif(metrics_example["p_init"], config_olif)
# example_end = run_example_olif(metrics_example["p_end"], config_olif)




### Figure
fig = plt.figure(figsize=cm2inch(1.5 * 8.6, 1.5 * 6.0))
gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.5,
    wspace=0.4,
    top=0.94,
    bottom=0.15,
    left=0.12,
    right=0.97,
)

#### Spike plot before learning
# Spike plot before learning for regression
ax = fig.add_subplot(gs[:, 0])

plot_spikes(ax, example_init, config_theta) 
# plot_spikes(ax, example_init, config_qif) 
# plot_spikes(ax, example_init, config_olif) 

ax.set_title("Epoch 0", pad=-1)
# Instead of an inset image, show a text annotation of the input value
ax.text(0.05, 0.95, f"Input: {example_init['input'][0]:.2f}", transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
panel_label(fig, ax, "(a)", x=-0.4, y=0.07)


#### Spike plot after learning
ax = fig.add_subplot(gs[:, 1])

plot_spikes(ax, example_end, config_theta)
ax.set_title(f"Epoch {config_theta['Nepochs']}", pad=-1)

# plot_spikes(ax, example_end, config_qif)
# ax.set_title(f"Epoch {config_qif['Nepochs']}", pad=-1)

# plot_spikes(ax, example_end, config_olif)
# ax.set_title(f"Epoch {config_olif['Nepochs']}", pad=-1)

ax.tick_params(labelleft=False)
ax.set_ylabel("")

### Trace output
ax = fig.add_subplot(gs[0, 2])

plot_traces(ax, example_end, config_theta)

# plot_traces(ax, example_end, config_qif)

# plot_traces(ax, example_end, config_olif)

panel_label(fig, ax, "(b)", x=-0.4, y=0.0)

def plot_loss(ax, metrics: dict, config: dict) -> None:
    """
    Plots the mean squared error (MSE) loss over training epochs.
    """
    # Assume metrics["loss"] might be a 2D array with shape (runs, epochs)
    loss = metrics["loss"]
    # If there's only one run, squeeze out the extra dimension:
    loss = jnp.squeeze(loss)  # Now loss.shape should match the number of epochs
    epochs = jnp.arange(1, loss.size + 1)
    ax.plot(epochs, loss.reshape(-1,), label="MSE Loss", color="C0", zorder=1)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Loss over Training")
    ax.legend()


#### Loss Plot (MSE)
ax = fig.add_subplot(gs[1, 2])

plot_loss(ax, metrics, config_theta)

# plot_loss(ax, metrics, config_qif)

# plot_loss(ax, metrics, config_olif)

panel_label(fig, ax, "(c)", x=-0.4, y=0.0)

plt.show()

# import scipy.io as sio
# IF_loss = sio.loadmat('experiments/regression_parabola/loss_comparison/IF500.mat')['loss']
# LIF_loss = sio.loadmat('experiments/regression_parabola/loss_comparison/LIF32t.mat')['loss']
# loss = metrics["loss"]
# # If there's only one run, squeeze out the extra dimension:
# loss = jnp.squeeze(loss)  # Now loss.shape should match the number of epochs
# epochs = jnp.arange(1, loss.size + 1)
# # sio.savemat('experiments/regression/loss_comparison/LIF8t.mat', {'loss': loss.reshape(-1,)})
# plt.figure()
# plt.semilogy(epochs, loss.reshape(-1,), label="QIF")
# # plt.loglog(epochs, IF_loss.reshape(-1,), label="IF")
# plt.semilogy(epochs, LIF_loss.reshape(-1,), label="LIF")
# plt.legend()
# plt.show()



# plt.figure(figsize=(6, 4))
# plt.plot(example_init['input'], example_end['label'], label="Ground Truth: $y=x^2$", color="blue")
# plt.plot(example_init['input'], example_end['predicted'], label="Learned Prediction", color="red", linestyle="--")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("Comparison of Ground Truth and Learned Function")
# plt.legend()
# plt.show()


import matplotlib.pyplot as plt
from jax import vmap
import jax.numpy as jnp
from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron, AbstractPseudoIFNeuron

# def encode_input_thermometer(x_value, num_neurons):
#     """Encodes a scalar input using thermometer coding."""
#     # Create thresholds
#     # log_space = jnp.logspace(-0.5, 0, num_neurons//2)
#     # thresholds = jnp.concatenate([-jnp.flip(log_space), log_space])
#     thresholds = jnp.linspace(-1, 1, num_neurons+1)[:-1]
    
#     # Generate binary pattern where neurons activate if input exceeds threshold
#     encoding = (x_value >= thresholds).astype(jnp.float32)
    
#     return encoding

def gaussian_receptive_field_encoding(x_value, min_val, max_val, num_neurons, beta=1.5):
    """
    Encodes a scalar input using Gaussian receptive fields according to the described method.
    
    Args:
        x_value: The input value to encode
        min_val: Minimum value of the data range
        max_val: Maximum value of the data range
        num_neurons: Number of neurons to use for encoding
        beta: Parameter controlling the width of receptive fields (1.0-2.0, default 1.5)
        
    Returns:
        Array of spike times for each encoding neuron
    """
    # Ensure x_value is within the specified range
    # x_value = jnp.clip(x_value, min_val, max_val)
    
    # Calculate centers for each neuron's receptive field
    i_values = jnp.arange(1, num_neurons + 1)
    # centers = min_val + (2*i_values - 3)/2 * (max_val - min_val)/(num_neurons - 2)
    centers = jnp.linspace(min_val, max_val, num_neurons)
    
    # Calculate width of receptive fields
    # sigma = 1.0 / (beta * (max_val - min_val)/(num_neurons - 2))
    sigma = (1.0 / beta) * (max_val - min_val) / (num_neurons - 2.0)
    
    # Calculate response values using Gaussian function
    responses = jnp.exp(-((x_value - centers) ** 2) / (2 * sigma ** 2))
    
    # Normalize responses to [0, 1] range
    responses = responses / jnp.max(responses)
    
    # print(responses)
    
    # Convert to spike times (higher response = earlier spike)
    T_max = 2.0  # Maximum time as described
    spike_times = (1.0 - responses) * T_max
    
    # # Neurons with times > 9 don't fire
    # # Use a high value (e.g., infinity) for non-firing neurons
    # spike_times = jnp.where(spike_times > 1, jnp.inf, spike_times)
    
    return spike_times


def plot_parabola(neuron: AbstractPhaseOscNeuron, p, config):
# def plot_parabola(neuron: AbstractPseudoIFNeuron, p, config):
    """
    Plots the ground truth parabola y = x^2 and the learned prediction of the network.
    
    Args:
        neuron: The spiking neuron model.
        p: The network parameters (e.g. learned weights and initial phases).
        config: Configuration dictionary (must include keys like 'Nin', 'Nhidden', 'Nlayer',
                'Nout', 'T', etc. for simulation).
    """
    # Generate a set of test inputs uniformly in [0, 1]
    x_vals = jnp.linspace(-1, 1, 1000).reshape(-1, 1)
    # print(x_vals)

    min_val = jnp.min(x_vals)
    max_val = jnp.max(x_vals)

    # encoded_inputs = jnp.zeros((1000, 32))
    # for i, x in enumerate(x_vals):
    #     # encoded_inputs = encoded_inputs.at[i].set(encode_input_thermometer(x, 32))
    #     spike_times = gaussian_receptive_field_encoding(
    #         x, min_val, max_val, 32
    #     )
    #     encoded_inputs = encoded_inputs.at[i].set(spike_times)

    nomalized = (x_vals - jnp.min(x_vals))/(jnp.max(x_vals)-jnp.min(x_vals))
    # nomalized = (x_vals - jnp.min(x_vals)+1e-6)/(jnp.max(x_vals)-jnp.min(x_vals)+1e-6)
    encoded_inputs=(1-nomalized)*config_theta["T"]
    encoded_inputs=encoded_inputs.reshape(-1,1)


    
    # Run the network simulation on each input using eventffwd.
    # vmap over the batch dimension (each input sample).
    outs = vmap(eventffwd, in_axes=(None, None, 0, None))(
        neuron, p, encoded_inputs, config
    )
    
    # Decode the network’s continuous output using the regression output decoder.
    # Each prediction will be an array of shape (1,), so squeeze to get a 1D array.
    t_outs = vmap(outfn, in_axes=(None, 0, None, None))(
        neuron, outs, p, config
    )
    # print(t_outs)
    # preds = jnp.squeeze(t_outs)  # learned predictions (shape: (100,))
    preds = jnp.squeeze(t_outs[:,1]-t_outs[:,0])
    
    
    # Compute the ground truth: y = x^2.
    ground_truth = (x_vals.squeeze()) ** 2

    fn_dydx = jax.grad(forwardfn, argnums=2)
    dydx = jax.vmap(fn_dydx, in_axes=(None, None, 0, None))(neuron, p, encoded_inputs, config)

    print(jnp.mean((ground_truth-preds)**2))
    print(preds)

    # Plot the curves
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals.squeeze(), ground_truth, label="Ground Truth: $y=x^2$", color="blue")
    plt.plot(x_vals.squeeze(), preds, label="Learned Prediction", color="red", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Ground Truth and Learned Function")
    plt.legend()
    plt.show()
    plt.savefig('regression.png')

    # Plot grad
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals.squeeze(), -dydx)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient")
    plt.legend()
    plt.show()


from spikegd.theta import ThetaNeuron
neuron = ThetaNeuron(config_theta["tau"], config_theta["I0"], config_theta["eps"])
plot_parabola(neuron, metrics_example["p_end"], config_theta)


# from spikegd.qif import QIFNeuron
# neuron = QIFNeuron(config_qif["tau"], config_qif["eps"], config_qif["alpha"])
# plot_parabola(neuron, metrics_example["p_end"], config_qif)


# from spikegd.olif import OscLIFNeuron
# neuron = OscLIFNeuron(config_olif["tau"], config_olif["I0"], config_olif["V_th"])
# plot_parabola(neuron, metrics_example["p_end"], config_olif)
