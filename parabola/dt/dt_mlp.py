from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import time
import scipy
import pickle
import os

X_train = np.linspace(-1, 1, 100)
X_train = X_train.reshape(-1, 1)
Y_train = X_train**2

def save_params(params, filename='dt_parabola32.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model parameters saved to {filename}")

def load_params(filename='dt_parabola32.pkl'):
    with open(filename, 'rb') as f:
        params = pickle.load(f)
    print(f"Model parameters loaded from {filename}")
    return params


@jax.custom_jvp
def heaviside(x):
    return jnp.where(x < 0, 0.0, 1.0)


@heaviside.defjvp
def heaviside_jvp(primals, tangents):
    (x,) = primals
    (x_dot,) = tangents
    primal_out = heaviside(x)
    # tangent_out = heaviside(x) * x_dot
    sigma = 0.5
    tangent_out = 1 / (sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(x**2) / (2 * sigma**2)) * x_dot
    return primal_out, tangent_out

class SpikeLinear(eqx.Module):
    linear: eqx.nn.Linear
    sim_length: int
    enable_shift: bool
    threshold_positive: float
    threshold_negative: float

    def __init__(self, linear: eqx.nn.Linear, sim_length: int, enable_shift: bool):
        self.linear = linear
        self.sim_length = sim_length
        self.enable_shift = enable_shift
        self.threshold_positive = 1.0
        self.threshold_negative = -1.0

    def __call__(self, x, membrane_potential):
        x = self.linear(x)
        if (
            self.enable_shift
            and self.threshold_positive is not None
            and self.threshold_negative is not None
        ):
            x = x + (self.threshold_positive + self.threshold_negative) * 0.5 / self.sim_length
        membrane_potential += x
        # spike_positive = (membrane_potential >= self.threshold_positive) * self.threshold_positive
        # spike_negative = (membrane_potential <= self.threshold_negative) * self.threshold_negative
        spike_positive = (
            heaviside(membrane_potential - self.threshold_positive) * self.threshold_positive
        )
        spike_negative = (
            heaviside(-membrane_potential + self.threshold_negative) * self.threshold_negative
        )
        spike = spike_positive + spike_negative
        membrane_potential -= spike
        return spike, membrane_potential


class SpikeMLP(eqx.Module):
    layers: list[SpikeLinear]
    # ann: eqx.nn.MLP
    sim_length: int
    depth: int
    output_shape: list

    def __init__(self, mlp, sim_length):
        self.layers = [
            SpikeLinear(layer, sim_length=sim_length, enable_shift=True) for layer in mlp.layers
        ]
        # self.ann = mlp
        self.sim_length = sim_length
        self.depth = mlp.depth
        self.output_shape = mlp.layers[-1].out_features

    def forward(self, x, membrane_state):
        new_membrane_state = []
        for layer_idx, layer in enumerate(self.layers):
            x, mem_pot = layer(x, membrane_state[layer_idx])
            new_membrane_state.append(mem_pot)
        return x, new_membrane_state

    def __call__(self, x):
        membrane_state = self.init_membrane_state()
        out = jnp.zeros(self.output_shape)
        for t in range(self.sim_length):
            current_output, membrane_state = self.forward(x, membrane_state)
            out += current_output
        return out / self.sim_length

    def init_membrane_state(self):
        return [jnp.zeros((layer.linear.out_features,)) for layer in self.layers]

def create_batches(X, Y, batch_size):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], Y[batch_indices]

def loss_fn(model, x_input, Y):
    Y_pred = jax.vmap(model)(x_input)
    return jnp.mean((Y_pred - Y) ** 2)
    

ann = eqx.nn.MLP(1, 1, 64, 4, activation=jax.nn.relu, key=jax.random.PRNGKey(0))
model = SpikeMLP(ann, sim_length=128)

# check_path = 'dt_parabola64.pkl'
# if os.path.exists(check_path):
#     print(f"Loading parameters from {check_path}")
#     loaded_params = load_params(check_path)
#     model = eqx.combine(loaded_params, static)
#     params, _ = eqx.partition(model, eqx.is_array)
# else:
#     print(f"No saved parameters found. Training from scratch.")

# opt = optax.adam(1e-3)
# schedule = optax.exponential_decay(1e-2, int(1e2 * len(train_loader)), 1 / jnp.e)
opt = optax.adabelief(1e-3, b1=0.9, b2=0.999)
opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))

@eqx.filter_jit
def train_step(model, opt_state, X, Y):
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, X, Y)
    updates, opt_state = opt.update(grad, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, opt_state, loss

# timing
times = []
# loss history
losses = []
batch_size = 10000000
pre_loss = 1000000000
for epoch in range(10000):
    for X_batch, Y_batch in create_batches(X_train, Y_train, batch_size):
        epoch_loss = 0
        batch_count = 0
        st=time.time()
        model, opt_state, batch_loss = train_step(model, opt_state, X_batch, Y_batch)
        et=time.time()
        times.append(et-st)
        epoch_loss += batch_loss
        batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count
    losses.append(avg_epoch_loss)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {avg_epoch_loss}")
        if avg_epoch_loss <= pre_loss:
            final_params, _ = eqx.partition(model, eqx.is_array)
            save_params(final_params, 'dt_parabola128.pkl')
            pre_loss = avg_epoch_loss

X_test = np.linspace(-1, 1, 1000).reshape(-1, 1)
Y_test = X_test**2
Y_pred_test = jax.vmap(model)(X_test)
scipy.io.savemat('dt_parabola_result128.mat', {'y_pred': Y_pred_test,
                                        'y_true': Y_test,
                                        'loss': np.array(losses),
                                        'time': np.array(times)})

