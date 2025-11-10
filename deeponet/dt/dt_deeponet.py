from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax

import jax.random as jr
import jax.tree_util as jtu
from tqdm import tqdm
from scipy.io import savemat
import scipy
import pickle
import os
# from jax.experimental import jax2tf
from functools import reduce
from jax.flatten_util import ravel_pytree
from jax import lax
import time


train_data = scipy.io.loadmat('train_data.mat')
branch_X_train = train_data['f_train'][:800, :]
trunk_X_train = train_data['x_train'].reshape(-1, 1)[:800, :]
Y = train_data['y_train'][:800, :]
print(branch_X_train.shape, trunk_X_train.shape, Y.shape)
test_data = scipy.io.loadmat('test_data.mat')
branch_X_test = test_data['f_test'][:800, :]
trunk_X_test = test_data['x_test'].reshape(-1, 1)[:800, :]
Y_test = test_data['y_test'][:800, :]

Y_min = jnp.min(Y)
Y_max = jnp.max(Y)
Y = (Y-Y_min)/(Y_max-Y_min)

branch_min = min(jnp.min(branch_X_test), jnp.min(branch_X_train))
branch_max = max(jnp.max(trunk_X_train), jnp.max(trunk_X_test))
branch_X_train = (branch_X_train-branch_min)/(branch_max-branch_min)
branch_X_test = (branch_X_test-branch_min)/(branch_max-branch_min)



def save_params(params, filename='dt_deeponet32.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump(params, f)
    print(f"Model parameters saved to {filename}")

def load_params(filename='dt_deeponet32.pkl'):
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
    sigma = 0.3
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
        # print(out.shape)
        # out = jnp.zeros((self.output_shape, self.sim_length))
        for t in range(self.sim_length):
            current_output, membrane_state = self.forward(x, membrane_state)
            # out = out.at[:, t].set(current_output) 
            # print(current_output.shape)
            out += current_output
        return out / self.sim_length
        # print(out.shape)
        # return out

    def init_membrane_state(self):
        return [jnp.zeros((layer.linear.out_features,)) for layer in self.layers]

class SpikeDeepONet(eqx.Module):
    branch_net: SpikeMLP
    trunk_net: SpikeMLP
    sim_length: int
    # bias: jnp.ndarray

    def __init__(self, branch_mlp, trunk_mlp, sim_length, key):
        branch_key, trunk_key = jax.random.split(key)
        self.branch_net = SpikeMLP(branch_mlp, sim_length)
        self.trunk_net = SpikeMLP(trunk_mlp, sim_length)
        self.sim_length = sim_length
        # self.bias = jax.random.normal(key, (branch_mlp.layers[0].in_features,))  # Initialize bias


    def __call__(self, branch_input, trunk_input):
        branch_output = jax.vmap(self.branch_net)(branch_input)
        trunk_output = jax.vmap(self.trunk_net)(trunk_input)
      

        # print('trunk', trunk_output.shape)
        return jnp.einsum('bo,to->bt', branch_output, trunk_output)
      

        # return sum.reshape(sum.shape[0], 1)



def create_batches(X, Y, batch_size):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], Y[batch_indices]

def loss_fn(model, branch_input, trunk_input, Y):
    # Y_pred = jax.vmap(model, in_axes=(None, 0))(branch_input, trunk_input)
    Y_pred = model(branch_input, trunk_input)
    # print(Y_pred.shape)
    return jnp.mean((Y_pred - Y) ** 2)
    # return jnp.sum(jnp.linalg.norm(Y_pred - Y, axis=1) / jnp.linalg.norm(Y, axis=1))

if __name__ == "__main__":


    
        N_EPOCHS = 1500
        # N_EPOCHS = 0

      


        key = jr.PRNGKey(42)
        key, init_key, model_key = jr.split(key,3)

      
        ann_branch = eqx.nn.MLP(51, 128, 64, 2, activation=jax.nn.relu, key=init_key)
        ann_trunk = eqx.nn.MLP(1, 128, 64, 2, activation=jax.nn.relu, key=init_key)
        model = SpikeDeepONet(ann_branch, ann_trunk, sim_length=128, key=init_key)
        
        
   
       
        params, static = eqx.partition(model, eqx.is_array)

        # check_path = 'dt_deeponet128.pkl'
        # if os.path.exists(check_path):
        #     print(f"Loading parameters from {check_path}")
        #     loaded_params = load_params(check_path)
        #     model = eqx.combine(loaded_params, static)
        #     params, _ = eqx.partition(model, eqx.is_array)
        # else:
        #     print(f"No saved parameters found. Training from scratch.")

      

      
        optimizer = optax.adabelief(1e-3, b1=0.9, b2=0.999)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        #bp
        @eqx.filter_jit
        def train_step(model, state, branch_input, trunk_input, Y):   

            loss, grad = eqx.filter_value_and_grad(loss_fn)(model, branch_input, trunk_input, Y)
            updates, opt_state = optimizer.update(grad, state, model)
            new_model = eqx.apply_updates(model, updates)
            return new_model, opt_state, loss




        # timing
        times = []
        # loss history
        losses = []
        batch_size = 50

        key = jax.random.PRNGKey(5432)
        sub_key = jr.split(key, N_EPOCHS)
        pre_loss = 1000000000
        for epoch in tqdm(range(N_EPOCHS)):
            epoch_loss = 0
            batch_count = 0
            for X_branch_batch, Y_batch in create_batches(branch_X_train, Y, batch_size):
                # bp
                st=time.time()
                model, opt_state, batch_loss = train_step(model, opt_state, X_branch_batch, trunk_X_train, Y_batch)
                et=time.time()
                times.append(et-st)
                epoch_loss += batch_loss
                batch_count += 1
            
            avg_epoch_loss = epoch_loss / batch_count
            losses.append(avg_epoch_loss)

            if epoch % 500 == 0:
                Y_pred = model(X_branch_batch, trunk_X_train)
                err_l2 = np.linalg.norm(Y_batch - Y_pred) / np.linalg.norm(Y_batch)
                sample_l2 = jnp.mean(jnp.linalg.norm(Y_batch - Y_pred, axis=1) / jnp.linalg.norm(Y_batch, axis=1))
                print(f"Epoch {epoch}, Loss: {avg_epoch_loss}, L2: {err_l2}, sample l2: {sample_l2}")
                if avg_epoch_loss <= pre_loss:
                    final_params, _ = eqx.partition(model, eqx.is_array)
                    save_params(final_params, 'dt_deeponet128.pkl')
                    pre_loss = avg_epoch_loss
           

savemat('dt_deeponet128.mat', {'y_pred': (model(branch_X_test, trunk_X_test))*(Y_max - Y_min) + Y_min,
                                'x':trunk_X_test,
                                        'y_true': Y_test,
                                        'loss': np.array(losses),
                                        'time': np.array(times)})
