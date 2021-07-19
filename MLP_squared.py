# This script tries to see
# Model source: https://sonnet.dev/
# Number of trainable source:
# https://stackoverflow.com/questions/38160940/how-to-count-total-number-of-trainable-parameters-in-a-tensorflow-model
import numpy as np
import sonnet as snt
import tensorflow as tf
import matplotlib.pyplot as plt


# %% Parameters.
# All parameters to be tweaked.
mlp_dimensions = [5, 5, 1]
num_training_iterations = 1000
learning_rate = 1e-3
range_tr = (-1, 1)
range_plt = (-2, 2)
iteration_print = 100

# Parameters depending on the tweaked ones.
x_plot = np.linspace(*range_plt, num=100)


# %% Data generator.
def get_data(low, high):
    x_np = np.random.uniform(low=low, high=high, size=(1, 1))
    y_np = np.square(x_np)
    x_tf = tf.convert_to_tensor(x_np, dtype=tf.float32)
    y_tf = tf.convert_to_tensor(y_np, dtype=tf.float32)
    return x_tf, y_tf


# %% Restart Training.
# Model.
model = snt.nets.MLP(mlp_dimensions)

# Optimizer.
optimizer = snt.optimizers.Adam(learning_rate)

# Starting values for training.
last_iteration = 0
loss_list = []

# %% Training (can be run multiple times for added num_training_iterations iterations).
gradients = None
for iteration in range(last_iteration, last_iteration + num_training_iterations):
    last_iteration = iteration
    x, y = get_data(*range_tr)
    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = tf.compat.v1.losses.mean_squared_error(y, y_pred)
    gradients = tape.gradient(loss, model.trainable_variables)
    loss_list.append(loss.numpy())
    if (iteration + 1) % iteration_print == 0:
        optimizer.apply(gradients, model.trainable_variables)
        print(f"Iteration: {iteration + 1}, Mean Loss: {np.mean(loss_list[-iteration_print:-1])}")


#%% Plot results.
num_trainable_variables = np.sum([np.prod(v.shape) for v in model.trainable_variables])
y_plot = np.array([model(tf.convert_to_tensor([[x_i]], dtype=tf.float32)).numpy().squeeze() for x_i in x_plot])
plt.plot(x_plot, np.square(x_plot), "k-")  # Ground truth
plt.plot(x_plot, y_plot, "k--")            # Model predicted
plt.legend(("Ground truth", "MLP predicted"), loc='upper center')
plt.title(rf"MLP tries to predict $x^2$ with {num_trainable_variables} trainable variables")
plt.show()

#%% Plot loss
start = 100
end = len(loss_list)
plt.plot(range(start, end), loss_list[start:end])
plt.ylabel("Loss (MSE)")
plt.xlabel("Training examples")
plt.show()

# %% Test
x, y = get_data(*range_tr)
y_pred = model(x)
print("x={:.2f}, y={:.3f}, y_pred={:.3f}, diff={:.3f}".format(*(i.numpy().squeeze() for i in (x, y, y_pred, y-y_pred))))
