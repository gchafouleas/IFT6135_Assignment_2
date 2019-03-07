import numpy as np
import matplotlib
import matplotlib.pyplot as plt

lc_path = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_5/learning_curves.npy"
x = np.load(lc_path)[()]
# You will need these values for plotting learning curves (Problem 4)
train_loss = []
t_losses = x['train_losses']
v_losses = x['val_losses']
num_epochs = 40
train = len(t_losses)/num_epochs
train_iters = int(train)
for i in range(0, len(t_losses), train_iters):
    train_loss.append(t_losses[(i + train_iters) -1])

valid_loss = []
valid_iter = int(len(v_losses)/num_epochs)
for i in range(0, len(v_losses), valid_iter):
    valid_loss.append(v_losses[(i + valid_iter) -1])

train_pll = []
t_ppl = x['train_ppls']
v_ppl = x['val_ppls']
train = len(t_ppl)/num_epochs
train_iters = int(train)
for i in range(0, len(t_ppl), train_iters):
    train_pll.append(t_ppl[(i + train_iters) -1])

valid_ppl = []
valid_iter = int(len(v_ppl)/num_epochs)
for i in range(0, len(v_ppl), valid_iter):
    valid_ppl.append(v_ppl[(i + valid_iter) -1])

plt.plot(train_loss, 'r--', valid_loss, 'bs')
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Loss Per Epoch")
plt.legend(labels = ["train", "valid"])
plt.show()

plt.plot(train_pll, 'r--', valid_ppl, 'bs')
plt.ylabel("ppl")
plt.xlabel("Epoch")
plt.title("PPL Per Epoch")
plt.legend(labels = ["train", "valid"])
plt.show()