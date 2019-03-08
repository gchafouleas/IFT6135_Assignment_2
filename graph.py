import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd

def get_values(path, file_path):
    x = np.load(lc_path)[()]
    times = []
    with open(file_path, 'rt') as file:
       time = 0
       for line in file:
            test = line.split()
            time += float(test[len(test)-1])
            times.append(time)


    train_loss = []
    t_losses = x['train_losses']
    v_losses = x['val_losses']
    num_epochs = 40
    train = len(t_losses)/num_epochs
    train_iters = int(train)
    for i in range(0, len(t_losses), train_iters):
        train_loss.append(t_losses[(i + train_iters) -1] /train_iters)

    valid_loss = []
    valid_iter = int(len(v_losses)/num_epochs)
    for i in range(0, len(v_losses), valid_iter):
        valid_loss.append(v_losses[(i + valid_iter) -1] /valid_iter)

    train_ppl = []
    t_ppl = x['train_ppls']
    v_ppl = x['val_ppls']
    train = len(t_ppl)/num_epochs
    train_iters = int(train)
    for i in range(0, len(t_ppl), train_iters):
        train_ppl.append(t_ppl[(i + train_iters) -1])

    valid_ppl = []
    valid_iter = int(len(v_ppl)/num_epochs)
    for i in range(0, len(v_ppl), valid_iter):
        valid_ppl.append(v_ppl[(i + valid_iter) -1])

    return train_loss, valid_loss, train_ppl, valid_ppl, times

def plot_graphs(train_loss, valid_loss, train_ppl, valid_ppl, times):

    plt.plot(train_loss, 'r--', valid_loss, 'bs')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title("Loss Per Epoch")
    plt.legend(labels = ["train", "valid"])
    plt.show()

    plt.plot(times, train_loss, 'r--')
    plt.plot(times, valid_loss, 'bs')
    plt.ylabel("Loss")
    plt.xlabel("wall clock time")
    plt.title("Loss Per wall clock")
    plt.legend(labels = ["train", "valid"])
    plt.show()

    plt.plot(train_ppl, 'r--', valid_ppl, 'bs')
    plt.ylabel("ppl")
    plt.xlabel("Epoch")
    plt.title("PPL Per Epoch")
    plt.legend(labels = ["train", "valid"])
    plt.show()

    plt.plot(times, train_ppl, 'r--')
    plt.plot(times, valid_ppl, 'bs')
    plt.ylabel("Loss")
    plt.xlabel("wall clock time")
    plt.title("Loss Per wall clock")
    plt.legend(labels = ["train", "valid"])
    plt.show()

#loading data from RNN
lc_path = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_27/learning_curves.npy"
filepath = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_27/log.txt"
train_loss, valid_loss_RNN, train_ppl, valid_ppl_RNN, times = get_values(lc_path, filepath)
plot_graphs(train_loss, valid_loss_RNN, train_ppl, valid_ppl_RNN, times)

#loading data from GRU
lc_path = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_27/learning_curves.npy"
filepath = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_27/log.txt"
train_loss, valid_loss_GRU, train_ppl, valid_ppl_GRU, times = get_values(lc_path, filepath)
plot_graphs(train_loss, valid_loss_GRU, train_ppl, valid_ppl_GRU, times)

#loading data from TRANSFORMER
lc_path = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_27/learning_curves.npy"
filepath = "RNN_ADAM_model=RNN_optimizer=ADAM_initial_lr=0.0001_batch_size=20_seq_len=35_hidden_size=1500_num_layers=2_dp_keep_prob=0.35_save_best_27/log.txt"
train_loss, valid_loss_TR, train_ppl, valid_ppl_TR, times = get_values(lc_path, filepath)
plot_graphs(train_loss, valid_loss_TR, train_ppl, valid_ppl_TR, times)

#plot all architecture graphs
plt.plot(times, valid_loss_RNN, 'r--')
plt.plot(times, valid_loss_GRU, 'bs')
plt.plot(times, valid_loss_TR, 'gs')
plt.ylabel("Loss")
plt.xlabel("wall clock time")
plt.title("Loss Per wall clock")
plt.legend(labels = ["RNN", "GRU", "TRANSFORMER"])
plt.show()

plt.plot(valid_loss_RNN, 'r--')
plt.plot(valid_loss_GRU, 'bs')
plt.plot(valid_loss_TR, 'gs')
plt.ylabel("Loss")
plt.xlabel("epoch")
plt.title("Loss Per epoch")
plt.legend(labels = ["RNN", "GRU", "TRANSFORMER"])
plt.show()