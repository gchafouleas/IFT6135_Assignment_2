import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd

def get_values(path):
    x = np.load(path)[()]

    train_loss = []
    avg_loss = x['avg_losses']
    return avg_loss

def plot_graph(valid_loss_RNN,valid_loss_GRU,valid_loss_TR, path, model_type = "RNN"):

    plt.plot(valid_loss_RNN, 'o-')
    plt.plot(valid_loss_GRU, 'o-')
    plt.plot(valid_loss_TR, 'o-')
    plt.ylabel("loss")
    plt.xlabel("time steps")
    plt.title("Average Loss Per time step")
    plt.legend(labels = ["RNN", "GRU", "Transformer"])
    plt.savefig(path + '_avg_loss_time_steps.png', bbox_inches='tight')
    plt.clf()

#loading data from RNN
lc_path = "RNN_4.1/Average_Loss.npy"
avg_loss_RNN = get_values(lc_path)
lc_path = "GRU_4.1/Average_Loss.npy"
avg_loss_GRU = get_values(lc_path)
lc_path = "Transformer/Average_Loss.npy"
avg_loss_TR = get_values(lc_path)
plot_graph(avg_loss_RNN, avg_loss_GRU, avg_loss_TR, "stats/")