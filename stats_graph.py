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

def get_values_grad(path):
    x = np.load(path)[()]

    train_loss = []
    avg_grad = x['avg_grad']
    return avg_grad

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

def plot_grad_graph(valid_loss_RNN,valid_loss_GRU, path, model_type = "RNN"):

    plt.plot(valid_loss_RNN, 'o-')
    plt.plot(valid_loss_GRU, 'o-')
    plt.ylabel("gradient")
    plt.xlabel("time steps")
    plt.title("Average Gradient Per Timestep")
    plt.legend(labels = ["RNN", "GRU"])
    plt.savefig(path + '_avg_grad_time_steps.png', bbox_inches='tight')
    plt.clf()

#avg loss stuff
try:
    lc_path = "RNN_4.1/Average_Loss.npy"
    avg_loss_RNN = get_values(lc_path)
    lc_path = "GRU_4.1/Average_Loss.npy"
    avg_loss_GRU = get_values(lc_path)
    lc_path = "Transformer/Average_Loss.npy"
    avg_loss_TR = get_values(lc_path)
    plot_graph(avg_loss_RNN, avg_loss_GRU, avg_loss_TR, "stats/")
except Exception as e:
    print('no avg loss graph')

#avg grad stuff
try:
    lc_path = "RNN_4.1/Average_Grad.npy"
    avg_grad_RNN = get_values_grad(lc_path)
    lc_path = "GRU_4.1/Average_Grad.npy"
    avg_grad_GRU = get_values_grad(lc_path)
    plot_grad_graph(avg_grad_RNN, avg_grad_GRU, "stats/")
except Exception as e:
    print('no avg grad graph')
