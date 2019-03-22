import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd

def get_values(path):
    x = np.load(lc_path)[()]

    train_loss = []
    avg_loss = x['avg_losses']
    return avg_loss

def plot_graph(train_loss, path, model_type = "RNN"):

    plt.plot(train_loss, 'o-')
    
    plt.ylabel("loss")
    plt.xlabel("time steps")
    plt.title(model_type + " - Loss Per time step")
    plt.savefig(path + model_type + '_avg_loss_time_steps.png', bbox_inches='tight')
    plt.clf()

#loading data from RNN
lc_path = "RNN_4.1/Average_Loss.npy"
avg_loss = get_values(lc_path)
plot_graph(avg_loss, "RNN_4.1/")