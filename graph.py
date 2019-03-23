import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import re
import pandas as pd

def get_values(path, file_path):
    x = np.load(path)[()]
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

def plot_graphs(train_loss, valid_loss, train_ppl, valid_ppl, times, path, model_type = "RNN"):

    plt.plot(train_loss, 'o-', valid_loss, 'o-')
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.title(model_type + " - Loss Per Epoch")
    plt.legend(labels = ["train", "valid"])
    plt.savefig(path + model_type + '_loss_epoch.png', bbox_inches='tight')
    plt.clf()

    plt.plot(times, train_loss, 'o-')
    plt.plot(times, valid_loss, 'o-')
    plt.ylabel("Loss")
    plt.xlabel("wall clock time")
    plt.title(model_type + " - Loss Per wall clock")
    plt.legend(labels = ["train", "valid"])
    plt.savefig(path + model_type + '_loss_clock.png', bbox_inches='tight')
    plt.clf()

    smallest_train_ppl = int(np.amin(train_ppl))
    smallest_valid_ppl = int(np.amin(valid_ppl))
    smallest_value = "Train ppl:{} valid ppl: {} ".format(smallest_train_ppl,smallest_valid_ppl)
    plt.plot(train_ppl, 'o-', valid_ppl, 'o-')
    plt.ylabel("ppl")
    plt.xlabel("Epoch")
    plt.title(model_type + " - PPL Per Epoch")
    plt.legend(labels = ["train: {}".format(smallest_train_ppl), "valid: {}".format(smallest_valid_ppl)])
    plt.savefig(path + model_type + '_ppl_epoch.png', bbox_inches='tight')
    plt.clf()

    plt.plot(times, train_ppl, 'o-')
    plt.plot(times, valid_ppl, 'o-')
    plt.ylabel("ppl")
    plt.xlabel("wall clock time")
    plt.title(model_type + " - PPL Per wall clock")
    plt.legend(labels = ["train: {}".format(smallest_train_ppl), "valid: {}".format(smallest_valid_ppl)])
    plt.savefig(path + model_type + '_ppl_clock.png', bbox_inches='tight')
    plt.clf()

#loading data from RNN
lc_path = "GRU_SGD_4.2/learning_curves.npy"
filepath = "GRU_SGD_4.2/log.txt"
train_loss, valid_loss_RNN, train_ppl, valid_ppl_RNN, times = get_values(lc_path, filepath)
plot_graphs(train_loss, valid_loss_RNN, train_ppl, valid_ppl_RNN, times, "GRU_SGD_4.2/", "GRU")

#loading data from GRU
lc_path = "GRU_4.1/learning_curves.npy"
filepath = "GRU_4.1/log.txt"
train_loss, valid_loss_GRU, train_ppl, valid_ppl_GRU, times = get_values(lc_path, filepath)
plot_graphs(train_loss, valid_loss_GRU, train_ppl, valid_ppl_GRU, times, "GRU_4.1/" ,"GRU")

#loading data from TRANSFORMER
lc_path = "GRU_Adam_4.2/learning_curves.npy"
filepath = "GRU_Adam_4.2/log.txt"
train_loss, valid_loss_TR, train_ppl, valid_ppl_TR, times = get_values(lc_path, filepath)
plot_graphs(train_loss, valid_loss_TR, train_ppl, valid_ppl_TR, times, "GRU_Adam_4.2/","GRU")

#plot all architecture graphs
plt.plot(times, valid_ppl_RNN, 'o-')
plt.plot(times, valid_ppl_GRU, 'o-')
plt.plot(times, valid_ppl_TR, 'o-')
plt.ylabel("ppl")
plt.xlabel("wall clock time")
plt.title("All - Loss Per wall clock")
plt.legend(labels = ["SGD", "SGD Momemtum", "Adam"])
#plt.savefig('plots/RNN_4_2_all_loss_clock.png', bbox_inches='tight')
plt.clf()

plt.plot(valid_ppl_RNN, 'o-')
plt.plot(valid_ppl_GRU, 'o-')
plt.plot(valid_ppl_TR, 'o-')
plt.ylabel("ppl")
plt.xlabel("epoch")
plt.title("All - Loss Per epoch")
plt.legend(labels = ["SGD", "SGD Momemtum", "Adam"])
#plt.savefig('plots/_4_2_all_loss_epoch.png', bbox_inches='tight')
plt.clf()