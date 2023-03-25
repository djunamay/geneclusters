import matplotlib.pyplot as plt
import numpy as np

def plot_permutations(permuted_t, observed_t, p, Nperm, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot = plt.hist(np.array(permuted_t).reshape(-1), bins = int(.2*Nperm), histtype='step')
    plt.axvline(observed_t, color='red')
    ax.text(observed_t[0]+(observed_t[0]*0.1), np.max(plot[0])-(0.1*np.max(plot[0])), 'p-value='+str(p), style='italic')
    ax.set_title(title, fontsize = 20)