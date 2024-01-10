import torch
import matplotlib.pyplot as plt
import numpy as np
plt.switch_backend('agg')


def optimizer_step(loss, optimizer, retain_g=False):
    optimizer.zero_grad()
    loss.backward(retain_graph=retain_g)
    optimizer.step()


def create_scatter(z, save_path, num_class=4):
    #fig = plt.figure(figsize=(4, 4), facecolor='w')
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])

    m = z.shape[0] // num_class
    #colors=["#0000FF", "#00FF00", "#FF0066", "#123456", "666688"]
    colors = ["#FFB6C1", "#D2691E", "#C71585", "#0000FF", "#00FF00", "#FF0066", "#8B008B", "#BA55D3", "#4B0082", "#F8F8FF", "#00008B", "#4169E1", "#1E90FF", "#87CEEB", "#7FFFAA", "#808000"]
    for i in range(num_class):
        z_i = z[(i * m):((i + 1) * m), :]
        c =  [((i * 1.0 / num_class ) + 0.05, (i*1.0/num_class)+0.05, (i*1.0/num_class)+0.05)] * m 
        c = np.vstack(c)
        #c = [i] * m 
        print(c)
        #plt.scatter(z_i[:, 0], z_i[:, 1], edgecolor='none', alpha=0.5, cmap="Dark2")
        #plt.scatter(z_i[:, 0], z_i[:, 1], c=c, cmap="Dark2", edgecolor='none')
        #plt.scatter(z_i[:, 0], z_i[:, 1], edgecolor='none', cmap="Dark2")
        #plt.scatter(z_i[:, 0], z_i[:, 1], edgecolor='none', c=c, cmap="Dark2")
        plt.scatter(z_i[:, 0], z_i[:, 1], edgecolor='none', c=colors[i])
        #if i > 3:
        #    break

    ax.axis('off')
    fig.savefig(save_path)
