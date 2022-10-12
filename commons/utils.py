import torch
import numpy as np
from matplotlib import pyplot as plt

is_train_on_gpu = torch.cuda.is_available()


def view_samples(samples, legend_lbl, img_size=128):
    fig, axes = plt.subplots(figsize=(25, 15), nrows=2, ncols=3, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        img = img.detach().cpu().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = ((img + 1)*255 / (2)).astype(np.uint8)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((img_size, img_size, 1)), cmap='gray')
        ax.text(4, 10, legend_lbl, fontsize=20, bbox={'facecolor': 'white', 'pad': 10})
