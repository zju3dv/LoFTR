import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# --- VISUALIZATION ---

def make_matching_figure(img0, img1, mkpts0, mkpts1, color, text=[], path=None):
    # draw image pair
    fig, axes = plt.subplots(1, 2, figsize=(10, 6), dpi=75)
    axes[0].imshow(img0, cmap='gray')
    axes[1].imshow(img1, cmap='gray')
    for i in range(2):   # clear all frames
        axes[i].get_yaxis().set_ticks([])
        axes[i].get_xaxis().set_ticks([])
        for spine in axes[i].spines.values():
            spine.set_visible(False)
    plt.tight_layout(pad=1)

    # draw matches
    fig.canvas.draw()
    transFigure = fig.transFigure.inverted()
    fkpts0 = transFigure.transform(axes[0].transData.transform(mkpts0))
    fkpts1 = transFigure.transform(axes[1].transData.transform(mkpts1))
    fig.lines = [matplotlib.lines.Line2D((fkpts0[i, 0], fkpts1[i, 0]), (fkpts0[i, 1], fkpts1[i, 1]),
                                         transform=fig.transFigure, c=color[i], linewidth=1) for i in range(len(mkpts0))]

    axes[0].scatter(mkpts0[:, 0], mkpts0[:, 1], c=color, s=4)
    axes[1].scatter(mkpts1[:, 0], mkpts1[:, 1], c=color, s=4)

    # put txts
    txt_color = 'k' if img0[:100, :200].mean() > 200 else 'w'
    fig.text(
        0.01, 0.99, '\n'.join(text), transform=fig.axes[0].transAxes,
        fontsize=15, va='top', ha='left', color=txt_color)

    # save or return figure
    if path:
        plt.savefig(str(path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        return fig


def error_colormap(err, thr, alpha=1.0):
    assert alpha <= 1.0 and alpha > 0, f"Invaid alpha value: {alpha}"
    x = 1 - np.clip(err / (thr * 2), 0, 1)
    return np.clip(
        np.stack([2-x*2, x*2, np.zeros_like(x), np.ones_like(x)*alpha], -1), 0, 1)
