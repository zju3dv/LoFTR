# %% IMPORTS
import matplotlib.pyplot as plt
import numpy as np


def draw_images(images: list, save_path: str =None):
    """
    Draw images
    """
    fig, ax = plt.subplots( len(images) - len(images)//2,len(images)//2, figsize=(10, 10))
    ax = ax.flatten()
    for i in range(len(images)):
        ax[i].imshow(images[i], cmap='gray')
        ax[i].set_xticks([])
        ax[i].set_yticks([])
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()


def draw_epipolar_line_on_image(img, x_shift, line, color, ax):
    """
    Draw epipolar line on image
    """
    # draw epipolar lines on destination image
    h1, w1 = img.shape
    a, b, c = line
    # check if point is out of y axes range
    p1_y = -c / b
    if p1_y > h1 - 1:
        p1 = np.array([x_shift + (-(b * (h1 - 1) + c) / a), h1 - 1])
    elif p1_y < 0:
        p1 = np.array([x_shift + (-(b * 0 + c) / a), 0])
    else:
        p1 = np.array([x_shift, p1_y])
    # check if point is out of x axes range
    p2_y = -(a * (w1 - 1) + c) / b
    if p2_y < 0:
        p2 = np.array([x_shift - (c / a), 0])
    elif p2_y > h1 - 1:
        p2 = np.array([x_shift - ((b * (h1 - 1) + c) / a), h1 - 1])
    else:
        p2 = np.array([x_shift + w1 - 1, p2_y])

    # ax.axline(p1, slope=-a/b, color=p[0].get_color(), linewidth=0.5)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '--', linewidth=0.4, color=color)

def draw_matches_on_images(img0, img1, kpts0, kpts1, conf, title:str = "LoFTR", draw_epipolar_lines: bool = False,
                           f_matrix: np.ndarray = None, show_confidences_hist: bool = False, conf_factor: float = 0.0, save_path: str = None):
    """
    Draw matches on images
    """

    plt.rcParams['figure.autolayout'] = True

    x_shift = img0.shape[1]
    if show_confidences_hist: # plot histogram of confidences
        fig = plt.figure()
        plt.hist(conf, bins=10)
        plt.title('Confidence Histogram')
        plt.xlabel('Confidence')
        plt.ylabel('Count')
        plt.xlim(0, 1)
        # plt.ylim(0, 1000)
        plt.grid(True)
        plt.show()

    kpts0 = kpts0[conf > conf_factor]
    kpts1 = kpts1[conf > conf_factor]
    # kpts1[:, 0] += x_shift
    image = np.concatenate([img0, img1], axis=1)
    # fit the fig plot to screen size


    fig, ax = plt.subplots(1, 1)
    ax.imshow(image, cmap='gray')
    ax.set_title(title)
    for i in range(len(kpts0)):
        kpts1_shift = np.array([kpts1[i, 0] + x_shift, kpts1[i, 1]])
        p = ax.plot(*np.vstack([kpts0[i], kpts1_shift]).T, 'o-', markersize=4, fillstyle='none', linewidth=0.9)
        # p = ax.plot(*np.vstack([kpts0[i], kpts1_shift]).T, 'o', markersize=4, fillstyle='none', linewidth=0.9)
        if draw_epipolar_lines:
            # draw epipolar lines on destination image
            line_1 = f_matrix @ np.hstack([kpts0[i], 1])
            draw_epipolar_line_on_image(img1, x_shift, line_1, p[0].get_color(), ax)

            # draw epipolar lines on source image
            line_0 = f_matrix.T@np.hstack([kpts1[i], 1])
            draw_epipolar_line_on_image(img0, 0, line_0, p[0].get_color(), ax)


    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()