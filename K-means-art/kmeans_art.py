import os
import subprocess
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm


def load_image(image_path, reshape=True):
    """
    load image and reshape to number-of-pixels X 3 matrix
    """
    img = plt.imread(image_path)
    img_shape = img.shape
    if reshape:
        img = img.reshape(-1, img_shape[-1])

    return img, img_shape


def image_kmeans(img, img_shape, n_clusters=8, show_image=False):
    """
    single image k-means
    """
    kmeans = KMeans(n_clusters=n_clusters, algorithm='full')
    X = kmeans.fit_predict(img)
    #  round clusters to the nearest integer (RGB compatible)
    clusters = np.rint(kmeans.cluster_centers_)
    img = np.array([clusters[pixel] for pixel in X], dtype=int)
    img = img.reshape(img_shape)

    if show_image:
        plt.imshow(img)

    return img


def fibonacci():
    """
    fibonacci sequence generator
    """
    a = 1
    b = 2
    while True:
        yield a
        a, b = b, a + b


def image_kmeans_mossaic(image_path,
                         width=4, height=3,
                         progression="arithmetic",
                         step=1,
                         save_image_path=None,
                         show_image=True
                         ):
    """
    Image k-means for different values of K.
    """
    # check types and create the iterable to loop over
    N_clusters = height * width

    img, img_shape = load_image(image_path, reshape=True)
    aspect_ratio = (img_shape[0] * height) / (img_shape[1] * width)
    figsize = (50, 50 * aspect_ratio)
    fig, axs = plt.subplots(height, width, figsize=figsize)

    for i in tqdm(range(N_clusters)):

        #  progression
        if progression == "arithmetic":
            n_clusters = i * step + 1
        elif progression == "geometric":
            n_clusters = step ** i
        elif progression == "fibonacci":
            if i == 0:
                fib = fibonacci()
            n_clusters = next(fib)
        else:
            raise ValueError("Progression type is not supported")

        #  order from left to right
        pos_x = int(i / width)
        pos_y = np.mod(i, width)

        # return the image and add to the subplot
        temp_img = image_kmeans(img, img_shape, n_clusters=n_clusters)

        #  deal with 1s in the grid dimensions
        if height != 1 and width != 1:
            axs[pos_x, pos_y].imshow(temp_img, aspect="auto")
            axs[pos_x, pos_y].axis("off")
        elif height == 1 and width != 1:
            axs[pos_y].imshow(temp_img, aspect="auto")
            axs[pos_y].axis("off")
        elif height != 1 and width == 1:
            axs[pos_x].imshow(temp_img, aspect="auto")
            axs[pos_x].axis("off")
        else:
            axs.imshow(temp_img)
            axs.axis("off")

    plt.subplots_adjust(wspace=0.01 * aspect_ratio, hspace=0.01)

    if show_image:
        plt.show()

    if save_image_path is not None:
        fig.savefig(save_image_path, bbox_inches='tight', format='png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a k-means mosaic from a jpg picture')
    parser.add_argument('--input_image_path', type=str, help='path to input .jpg picture')
    parser.add_argument('--output_image_path', type=str, help='path to output .png picture')
    parser.add_argument('--width', type=int, help='width of mosaic', nargs='?', default=4)
    parser.add_argument('--height', type=int, help='height of mosaic', nargs='?', default=3)
    parser.add_argument('--step', type=int, help='step change for each image', nargs='?', default=1)
    parser.add_argument('--progression', type=str, help='progression type', nargs='?', default='fibonacci')
    args = parser.parse_args()

    if not os.path.isfile(args.input_image_path):
        raise ValueError(f"{args.input_image_path} is not a valid file path")

    image_kmeans_mossaic(args.input_image_path,
                         width=args.width, height=args.height,
                         progression=args.progression,
                         step=args.step,
                         save_image_path=args.output_image_path,
                         show_image=False
                         )

    subprocess.run(['open', args.output_image_path], check=True)
