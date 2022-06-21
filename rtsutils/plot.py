import base64
import itertools
from io import BytesIO

import PIL.ImageDraw as ImageDraw
import numpy as np
import seaborn as sns
from IPython.display import display
from PIL import Image
from matplotlib import pyplot as plt
from numpy.random import RandomState


def display_close(fig):
    display(fig)
    plt.close(fig)


def display_text(text, fontsize=18):
    fig = plt.figure(figsize=(12, 0.1))
    plt.title(text, fontsize=fontsize)
    plt.axis("off")
    display(fig)
    plt.close(fig)


def create_bw_image(image_cells):
    rows = image_cells.shape[0]
    cols = image_cells.shape[1]
    blank_image = Image.new(
        "L", (image_cells.shape[3] * cols, image_cells.shape[2] * rows)
    )
    for i_row, i_col in itertools.product(range(rows), range(cols)):
        x = image_cells[i_row, i_col]
        x = np.clip(255 - np.round(x * 255), 0, 255).astype(np.uint8)
        blank_image.paste(
            Image.fromarray(x),
            (i_col * image_cells.shape[3], i_row * image_cells.shape[2]),
        )
    return blank_image


def create_rgb_image(image_cells):
    rows = image_cells.shape[0]
    cols = image_cells.shape[1]
    blank_image = Image.new(
        "RGB", (image_cells.shape[4] * cols, image_cells.shape[3] * rows)
    )
    for i_row, i_col in itertools.product(range(rows), range(cols)):
        x = image_cells[i_row, i_col]
        x = np.clip(np.round(x * 255), 0, 255).astype(np.uint8).transpose(1, 2, 0)
        blank_image.paste(
            Image.fromarray(x),
            (i_col * image_cells.shape[4], i_row * image_cells.shape[3]),
        )
    return blank_image


def create_image_with_label(X, y):
    im = create_bw_image(X)
    im = im.convert("RGB")
    draw = ImageDraw.Draw(im)
    for i_row in range(len(y)):
        for i_col in range(len(y[i_row])):
            draw.text(
                (i_col * 28, i_row * 28),
                str(y[i_row, i_col]),
                (255, 50, 255),
            )
    return im


def rescale(im, scale_factor, resample=Image.BICUBIC):
    return im.resize(
        (int(round(im.width * scale_factor)), int(round(im.height * scale_factor))),
        resample=resample,
    )


def stack_images_in_rows(*batch_images, n_cols):
    padded_batch_images = []
    for b in batch_images:
        # extra cols mean columns on new uncompleted row
        n_extra_cols = len(b) % n_cols
        if n_extra_cols != 0:
            # fill remaining images
            # this would fail if b is too small even for first row
            # but seems unlikely
            b = np.concatenate((b, np.zeros_like(b[:n_cols-n_extra_cols]) + 0.5))
        assert (len(b) % n_cols) == 0
        padded_batch_images.append(b)
    n_rows_per_batch = padded_batch_images[0].shape[0] // n_cols
    reshaped_batches = [
        b.reshape(n_rows_per_batch, n_cols, *b.shape[1:])
        for b in padded_batch_images
    ]
    n_rows = n_rows_per_batch * len(padded_batch_images)

    return np.stack(reshaped_batches, axis=1).reshape(
        n_rows, n_cols, *padded_batch_images[0].shape[1:]
    )


def plot_dist_comparison(vals_a, vals_b):
    rng = RandomState(20171901)
    plt.plot(
        vals_a, rng.randn(len(vals_a)) * 0.05, marker="o", alpha=0.9, linestyle="None"
    )
    plt.plot(
        vals_b, rng.randn(len(vals_b)) * 0.05, marker="o", alpha=0.9, linestyle="None"
    )
    oldxlim = plt.xlim()
    plt.figure()
    sns.distplot(vals_a)
    sns.distplot(vals_b)
    plt.xlim(oldxlim)


# https://www.kaggle.com/stassl/displaying-inline-images-in-pandas-dataframe
def image_base64(im):
    with BytesIO() as buffer:
        im.save(buffer, "jpeg")
        return base64.b64encode(buffer.getvalue()).decode()


def image_formatter(im):
    return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'
