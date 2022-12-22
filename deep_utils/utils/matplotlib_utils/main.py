MATPLOTLIB_COLORS = ["green", "orange", "blue", "black"]


def show_plt(img, is_rgb, figsize=None):
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    if not is_rgb:
        img = img[..., ::-1]
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()


def matplotlib_grid(images, n_columns, n_rows):
    assert n_rows * n_columns > len(images), "number of images should be less than n_rows * n_columns"
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(n_columns * n_rows)
    for ax, img in zip(axes, images):
        ax.imshow(img)
    plt.show()
