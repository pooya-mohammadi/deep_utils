def show_plt(img, is_rgb, figsize=None):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    if not is_rgb:
        img = img[..., ::-1]
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
