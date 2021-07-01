def show_plt(img):
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
