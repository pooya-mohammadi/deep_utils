def grouped_mean_std_bar_plot(
    x_names,
    mean,
    std,
    xlabel="",
    ylabel="",
    title="",
    legends="",
    width=0.25,
    save_path=None,
    ylim=(0, 1),
):
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.arange(len(x_names))
    w = -len(mean) // 2 * (width / 2) if len(mean) % 2 == 0 else - \
        len(mean) // 2 * width
    plt.figure(figsize=(25, 12))
    # plot data in grouped manner of bar type
    colors = ["green", "orange"]
    for m, s, color, legend in zip(mean, std, colors, legends):
        plt.bar(x - w, m, width, color=color, label=legend)
        plt.errorbar(x - w, m, yerr=s, fmt="o", color="r")
        w += width
    plt.xticks(x, x_names)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(ylim)
    plt.title(title)
    plt.legend(loc="best", fancybox=True, framealpha=0.5)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()
