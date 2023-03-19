import matplotlib.pyplot as plt
import numpy as np

with plt.xkcd():
    fig, ax = plt.subplots()


    rng = np.random.default_rng(1234)

    progress = np.arange(0, 100) * 0.01
    progress = np.abs(rng.normal(0, 0.1, 100) + progress)

    ax.plot(progress, color='k')
    ax.annotate(
        "WE'RE HERE",
        xy=(100, 1),
        xytext=(50, 5),
        arrowprops=dict(arrowstyle='->')
    )

    x_forecast = np.arange(100, 200)
    y_forecast = x_forecast * 0.01

    ax.plot(x_forecast, y_forecast, linestyle='--', color='k')

    ax.spines.right.set_color('none')
    ax.spines.top.set_color('none')
    xticks = [0, 100, 200, 300]
    xlabels = [f'week {int(w/100)}' for w in xticks]
    ax.set_xticks(xticks, xlabels)
    ax.set_yticks([])
    ax.set_ylim([0, 10])

    ax.set_xlabel('time')
    ax.set_ylabel('progress')
    plt.tight_layout()
    plt.savefig('progress.pdf', dpi=300)
