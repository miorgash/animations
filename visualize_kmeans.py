import matplotlib.pyplot as plt

def begin():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    return fig, ax


def viz_samples(ax, df, c=None):
    ax.scatter(x=df['p1'], y=df['p2'], c=c,
               cmap='tab10', s=100, alpha=0.4, marker='o')


def viz_centroids(ax, df, c=None):
    ax.scatter(x=df['p1'], y=df['p2'], c=c,
               cmap='tab10', s=600, alpha=0.8, marker='x')


def save(fig, i):
    fig.savefig('fig/figure_{0:02d}.png'.format(i))
    plt.close()


def generate_seq():
    seq = 0
    while True:
        yield seq
        seq += 1