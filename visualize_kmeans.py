import matplotlib.pyplot as plt

def viz(i, sample_df=None, centroid_df=None):
    # figure and axes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # plot
    if sample_df is not None:
        ax.scatter(x=sample_df['p1'], y=sample_df['p2'], c=sample_df['cls'],
                   cmap='tab10', s=100, alpha=0.4, marker='o')

    if centroid_df is not None:
        ax.scatter(x=centroid_df['p1'], y=centroid_df['p2'], c=centroid_df['cls'],
                   cmap='tab10', s=600, alpha=0.8, marker='x')

    # visualize
    # plt.show()
    fig.savefig('fig/figure_{0:02d}.png'.format(i))
    plt.close()


def generate_seq():
    seq = 0
    while True:
        yield seq
        seq += 1