import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib as pl
from PIL import Image
from numpy import random
sns.set_style('darkgrid')
sns.set_context('poster')

def make_gif(file_path):
    image_path = pl.Path('fig')
    append_images = [Image.open(f) for f in image_path.glob('*.png')]
    im = append_images[0]
    im.save(file_path,
            save_all=True,
            duration=600,
            append_images=append_images[1:])


def delete_png(dir_path):
    p = pl.Path(dir_path)
    print(p)
    for f in p.glob('*.png'):
        print(f)
        f.unlink()



def mkmeans(feature_df, feature_list):

    def init_class(feature_df, n=3):

        sample_df = feature_df.assign(cls=random.randint(0, n, feature_df.shape[0]))

        return sample_df


    def assign_to_class(feature_df, centroid_df):

        cols = feature_df.columns

        product_df = pd.merge(
            feature_df \
                .reset_index() \
                .assign(dummy_key='dummy') \
                .rename(columns={cols[0]: 'p1', cols[1]: 'p2'}),
            centroid_df \
                .assign(dummy_key='dummy') \
                .rename(columns={cols[0]: 'q1', cols[1]: 'q2'}),
            on='dummy_key', how='inner').drop(labels=['dummy_key'], axis=1)

        # calculate euclidean distance
        euc_df = product_df.assign(euc=product_df.apply(
            lambda x: np.sqrt((x['p1'] - x['q1']) ** 2 + (x['p2'] - x['q2']) ** 2), axis=1))

        sample_df = euc_df[euc_df['euc'] == euc_df.groupby(by='index')['euc'] \
            .transform(min)][['p1', 'p2', 'cls']]

        return sample_df


    def calc_centroids(sample_df):

        return sample_df.groupby(by=['cls'])[['p1', 'p2']].mean().reset_index()


    def evaluate(old_centroid_df, centroid_df):

        return (old_centroid_df == centroid_df).all().all()


    def viz(i, sample_df=None, centroid_df=None):

        # figure and axes
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # plot
        if sample_df is not None:
            ax.scatter(x=sample_df['p1'], y=sample_df['p2'], c=sample_df['cls'],
                       cmap='tab10', s=100,
                       alpha=0.4, marker='o')

        if centroid_df is not None:
            ax.scatter(x=centroid_df['p1'], y=centroid_df['p2'], c=centroid_df['cls'],
                       cmap='tab10', s=600,
                       alpha=0.8, marker='x')

        # visualize
        # plt.show()
        fig.savefig('fig/figure_{0:02d}.png'.format(i))
        plt.close()

    def generate_seq():
        seq = 0
        while True:
            yield seq
            seq += 1

    # **
    # main process
    # *

    # rename columns
    feature_df = feature_df.loc[:, feature_list]
    feature_df.rename(
        columns={feature_list[0]: 'p1', feature_list[1]: 'p2'},
        inplace=True)

    # variables and objects
    match = False
    isfirst = True
    seq = generate_seq()

    # initialize centroids; random
    print('initialize classes')
    sample_df = init_class(feature_df)

    # initial centroids
    viz(seq.__next__(), sample_df=sample_df)

    # repeat attempts
    while not match:

        if isfirst:
            print('initialize centroids')
            centroid_df = calc_centroids(sample_df)
            isfirst = False

            # plot --
            viz(seq.__next__(), sample_df=sample_df, centroid_df=centroid_df)
            # -- plot

        else:
            print('calculate centroids again')
            old_centroid_df = centroid_df
            centroid_df = calc_centroids(sample_df)
            match = evaluate(old_centroid_df, centroid_df)

            # plot --
            viz(seq.__next__(), sample_df=sample_df)
            viz(seq.__next__(), sample_df=sample_df, centroid_df=centroid_df)
            # -- plot

        # classify again
        print('classify again')
        sample_df = assign_to_class(feature_df, centroid_df)

        # plot --
        viz(seq.__next__(), sample_df=sample_df, centroid_df=centroid_df)
        # -- plot

    return sample_df