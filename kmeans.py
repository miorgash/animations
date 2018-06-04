import numpy as np
import pandas as pd
import seaborn as sns
from numpy import random
from visualize_kmeans import generate_seq, begin, viz_samples, viz_centroids, save

def mkmeans(feature_df, feature_list):

    def init_class(feature_df, n=3):

        return list(random.randint(0, n, feature_df.shape[0]))


    def classify(feature_df, centroid_df):

        product_df = pd.merge(
            feature_df \
                .reset_index() \
                .assign(dummy_key='dummy'),
            centroid_df \
                .reset_index() \
                .rename(columns={'index': 'cls'}) \
                .assign(dummy_key='dummy'),
            on='dummy_key', how='inner').drop(labels=['dummy_key'], axis=1)

        # calculate euclidean distance
        euc_df = product_df.assign(euc=product_df.apply(
            lambda x: np.sqrt((x.iloc[1] - x.iloc[4]) ** 2 + (x.iloc[2] - x.iloc[5]) ** 2), axis=1))

        classes = list(euc_df[euc_df['euc'] == euc_df.groupby(by='index')['euc'] \
            .transform(min)].iloc[:, 3])

        return classes


    def calc_centroids(feature_df, classes):

        return feature_df.groupby(by=classes).mean()


    def evaluate(old_centroid_df, centroid_df):

        return (old_centroid_df == centroid_df).all().all()


    # variables and objects
    match = False
    isfirst = True
    seq = generate_seq()

    # figures' setting
    sns.set_style('darkgrid')
    sns.set_context('paper')

    # **
    # main process
    # *

    # rename columns
    feature_df = feature_df.loc[:, feature_list]

    # initialize class (random)
    classes = init_class(feature_df)

    # repeat attempts
    while not match:

        # old class
        fig, ax = begin()
        viz_samples(ax, feature_df, classes)
        save(fig, seq.__next__())

        if isfirst:
            # calculate centroids
            centroid_df = calc_centroids(feature_df, classes)
            isfirst = False

        else:
            # calculate centroids
            old_centroid_df = centroid_df
            centroid_df = calc_centroids(feature_df, classes)

            # evaluate the attempt
            match = evaluate(old_centroid_df, centroid_df)

        # new centroid / old class
        fig, ax = begin()
        viz_samples(ax, feature_df, classes)
        viz_centroids(ax, centroid_df, centroid_df.index)
        save(fig, seq.__next__())

        # classify
        classes = classify(feature_df, centroid_df)

        # new centroid / new class
        fig, ax = begin()
        viz_samples(ax, feature_df, classes)
        viz_centroids(ax, centroid_df, centroid_df.index)
        save(fig, seq.__next__())

    return feature_df.assign(cls=classes)