import numpy as np
import pandas as pd
import seaborn as sns
from numpy import random
from visualize_kmeans import viz, generate_seq

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


    # **
    # main process
    # *

    # figures' setting
    sns.set_style('darkgrid')
    sns.set_context('paper')

    # rename columns
    feature_df = feature_df.loc[:, feature_list]
    feature_df.rename(
        columns={feature_list[0]: 'p1', feature_list[1]: 'p2'},
        inplace=True)

    # variables and objects
    match = False
    isfirst = True
    seq = generate_seq()

    # initialize class  (random)
    sample_df = init_class(feature_df)

    # repeat attempts
    while not match:

        # old class
        viz(seq.__next__(), sample_df=sample_df)

        if isfirst:
            # calculate centroids
            centroid_df = calc_centroids(sample_df)
            isfirst = False

        else:
            # calculate centroids
            old_centroid_df = centroid_df
            centroid_df = calc_centroids(sample_df)

            # evaluate the attempt
            match = evaluate(old_centroid_df, centroid_df)

        # new centroid / old class
        viz(seq.__next__(), sample_df=sample_df, centroid_df=centroid_df)

        # classify
        sample_df = assign_to_class(feature_df, centroid_df)
        viz(seq.__next__(), sample_df=sample_df, centroid_df=centroid_df)  # new centroid / new class

    return sample_df