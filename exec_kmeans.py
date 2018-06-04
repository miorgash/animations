import pandas as pd
from sklearn import datasets
from kmeans import make_gif, delete_png, mkmeans

if __name__ == '__main__':
    # load data
    wine = datasets.load_wine()
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)

    # execute
    sample_df = mkmeans(wine_df, ['alcohol', 'flavanoids'])

    # make GIF
    make_gif('fig/kmeans.gif')

    # delete png
    delete_png('fig')