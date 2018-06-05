import matplotlib.pyplot as plt

class FigMaker:

    global seq

    def __init__(self, samples=None, sample_color=None, centroids=None, centroid_color=None):
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.samples = samples
        self.sample_color = sample_color
        self.centroids = centroids
        self.centroid_color = centroid_color

        global seq
        seq = self.generate_seq()


    def generate_seq(self):
        seq = 0
        while True:
            yield seq
            seq += 1


    def viz_samples(self):
        self.ax.scatter(x=self.samples.iloc[:, 0], y=self.samples.iloc[:, 1], c=self.sample_color,
                   cmap='tab10', s=100, alpha=0.4, marker='o')


    def viz_centroids(self, c=None):
        self.ax.scatter(x=self.centroids.iloc[:, 0], y=self.centroids.iloc[:, 1], c=self.centroid_color,
                   cmap='tab10', s=600, alpha=0.8, marker='x')


    def save(self, i):
        self.fig.savefig('fig/figure_{0:02d}.png'.format(i))
        plt.close()


    def make_samplefig(self):
        global seq
        self.viz_samples()
        self.save(seq.__next__())


    def make_centroidfig(self):
        global seq
        self.viz_centroids()
        self.save(seq.__next__())

    def make_fig(self):
        global seq
        self.viz_samples()
        self.viz_centroids()
        self.save(seq.__next__())


