import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
from sklearn import metrics
import pandas as pd
import glob as glob

from SimulationExperiments.digits5.d5_dataloader import load_digits

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.manifold import TSNE

import tqdm

# load data once in the program and keep in class

SOURCE_SAMPLE_SIZE = 5000
TARGET_SAMPLE_SIZE = 9000
img_shape = (32, 32, 3)

class DigitsData(object):
    def __init__(self, test_size=SOURCE_SAMPLE_SIZE):
        self.x_train_dict, self.y_train_dict, self.x_test_dict, self.y_test_dict = load_digits(test_size=test_size)


if __name__ == '__main__':
    data = DigitsData()
    print("Works")
    SOURCE_DOMAINS = ['mnist', 'svhn', 'syn', 'usps']
    x_source_tr = np.concatenate([data.x_train_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)
    feature_extractors = "/local/home/sfoell/GitHub/gdu-old/SimulationExperiments/digits5/2022-01-20"
    feature_extractors = np.unique(sorted(glob.glob(feature_extractors + '/**/mnistm/**/feature_extractor.h5.tmp', recursive=True)))
    for i in feature_extractors:
        feature_extractor = tf.keras.models.load_model(i)
        x = feature_extractor.predict(x_source_tr).squeeze()

        tsne = TSNE(n_components = 2, random_state=1)
        tsne_results = tsne.fit_transform(x)

        cluster_scores = pd.DataFrame(columns = ["calinski_harabasz_score", "davies_bouldin_score", "silhouette_score"], index = list(range(2, 26)))
        K = range(2, 26)

        for num_clusters in tqdm.tqdm(K):
            kmeans = KMeans(n_clusters=num_clusters, random_state=1).fit(x)
            tsne_kmeans = KMeans(n_clusters=num_clusters, random_state=1).fit(tsne_results)

            cluster_scores.at[num_clusters, "calinski_harabasz_score"] = np.round(calinski_harabasz_score(x, kmeans.labels_), 4)
            cluster_scores.at[num_clusters, "davies_bouldin_score"] = np.round(davies_bouldin_score(x, kmeans.labels_), 4)
            cluster_scores.at[num_clusters, "silhouette_score"] = np.round(silhouette_score(x, kmeans.labels_), 4)

            cluster_scores.at[num_clusters, "calinski_harabasz_score_tsne"] = np.round(calinski_harabasz_score(tsne_results, tsne_kmeans.labels_), 4)
            cluster_scores.at[num_clusters, "davies_bouldin_score_tsne"] = np.round(davies_bouldin_score(tsne_results, tsne_kmeans.labels_), 4)
            cluster_scores.at[num_clusters, "silhouette_score_tsne"] = np.round(silhouette_score(tsne_results, tsne_kmeans.labels_), 4)
            cluster_scores.to_csv(f'/local/home/sfoell/GitHub/gdu-old/SimulationExperiments/digits5/cluster-scores{i.split("/")[-2].split("_")[-1]}.csv')
            print(cluster_scores)

            #cluster_scores.at[num_clusters, "calinski_harabasz_score_tsne"] = np.round(calinski_harabasz_score(tsne_results, kmeans.labels_), 4)
            #cluster_scores.at[num_clusters, "davies_bouldin_score"] = np.round(davies_bouldin_score(tsne_results, kmeans.labels_), 4)
            #cluster_scores.at[num_clusters, "silhouette_score"] = np.round(silhouette_score(tsne_results, kmeans.labels_), 4)
