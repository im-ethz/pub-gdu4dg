import numpy as np
from sklearn.cluster import KMeans
import tensorflow as tf
from sklearn import metrics

from SimulationExperiments.experiment_4_digits.digits_5_classification import DigitsData

if __name__ == '__main__':
    data = DigitsData()
    SOURCE_DOMAINS = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    x_source_tr = np.concatenate([data.x_train_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)
    y_source_tr = np.concatenate([data.y_train_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)
    # TODO tf.data.Dataset.from_tensor_slices((x_source_tr, y_source_tr)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    x_source_te = np.concatenate([data.x_test_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)
    y_source_te = np.concatenate([data.y_test_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)

    x = np.concatenate([x_source_te, x_source_tr], axis=0)
    # feature_extractor = get_lenet_feature_extractor()
    feature_extractor = tf.keras.models.load_model('file_path')
    x = feature_extractor.predict(x)
    print('Feature extraction finished')
    x = x.squeeze()
    Sum_of_squared_distances = []
    K = range(2, 10)
    calinski_harabasz = []
    davies_bouldin = []
    silhouette_scores = []
    for num_clusters in K:
        kmeans_model = KMeans(n_clusters=num_clusters, random_state=1).fit(x)
        labels = kmeans_model.labels_
        calinski_harabasz_score = metrics.calinski_harabasz_score(x, labels)
        calinski_harabasz.append(calinski_harabasz_score)
        davies_bouldin_score = metrics.davies_bouldin_score(x, labels)
        davies_bouldin.append(davies_bouldin_score)
        silhouette_score = metrics.silhouette_score(x, labels, metric='euclidean')
        silhouette_scores.append(silhouette_score)
        print(num_clusters, calinski_harabasz_score, davies_bouldin_score, silhouette_score)

    print(calinski_harabasz, davies_bouldin, silhouette_scores)

