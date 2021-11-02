import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf

from SimulationExperiments.experiment_4_digits.digits_5_classification import DigitsData
from SimulationExperiments.experiment_4_digits.digits_utils import get_lenet_feature_extractor

if __name__ == '__main__':
    data = DigitsData()
    SOURCE_DOMAINS = ['mnist', 'mnistm', 'svhn', 'syn', 'usps']
    x_source_tr = np.concatenate([data.x_train_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)
    y_source_tr = np.concatenate([data.y_train_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)

    # tf.data.Dataset.from_tensor_slices((x_source_tr, y_source_tr)).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    x_source_te = np.concatenate([data.x_test_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)
    y_source_te = np.concatenate([data.y_test_dict[source.lower()] for source in SOURCE_DOMAINS], axis=0)

    x = np.concatenate([x_source_te, x_source_tr], axis=0)
    feature_extractor = tf.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3)) # get_lenet_feature_extractor()
    x = feature_extractor.predict(x)
    x = x.squeeze()
    Sum_of_squared_distances = []
    K = range(1, 10)
    for num_clusters in K:
        kmeans = KMeans(n_clusters=num_clusters)
        kmeans.fit(x)
        Sum_of_squared_distances.append(kmeans.inertia_)
    plt.plot(K, Sum_of_squared_distances, 'bx-')
    plt.xlabel('Values of K')
    plt.ylabel('Sum  of  squared   distances / Inertia')
    plt.title('Elbow Method For Optimal k')
    plt.show()
    plt.savefig('K-Means.png')

