import numpy as np
import tensorflow as tf
from sklearn.utils import resample
from tensorflow.python.keras.models import Sequential
from Model.DomainAdaptation.domain_adaptation_layer import DGLayer

class DomainCallback(tf.keras.callbacks.Callback):
    """ """
    def __init__(self, train_data, test_data, print_res=True, max_sample_size=1000):
        super(DomainCallback, self).__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.domain_layer = None
        self.history = {}
        self.print_res = print_res
        self.max_sample_size = min(max_sample_size, len(train_data), len(test_data))

    def on_train_begin(self, logs=None):
        self.epoch = []


    def on_epoch_end(self, epoch, logs):
        """ adds values of domain generalization layer to the history during training

        Parameters
        ----------
        epoch : `int`
            current epoch
            
        logs : `dict`
            dictionary with logs during training


        """

        try:
            feature_extractor = self.model.feature_extractor
            for l in range(len(self.model.prediction_layer.layers)):
                if isinstance(self.model.prediction_layer.layers[l], DGLayer):
                    self.domain_layer_index = l
                    self.domain_layer = self.model.prediction_layer.layers[l]

        except:
            for l in range(len(self.model.layers)):
                if isinstance(self.model.layers[l], DGLayer):
                    self.domain_layer_index = l
                    self.domain_layer = self.model.layers[l]

            if self.domain_layer is None:
                print('no DomainAdaptationLayer found')

            else:
                feature_extractor = Sequential(self.model.layers[:self.domain_layer_index])

        if isinstance(self.train_data, tf.data.Dataset):
            random_sample_train = self.train_data.shuffle(len(self.train_data), seed=123).take(self.max_sample_size)
            random_sample_test = self.train_data.shuffle(len(self.test_data), seed=123).take(self.max_sample_size)
        else:
            random_sample_train = resample(self.train_data, replace=False, n_samples=self.max_sample_size)
            random_sample_test = resample(self.test_data, replace=False, n_samples=self.max_sample_size)


            eval_dict = {}
            train_features = feature_extractor(random_sample_train)
            test_features = feature_extractor(random_sample_test)

            # distribution  of the training data
            domain_train_prob = self.domain_layer.get_domain_prob(train_features).numpy()
            eval_dict['DOMAIN_PROB_TRAIN'] = np.mean(domain_train_prob, axis=0).round(3)
            eval_dict['PROB_STD_TRAIN'] = np.std(domain_train_prob, axis=0).round(3)


            # distribution  of the test data
            domain_train_prob = self.domain_layer.get_domain_prob(test_features).numpy()
            eval_dict['DOMAIN_PROB_TEST'] = np.mean(domain_train_prob, axis=0).round(3)
            eval_dict['PROB_STD_TEST'] = np.std(domain_train_prob, axis=0).round(3)

            eval_dict['DOMAIN_VARIANCE'] = np.round(self.domain_layer.get_domain_distributional_variance(), 4)

            eval_dict['MMD_TRAIN'] = np.round(self.domain_layer.get_OLS_penalty(train_features).numpy(), 4)
            eval_dict['MMD_TEST'] = np.round(self.domain_layer.get_OLS_penalty(test_features).numpy(), 4)

            eval_dict.update(self.domain_layer.get_orth_penalty())
            logs.update(eval_dict)

            if self.print_res:
                try:
                    output = "EPOCH:" + str(epoch) + "\t || " + "  || ".join([key + ": " + str(np.round(logs[key], 3)) + "0" * (3 - len(str(np.round(logs[key], 3)).split(".")[1])) for key in logs.keys()]) + "  ||"
                    output = output.replace("]0", "]")
                    print(output)
                except:
                    pass
            self.epoch.append(epoch)

            #logs = logs or {}
            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)

            self.model.history = self


class DomainRegularizationCallback(tf.keras.callbacks.Callback):
    """ """
    def __init__(self, num_epochs, gamma=10):
        super(DomainRegularizationCallback, self).__init__()
        self.num_epochs = num_epochs
        self.gamma = gamma
        self.domain_layer_index = None
        self.domain_layer = None

    def on_epoch_end(self, epoch, logs=None):
        """

        Parameters
        ----------
        epoch :
            
        logs :
            (Default value = None)

        Returns
        -------

        
        """
        if self.domain_layer is None:
            for l in range(len(self.model.layers)):
                if isinstance(self.model.layers[l], DGLayer):
                    self.domain_layer_index = l
                    self.domain_layer = self.model.layers[l]

        p = epoch/self.num_epochs

        lamb_param = 2/(1 + np.exp(-self.gamma*p)) - 1
        self.domain_layer.set_domain_reg_param(lamb_param)


class FreezeFeatureExtractor(tf.keras.callbacks.Callback):
    """ """
    def __init__(self, num_epochs):
        super(FreezeFeatureExtractor, self).__init__()
        self.num_epochs = num_epochs
        self.epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        """

        Parameters
        ----------
        epoch :
            
        logs :
            (Default value = None)

        Returns
        -------

        
        """
        if self.epoch > self.num_epochs:
            self.model.feature_extractor.trainable = False
            self.epoch += 1

