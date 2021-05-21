import warnings
warnings.filterwarnings('ignore')

import numpy as np
import tensorflow as tf

from tensorflow.python.ops.math_ops import reduce_sum, reduce_prod, reduce_mean, scalar_mul, add, tanh, multiply, sqrt, mat_mul
from tensorflow.python.ops.nn_ops import softmax
from tensorflow.python.ops.array_ops import stack, transpose
from tensorflow.python.ops.linalg.linalg import diag_part


class DomainRegularizer(tf.keras.regularizers.Regularizer):
    """
       Regularization of the domains that are included in the `DomainAdaptationLayer`. The regularization cosists of
       two penalty term, namely the orthogonal penalty and the MMD penalty.
       The orthogonal penalty can bei chosen as `SRIP` or `IPS`. The SRIP is implemented as described in https://arxiv.org/abs/1810.09102.
       The SRIP leads generally in, addtion to orthogonalty, to unit singular values (i.e. well conditioned).

        the second penalty is the MMD, which gives the distance, between the projection of the batch data from the
        RKHS and the actual value. A small MMD penalty tells that the data can be good represented by the domains.


        Note that the regularization will computed for each domain seperately, i.e. the regularization computed here,
        does not represent the full regularization, but the proportion of the full regularization which depends on one domain.

       Parameters
       -----------
       param: double
           factor
       domains: list
           list of the weights of all domains of the layers,
       kernel : tfp.math.psd_kernels instance
           kernel which is used in the DomainAdaptationLayer
       num_domains : int, optional
           number of domains, included in the domains
       orth_pen_method : string, optional
           method which is used as the penalty for the orthogonalization
       domain_number : int
            number of the domain, which will be regularized by this regularization
       mmd_penalty: boolean, optional
           if True, the regularization will include the MMD penalty

       batch_sample : tf.Tensor, optional
           batch_sample is a placeholder and will is updated in each step in the training
       Returns
       -------
        double

       Notes
       ------
       - will be done later


       """

    def __init__(self,
                 kernel=None,
                 domains=None,
                 domain_number=None,
                 similarity_measure=None,
                 orth_pen_method='SRIP',
                 lambda_OLS=1e-2,
                 lambda_sparse=0.05,
                 lambda_orth=0.2
                 ):


        self.domains = domains
        self.kernel = kernel
        self.num_domains = None
        self.domain_dimension = None
        self.domain_number = domain_number

        self.similarity_measure = similarity_measure
        # use orthogonal penalty only in case of projection
        self.orth_pen_method = orth_pen_method if similarity_measure.lower() != "projection" else "NONE"

        self.lambda_OLS = lambda_OLS
        self.lambda_sparse = lambda_sparse
        self.lambda_orth = lambda_orth

        # PLACEHOLDER (will be updated throughout the training)
        self.alpha = None
        self.batch_sample = None

    # for debugging purpose...
    #h = tf.random.normal(shape=(self.domain_basis['domain_0'].numpy().shape[0], input_shape[-1]))
    # h = tf.random.normal(shape=(42, self.domains[0].shape[1]))
    # self.batch_data = h
    #h = tf.Tensor(None, dtype=np.float32)
    #x_domain = tf.random.normal(shape=(2000, 45))*1

    def __call__(self, weight_matrix):
        if self.num_domains is None:
            self.num_domains = len(self.domains)

        if self.domain_dimension is None:
            self.domain_dimension = self.domains[0].shape[0]

        if self.orth_pen_method.lower() in ['srip', "so", "mc"]:
                domains = self.domains + [weight_matrix]
                self.gram_matrix = gram_matrix = tf.map_fn(fn=lambda d_j: tf.map_fn(fn=lambda d_k: reduce_mean(self.kernel.matrix(d_j, d_k)), elems=stack(domains)), elems=stack(domains))
                self.gram_diag = gram_diag = tf.linalg.diag(diag_part(gram_matrix))


        if self.orth_pen_method.lower() == 'srip':
            domain_orthogonality_penalty = (tf.linalg.svd(gram_matrix - gram_diag, compute_uv=False)[0])
            self.orth_pen_name = "SRIP"

        elif self.orth_pen_method.lower() == "so":
            domain_orthogonality_penalty = tf.norm(gram_matrix - gram_diag, ord='fro', axis=(0, 1))
            self.orth_pen_name = "soft_orthogonality"

        elif self.orth_pen_method.lower() == "mc":
            domain_orthogonality_penalty = tf.norm(gram_matrix - gram_diag, ord=np.inf, axis=(0, 1))
            self.orth_pen_name = "mutual_coherence"

        elif self.orth_pen_method.lower() == 'icp':
            domain_orthogonality_penalty = reduce_sum([reduce_mean(self.kernel.matrix(domain, weight_matrix)) for domain in self.domains])
            self.orth_pen_name = "cross_domain_IPS"

        else:
            domain_orthogonality_penalty = float(0.0)
            self.orth_pen_name = "NONE"

        self.orthogonal_penalty = float(1 / (self.num_domains + 1)) * (domain_orthogonality_penalty)

        if self.lambda_sparse > 0:
            self.sparse_penalty = tf.norm(self.alpha_coefficients, ord=1)
        else:
            self.sparse_penalty = float(0)

        self.OLS_penalty = self.get_OLS_penalty(weight_matrix)


        # final output
        self.domain_penalty =  self.lambda_OLS * self.OLS_penalty + self.lambda_sparse * self.sparse_penalty + self.lambda_orth * self.orthogonal_penalty

        return self.domain_penalty

    def get_config(self):
        return {self.orth_pen_name: self.orthogonal_penalty,
                'MMD': self.domain_mmd
                }


    @tf.function
    def compute_alpha(self, h, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme = diag_part(self.get_kme_gram(domains=domains))

        alpha = tf.squeeze(tf.vectorized_map(lambda d: reduce_mean(self.kernel.matrix(h, d[0])/d[1], axis=-1, keepdims=True),
                                     elems=[stack(domains), squared_kme]), axis=-1)

        return tf.transpose(alpha)


    @tf.function
    def get_kme_gram(self, domains=None):
        kme_gram_matrix = tf.map_fn(fn=lambda d_i: tf.map_fn(fn=lambda d_j: reduce_mean(self.kernel.matrix(d_i, d_j)), elems=stack(domains)), elems=stack(domains), parallel_iterations=10)

        return kme_gram_matrix

    @tf.function
    def get_kme_squared_norm(self, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme_norm = tf.map_fn(fn=lambda d_j: reduce_mean(self.kernel.matrix(d_j, d_j)), elems=stack(domains))

        return squared_kme_norm

    #@tf.function
    def get_OLS_penalty(self, weight_matrix):

        domains = self.domains + [weight_matrix]
        # (1)
        pen_1 = diag_part(self.kernel.matrix(self.batch_data, self.batch_data))
        # (2)
        pen_2 = reduce_mean(transpose(tf.vectorized_map(lambda d: scalar_mul((1/d[1]), reduce_mean(self.kernel.matrix(d[0], self.batch_data), axis=0)), elems=[transpose(stack(domains)), transpose(self.get_kme_squared_norm(domains))])), axis=-1)
        # (3)
        pen_3 = reduce_mean(reduce_sum(transpose(tf.vectorized_map(lambda d_j: d_j[1] * reduce_sum(transpose(
            tf.vectorized_map(lambda d_k: d_k[1] * reduce_mean(self.kernel.matrix(d_k[0], d_j[0])),
                              elems=[transpose(domains), transpose(self.alpha_coefficients)])), axis=-1),
                                                                   elems=[transpose(domains),
                                                                          transpose(self.alpha_coefficients)])), axis=-1))

        return sqrt(reduce_mean(pen_1) + float(-2.0) * reduce_mean(pen_2) + reduce_mean(pen_3))

    def set_domains(self, domains):
        self.domains = domains

    def set_batch_sample(self, batch_sample):
        self.batch_sample = batch_sample

    def set_alpha_coefficients(self, alpha_coefficients):
        self.alpha_coefficients = alpha_coefficients

    def set_penalty(self, penalty):
        self.penalty = penalty

    def set_input(self, h):
        self.batch_data = h

    def set_param(self, param):
        self.param = param

class KMEOrthogonalityRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, param=0.01, kernel=None, domains=None):
        self.param = param
        #self.domains = domains
        self.kernel = kernel

    def __call__(self, weight_matrix):
        gram_matrix = (self.kernel.matrix(weight_matrix, weight_matrix))
        return self.param * tf.math.reduce_sum(tf.norm(gram_matrix, axis=0))


    def set_domains(self, domains):
        self.domains = domains



'''ragularizations as described in https://arxiv.org/abs/1810.09102.'''

class L2Gram(tf.keras.regularizers.Regularizer):
    def __init__(self, param=0.01, kernel=None):
        self.param = param
        #self.domains = domains
        self.kernel = kernel

    def __call__(self, weight_matrix):
        gram_matrix = (self.kernel.matrix(weight_matrix, weight_matrix))
        return self.param * tf.math.reduce_sum(tf.norm(gram_matrix, axis=0))*(1/gram_matrix.shape[0])

class L1Gram(tf.keras.regularizers.Regularizer):
    def __init__(self, param=0.01, kernel=None, domains=None):
        self.param = param
        #self.domains = domains
        self.kernel = kernel

    def __call__(self, weight_matrix):
        gram_matrix = (self.kernel.matrix(weight_matrix, weight_matrix))
        return self.param * tf.math.reduce_sum(tf.norm(gram_matrix, axis=0, ord=1))*(1/gram_matrix.shape[0])

class DSOGram(tf.keras.regularizers.Regularizer):
    def __init__(self, param=0.01, kernel=None, domains=None):
        self.param = param
        #self.domains = domains
        self.kernel = kernel

    def __call__(self, weight_matrix):
        gram_matrix = (self.kernel.matrix(weight_matrix, weight_matrix))
        return self.param * tf.math.reduce_sum(tf.norm(tf.matmul(gram_matrix)))*(1/gram_matrix.shape[0])

class SOGram(tf.keras.regularizers.Regularizer):
    def __init__(self, param=0.01, kernel=None, domains=None):
        self.param = param
        #self.domains = domains
        self.kernel = kernel

    def __call__(self, weight_matrix):
        gram_matrix = (self.kernel.matrix(weight_matrix, weight_matrix))
        return self.param * reduce_mean((tf.norm(tf.matmul(gram_matrix, tf.transpose(gram_matrix)) - tf.eye(weight_matrix.shape[0]), ord='fro', axis=(0, 1))))


class SRIPGram(tf.keras.regularizers.Regularizer):
    def __init__(self, param=0.01, kernel=None, domains=None):
        self.param = param
        #self.domains = domains
        self.kernel = kernel

    def __call__(self, weight_matrix):
        gram_matrix = (self.kernel.matrix(weight_matrix, weight_matrix))
        return self.param * (tf.linalg.svd(tf.matmul(gram_matrix, tf.transpose(gram_matrix)) - tf.eye(weight_matrix.shape[0]), compute_uv=False)[0])


class HSICRegularizer(tf.keras.regularizers.Regularizer):

    ''' Regularizer to make the domains independent'''

    def __init__(self, param=0.5, kernel=None, domains=None):
        self.param = param
        self.kernel = kernel
        self.domains = domains

        # WARNING: each domain has to have the same dimension
        self.domain_dimension = domains[0].shape[0]
        self.H = tf.eye(self.domain_dimension) - (1/self.domain_dimension) * tf.ones(shape=(self.domain_dimension, self.domain_dimension))

    def __call__(self, weight_matrix):

        # compute the Gram matrices of each domain
        gram_matrices = [self.H * self.kernel.matrix(domain, domain) * self.H for domain in self.domains]

        hisc_list = []
        for domain_i in range(len(self.domains)):
            for domain_j in range(domain_i+1, len(self.domains)):
                hisc_list.append(tf.linalg.trace(gram_matrices[domain_i] * gram_matrices[domain_i]))

        hisc_penalty = (1/self.domain_dimension) * tf.math.reduce_sum(hisc_list)

        return self.param * hisc_penalty


    def set_domains(self, domains):
        self.domains = domains



