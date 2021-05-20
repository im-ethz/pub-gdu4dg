import numpy as np
import tensorflow as tf

from tensorflow.python.ops.math_ops import reduce_sum, reduce_prod, reduce_mean, scalar_mul, add, tanh, multiply, sqrt
from tensorflow.python.ops.nn_ops import softmax
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
       orthogonalization_penalty : string, optional
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

    def __init__(self, param=0.1, domain_number=None, kernel=None, domains=None,
                 orthogonalization_penalty='SRIP', similarity_measure=None,
                 C_domains=None,
                 amplitude=1.0,
                 param_orth=None,
                 sparse_reg=True
                 ):

        self.param = param
        self.param_orth = param_orth if param_orth is not None else param
        self.domains = domains
        self.kernel = kernel
        self.aplitude = amplitude
        self.num_domains = None
        self.domain_dimension = None
        self.sparse_reg = sparse_reg
        self.domain_number = domain_number

        self.orthogonalization_penalty = orthogonalization_penalty
        self.similarity_measure = similarity_measure
        self.C_domains = C_domains if self.similarity_measure.lower() != "projected" else None

        self.lambda_alpha = 0.2
        self.lambda_sparse = 0.05
        self.mmd_penalty = True
        self.use_kme_gram = True

        # PLACEHOLDER (will be updated throughout the training)
        self.alpha = None
        self.batch_sample = None

    # for debugging purpose...
    #h = tf.random.normal(shape=(self.domain_basis['domain_0'].numpy().shape[0], input_shape[-1]))
    #h = tf.random.normal(shape=(42, self.domains[0].shape[1]))
    # self.batch_data = h
    #h = tf.Tensor(None, dtype=np.float32)
    #x_domain = tf.random.normal(shape=(2000, 45))*1

    #domain = 'domain_0'
    #domain = 0

    def __call__(self, weight_matrix):


        if self.num_domains is None:
            self.num_domains = len(self.domains)

        if self.domain_dimension is None:
            self.domain_dimension = self.domains[0].shape[0]

        #other_domains_list = [self.domains[k] for k in range(len(self.domains)) if k != self.domain_number]
        if self.orthogonalization_penalty.lower() in ['srip', "so", "mc"]:
            if self.use_kme_gram:
                domains = tf.stack(self.domains + [weight_matrix])
                gram_matrix = tf.map_fn(fn=lambda d: tf.map_fn(fn=lambda t: reduce_mean(self.kernel.matrix(t, d)), elems=domains), elems=domains, parallel_iterations=10)

            else:
                domain_vectors = tf.concat(self.domains + [weight_matrix], axis=0)
                gram_matrix = self.kernel.matrix(domain_vectors, domain_vectors)

            gram_diag = tf.linalg.diag(diag_part(gram_matrix))


        if self.orthogonalization_penalty.lower() == 'srip':
            domain_orthogonality_penalty = (tf.linalg.svd(gram_matrix - gram_diag, compute_uv=False)[0])
            self.orth_pen_name = "SRIP"

        elif self.orthogonalization_penalty.lower() == "so":
            domain_orthogonality_penalty = tf.norm(gram_matrix - gram_diag, ord='fro', axis=(0, 1))
            self.orth_pen_name = "soft_orthogonality"

        elif self.orthogonalization_penalty.lower() == "mc":
            domain_orthogonality_penalty = tf.norm(gram_matrix - gram_diag, ord=np.inf, axis=(0, 1))
            self.orth_pen_name = "mutual_coherence"

        elif self.orthogonalization_penalty.lower() == 'icp':
            domain_orthogonality_penalty = reduce_sum([reduce_mean(self.kernel.matrix(domain, weight_matrix)) for domain in self.domains])
            self.orth_pen_name = "cross_domain_IPS"

        else:
            domain_orthogonality_penalty = float(0.0)
            self.orth_pen_name = "NONE"

        self.orth_pen = float(1/(self.num_domains+1)) * (domain_orthogonality_penalty)

        ########################
        # MMD REG
        #######################
        if self.mmd_penalty:
            if self.sparse_reg:
                self.mmd = 0.5 * self.get_mmd_penalty(weight_matrix) + (self.lambda_sparse / 2) * reduce_mean(self.kernel.matrix(weight_matrix, weight_matrix)) + self.lambda_alpha * tf.norm(self.alpha_coefficients, ord=1)

            else:
                self.mmd = 0.5 * self.get_mmd_penalty(weight_matrix)

        else:
            self.mmd = float(0)

        return self.param * self.mmd + self.param_orth * self.orth_pen

    def get_config(self):
        return {self.orth_pen_name: self.orth_pen,
                'MMD': self.domain_mmd
                }

    #@tf.function
    def get_mmd_penalty(self, weight_matrix):
        # (1)
        pen_1 = float(1 / self.num_domains + 1) * reduce_mean(diag_part(self.kernel.matrix(self.batch_data, self.batch_data)))
        # (2)
        pen_2 = reduce_mean(multiply(self.alpha_coefficients[:, self.domain_number], reduce_mean(self.kernel.matrix(self.batch_data, weight_matrix), axis=-1)))
        # (3)
        pen_3 = reduce_mean(
        [reduce_mean(self.kernel.matrix(weight_matrix, self.domains[k])) * reduce_sum(tf.matmul(tf.expand_dims(self.alpha_coefficients[:, self.domain_number], axis=-1), tf.expand_dims(self.alpha_coefficients[:, k], axis=-1), transpose_b=True)) for k in range(len(self.domains))] \
        +
        [reduce_mean(self.kernel.matrix(weight_matrix, weight_matrix)) * reduce_sum(tf.matmul(tf.expand_dims(self.alpha_coefficients[:, self.domain_number], axis=-1), tf.expand_dims(self.alpha_coefficients[:, self.domain_number], axis=-1), transpose_b=True))]
        )

        return sqrt(pen_1 + float(-2.0) * pen_2 + pen_3)

    def set_domains(self, domains):
        self.domains = domains

    def set_C(self, C_domains):
        self.C_domains = C_domains

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

    #@tf.function
    def get_kme_gram(self, weight_matrix):
        from datetime import datetime
        start = datetime.now()
        kme_gram_list = []
        for i in range(self.num_domains + 1):
            kme_gram_row_list = []
            for j in range(self.num_domains + 1):
                domain_i = weight_matrix if i == self.num_domains else self.domains[i]
                domain_j = weight_matrix if j == self.num_domains else self.domains[j]
                #print("i: {i}, j: {j}, duration:{dur}".format(i=i, j=j, dur=datetime.now()-start))
                start = datetime.now()

                kme_gram_row_list.append(reduce_mean(self.kernel.matrix(domain_i, domain_j)))

            kme_gram_row_tensor = tf.stack(kme_gram_row_list, axis=0)
            kme_gram_list.append(kme_gram_row_tensor)

        kme_gram_tensor = tf.stack(kme_gram_list, axis=0)
        return kme_gram_tensor

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



