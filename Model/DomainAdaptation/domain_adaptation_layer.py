import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops.math_ops import reduce_sum, add, scalar_mul, reduce_mean,  multiply, sqrt
from tensorflow.python.ops.linalg.linalg import diag_part, diag

from tensorflow.python.ops.array_ops import stack, transpose, expand_dims, squeeze
from tensorflow.python.ops import nn
from tensorflow.python.ops.gen_math_ops import mat_mul
from tensorflow.python.ops.nn_ops import softmax
from tensorflow.python.keras import activations
from tensorflow.python.ops.parallel_for.control_flow_ops import vectorized_map
from tensorflow.python.keras.initializers import GlorotUniform

from tensorflow.python.keras.constraints import max_norm, MinMaxNorm

from tensorflow.python.keras import backend as K

import tensorflow_probability as tfp
from tensorflow_probability.python.math.psd_kernels import ExponentiatedQuadratic, RationalQuadratic, MaternFiveHalves


from Model.DomainAdaptation.domain_adaptation_regularization import DomainRegularizer




class DomainAdaptationLayer(Layer):
    """

          This layer performs a unsupervised domain adaptation. The layer contains domains in the form of weighs ('domain_basis).
          Each time the layer is called the input os compared

          Parameters
          -----------
          kernel : tfp.math.psd_kernels instance, optional
              kernel function for the domains and the acossiated RKHS,
              Gaussian kernel is used as a default

          sigma: double, optional
              sigma of the Gaussian kernel function

          similarity_measure : string, in [`projected`,`MMD`,  'cosine_similarity']
              method how each domain is weighted in the computation of the output
              projected (default): orthogonal projection of the input into the domains, asuming orthogonal domains
              MMD: Softmax of the MMD of each domain with the
              cosine_similarity: inner product similarity of the input and each domain

          softness_param: double, optional
              used only in case of when similarity_measure is MMD od cosine_similarity. Is used as scaling factor in the
              softmax function. A higher value leads to a higher probability for higher values in the softmax function.

          domain_units: int
              numbe of domains included in the layer

          domain_dim: int
              number of the vectors included in each domain

          domain_reg_method: string
              method, which is used a regularization for the domains

          domain_reg_param: double
              how strong the regularization is taken into account

          bias: boolean
              if `True` the model will include bias



          Notes
          ------
          - will be done later

          Examples
          --------
            >>> # simple example how the layer can be included in a Sequential model
            >>> model = Sequential(Input())
            >>> model.add(Dense(100))
            >>> model.add(Dense(10))
            >>> model.add(DomainAdaptationLayer(num_domains=42,
            >>>                     domain_dimension=25,
            >>>                     softness_param=5,
            >>>                     sigma=1.2,
            >>>                     similarity_measure='projected',
            >>>                     domain_reg_method='SRIP',
            >>>                     domain_reg_param=1e-4))

          """
    def __init__(self,

                 # domain params
                 domain_units,
                 M, # basis size
                 kernel=None, # Gaussian kernel is set as default
                 sigma=0.5,
                 amplitude=None,

                 # domain inference parameter
                 similarity_measure='projected',  #`MMD` 'cosine_similarity'
                 softness_param=1.0, # softness parameter, which is used in case of mmd- or cosine-similarity

                 # network parameter
                 units=None,
                 activation=None,
                 bias=True,

                 # regularization params
                 domain_reg_method="None",
                 lambda_OLS=1e-2,
                 lambda_orth=0.2,
                 lambda_sparse=0.05,

                 **kwargs):
        super(DomainAdaptationLayer, self).__init__(**kwargs)

        self.num_domains = domain_units
        self.domain_dimension = M
        self.activation = activations.get(activation)

        # kernel attributes (Gaussian kernel is set as default)
        self.sigma = sigma
        self.amplitude = amplitude
        if kernel is None:
            self.kernel = ExponentiatedQuadratic(name='gaussian_kernel', amplitude=self.amplitude, length_scale=self.sigma, feature_ndims=1)
            #self.kernel = RationalQuadratic(name='rational_quadratic', scale_mixture_rate=None, amplitude=self.amplitude, length_scale=self.sigma, feature_ndims=1)
            #self.kernel = MaternFiveHalves(length_scale=sigma, feature_ndims=1, amplitude=amplitude, name="generalized_matern")

        else:
            self.kernel = kernel

        self.similarity_measure = similarity_measure
        self.softness_param = softness_param

        self.domain_reg_method = domain_reg_method

        self.bias = bias

        self.domain_reg = None
        self.units = units

        self.lambda_OLS = lambda_OLS
        self.lambda_sparse = lambda_sparse
        self.lambda_orth = lambda_orth if self.similarity_measure.lower() == "projecton" else 0

        self.domain_reg = bool(max(self.lambda_OLS, self.lambda_orth, self.lambda_orth) > 0.0)

    def build(self, input_shape):

        if self.units is None:
            self.units = input_shape[-1]

        if self.domain_reg:
            self.domain_basis_reg_dict = {"domain_reg_{}".format(domain_num): DomainRegularizer(domains=None,
                                                                                                kernel=self.kernel,
                                                                                                domain_number=domain_num,
                                                                                                lambda_sparse=self.lambda_sparse,
                                                                                                lambda_OLS=self.lambda_OLS,
                                                                                                lambda_orth=self.lambda_orth,
                                                                                                similarity_measure=self.similarity_measure,
                                                                                                orth_pen_method=self.domain_reg_method) for domain_num in range(self.num_domains)}

        else:
            self.domain_basis_reg_dict = {"domain_reg_{}".format(domain_num): None for domain_num in range(self.num_domains)}



        self.domain_basis = {'domain_{}'.format(domain): self.add_weight(name="domain_basis_" + str(domain),
                                                                         shape=(self.domain_dimension, input_shape[-1],),
                                                                         trainable=True,
                                                                         regularizer=self.domain_basis_reg_dict["domain_reg_{}".format(domain)],
                                                                         initializer=tf.keras.initializers.RandomNormal(mean=domain*5*(-1)**(domain), stddev=(domain+1)*0.05)) for domain in range(self.num_domains)}

        #uodate the regularization parameters
        if self.domain_reg:
            for domain in range(len(self.domain_basis_reg_dict)):
                domain_reg_key = "domain_reg_{}".format(domain)
                self.domain_basis_reg_dict[domain_reg_key].set_domains(list(self.domain_basis.values()))


        # domain weights anf bias

        self.W_domains = {'domain_{}'.format(domain): self.add_weight(name="weights_domain_"+str(domain), shape=(input_shape[-1], self.units,), trainable=True,
                                                                          initializer=GlorotUniform()) for domain in range(self.num_domains)}

        if self.bias:
            self.B_domains = {'domain_{}'.format(domain): self.add_weight(name="bias_domain_"+str(domain), shape=(self.units, ), trainable=True, initializer=tf.keras.initializers.Zeros()) for domain in range(self.num_domains)}

        super(DomainAdaptationLayer, self).build(input_shape)

    # for debugging purpose
    # h = tf.random.normal(shape=(42, input_shape[-1]))
    # h = tf.random.normal(shape=input_shape)
    # h = tf.Tensor(input_shape, dtype=np.float32)


    def call(self, h):
        if self.similarity_measure == 'projected':
            domain_probability = self.compute_alpha(h)

        elif self.similarity_measure == 'cosine_similarity':
            domain_probability = self.cosine_similarity_softmax(h)

        else:
            domain_probability = self.mmd_softmax(h)

        if self.domain_reg:
            for domain in range(len(self.domain_basis_reg_dict)):
                domain_reg_key = "domain_reg_{}".format(domain)
                self.domain_basis_reg_dict[domain_reg_key].set_domains([list(self.domain_basis.values())[j] for j in list(set(range(len(self.domain_basis_reg_dict))) - {domain})])
                self.domain_basis_reg_dict[domain_reg_key].set_input(h)
                self.domain_basis_reg_dict[domain_reg_key].set_alpha_coefficients(domain_probability)

        h_matrix_matmul = vectorized_map(lambda W: mat_mul(h, W), elems=stack(list(self.W_domains.values())))

        if self.bias:
            h_matrix_matmul =squeeze(vectorized_map(lambda t: nn.bias_add(expand_dims(t[0], axis=-1), t[1]), elems=[h_matrix_matmul, stack(list(self.B_domains.values()))]), axis=-1)

        if self.activation is not None:
            h_prob_weighted = vectorized_map(lambda t: multiply(transpose(self.activation(t[0])), t[1]), elems=[h_matrix_matmul, transpose(domain_probability)])

        else:
            h_prob_weighted = vectorized_map(lambda t: multiply(transpose(t[0]), t[1]), elems=[h_matrix_matmul, transpose(domain_probability)])

        h_out = transpose(reduce_sum(h_prob_weighted, axis=0))
        return h_out

    @tf.function
    def get_kme_gram(self, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        kme_gram_matrix = tf.map_fn(fn=lambda d_i: tf.map_fn(fn=lambda d_j: reduce_mean(self.kernel.matrix(d_i, d_j)), elems=stack(domains)), elems=stack(domains), parallel_iterations=10)

        return kme_gram_matrix

    @tf.function
    def get_kme_squared_norm(self, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme_norm = tf.map_fn(fn=lambda d_j: reduce_mean(self.kernel.matrix(d_j, d_j)), elems=stack(domains))

        return squared_kme_norm

    @tf.function
    def get_domain_prob(self, h, domains=None):
        if self.similarity_measure == 'projected':
            domain_probability = self.compute_alpha(h, domains=domains)
        elif self.similarity_measure == 'cosine_similarity':
            domain_probability = self.cosine_similarity_softmax(h, domains=domains)
        else:
            domain_probability = self.mmd_softmax(h, domains=domains)

        return domain_probability


    @tf.function
    def mmd_softmax(self, h, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        mmd = tf.map_fn(lambda d: (diag_part(self.kernel.matrix(h, h)) - 2 * reduce_mean(self.kernel.matrix(h, d), axis=1) + reduce_mean(self.kernel.matrix(d, d))), elems=stack(domains))

        domain_probability_mmd = transpose(softmax((-1) * self.softness_param * mmd, axis=0))

        return domain_probability_mmd

    @tf.function
    def cosine_similarity_softmax(self, h, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        cosine_sim = tf.map_fn(lambda d: self.softness_param * reduce_mean(self.kernel.matrix(h, d), axis=1) / (sqrt(tf.linalg.diag_part(self.kernel.matrix(h, h))) * float(1 / self.domain_dimension) * sqrt(reduce_sum(self.kernel.matrix(d, d)))), elems=stack(domains))

        domain_probability_cosine_sim = transpose(softmax(cosine_sim, axis=0))

        return domain_probability_cosine_sim

    @tf.function
    def compute_alpha(self, h, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme = diag_part(self.get_kme_gram(domains=domains))

        alpha = squeeze(vectorized_map(lambda d: reduce_mean(self.kernel.matrix(h, d[0])/d[1], axis=-1, keepdims=True), elems=[stack(domains), squared_kme]), axis=-1)

        return transpose(alpha)


    #@tf.function
    def get_mmd_penalty(self, h):
        domains = list(self.domain_basis.values())
        self.alpha_coefficients = self.get_domain_prob(h, domains)

        # (1)
        pen_1 = reduce_mean(diag_part(self.kernel.matrix(h, h)))
        # (2)
        kme_squared_norm = self.get_kme_squared_norm(domains)
        pen_2 = reduce_mean(transpose(vectorized_map(lambda d: scalar_mul((1/d[1]), reduce_mean(self.kernel.matrix(d[0], self.batch_data), axis=0)), elems=[transpose(stack(domains)), transpose(kme_squared_norm)])), axis=-1)
        # (3)
        pen_3 = reduce_mean(reduce_sum(transpose(vectorized_map(lambda d_j: d_j[1] * reduce_sum(transpose(vectorized_map(lambda d_k: d_k[1] * reduce_mean(self.kernel.matrix(d_k[0], d_j[0])), elems=[transpose(domains), transpose(self.alpha_coefficients)])), axis=-1), elems=[transpose(domains), transpose(self.alpha_coefficients)])), axis=-1))

        mmd_penalty = sqrt(pen_1 + (-2) * pen_2 + pen_3)

        return mmd_penalty

    @tf.function
    def get_domain_distributional_variance(self):
        domain_gram_matrix = self.get_kme_gram()
        domain_distributional_variance = reduce_mean(diag_part(domain_gram_matrix)) - reduce_mean(domain_gram_matrix)
        return domain_distributional_variance



    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_domain_basis(self):
        return {domain: self.domain_basis[domain].numpy() for domain in self.domain_basis.keys()}


