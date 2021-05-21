import numpy as np
import tensorflow as tf

from tensorflow.python.keras.layers import Layer
from tensorflow.python.ops.math_ops import reduce_sum, add, scalar_mul, reduce_mean, tanh, multiply, sqrt, pow
from tensorflow.python.ops.linalg.linalg import diag_part, diag
from tensorflow.python.framework import dtypes, ops

from tensorflow.python.ops.array_ops import stack
from tensorflow.python.ops import nn
from tensorflow.python.ops.gen_math_ops import mat_mul
from tensorflow.python.ops.nn_ops import softmax
from tensorflow.python.keras import activations

from tensorflow.python.keras.constraints import max_norm, MinMaxNorm

from tensorflow.python.keras import backend as K

import tensorflow_probability as tfp

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

          num_domains: int
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
                 num_domains,
                 domain_dimension,
                 sigma=0.5,
                 normalized_kernel=False,
                 amplitude=None,
                 kernel=None,
                 similarity_measure='projected',  #`MMD` 'cosine_similarity'
                 softness_param=1.0,
                 domain_reg_method="None",
                 domain_reg_param=0.0,
                 bias=True,
                 activation=None,
                 representer=False,
                 units=None,
                 **kwargs):
        super(DomainAdaptationLayer, self).__init__(**kwargs)

        self.num_domains = num_domains
        self.domain_dimension = domain_dimension
        self.activation = activations.get(activation)
        # kernel attributes (Gaussian kernel is set as default)
        self.sigma = sigma
        self.amplitude = amplitude
        if kernel is None:
            self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(name='gaussian_kernel', amplitude=self.amplitude, length_scale=self.sigma, feature_ndims=1)
            #self.kernel = tfp.math.psd_kernels.RationalQuadratic(name='rational_quadratic', scale_mixture_rate=None, amplitude=self.amplitude, length_scale=self.sigma, feature_ndims=1)
            #self.kernel = tfp.math.psd_kernels.MaternFiveHalves(length_scale=sigma, feature_ndims=1, amplitude=amplitude, name="generalized_matern")

        else:
            self.kernel = kernel

        self.similarity_measure = similarity_measure
        self.softness_param = softness_param

        self.domain_reg_method = domain_reg_method
        self.domain_reg_param = domain_reg_param

        self.bias = bias

        self.domain_reg = None
        self.units = units
        self.representer = representer
        #self.c_trans = True if self.similarity_measure.lower() in ['mmd', 'cosine_similarity'] else False
        self.c_trans = False

    def build(self, input_shape):

        if self.units is None:
            self.units = input_shape[-1]

        if self.domain_reg_param > 0.0:
            self.domain_basis_reg_dict = {"domain_reg_{}".format(domain_num): DomainRegularizer(domains=None,
                                                                                                kernel=self.kernel,
                                                                                                domain_number=domain_num,
                                                                                                param=self.domain_reg_param,
                                                                                                similarity_measure = self.similarity_measure,
                                                                                                amplitude=self.amplitude,
                                                                                                orthogonalization_penalty=self.domain_reg_method) for domain_num in range(self.num_domains)}

            self.domain_reg = True

        else:
            self.domain_basis_reg_dict = {"domain_reg_{}".format(domain_num): None for domain_num in range(self.num_domains)}
            self.domain_reg = False

        self.domain_basis = {'domain_{}'.format(domain): self.add_weight(name="domain_basis_" + str(domain),
                                                                         shape=(self.domain_dimension, input_shape[-1],),
                                                                         trainable=True,
                                                                         regularizer=self.domain_basis_reg_dict["domain_reg_{}".format(domain)],
                                                                         initializer=tf.keras.initializers.RandomNormal(mean=domain*5*(-1)**(-1)**(domain), stddev=(domain+1)*0.05)) for domain in range(self.num_domains)}
        if self.c_trans:
            self.C_domains = {'domain_{}'.format(domain): self.add_weight(name="C_domain_" + str(domain), shape=(input_shape[-1], input_shape[-1]), trainable=True, initializer=tf.keras.initializers.GlorotUniform()) for domain in range(self.num_domains)}

        #uodate the regularization parameters
        if self.domain_reg:
            for domain in range(len(self.domain_basis_reg_dict)):
                domain_reg_key = "domain_reg_{}".format(domain)
                self.domain_basis_reg_dict[domain_reg_key].set_domains(list(self.domain_basis.values()))
                #if self.similarity_measure.lower() in ['cosine_similarity', 'mmd']:
                #    self.domain_basis_reg_dict[domain_reg_key].set_C(list(self.C_domains.values()))

        if self.representer:
            self.W_domains = {'domain_{}'.format(domain): self.add_weight(name="weights_domain_"+str(domain), shape=(self.domain_dimension, self.units,), trainable=True,
                                                                          initializer=tf.keras.initializers.GlorotUniform()) for domain in range(self.num_domains)}

        else:
            self.W_domains = {'domain_{}'.format(domain): self.add_weight(name="weights_domain_"+str(domain), shape=(input_shape[-1], self.units,), trainable=True,
                                                                          initializer=tf.keras.initializers.GlorotUniform()) for domain in range(self.num_domains)}

        if self.bias:
            self.B_domains = {'domain_{}'.format(domain): self.add_weight(name="bias_domain_"+str(domain), shape=(self.units, ), trainable=True, initializer=tf.keras.initializers.Zeros()) for domain in range(self.num_domains)}

        super(DomainAdaptationLayer, self).build(input_shape)

    # for debugging purpose
    # h = tf.random.normal(shape=(42, input_shape[-1]))
    # h = tf.random.normal(shape=input_shape)
    # h = tf.Tensor(input_shape, dtype=np.float32)
    # x_domain = tf.random.normal(shape=(2000, 45))*1

    # domain = 'domain_0'
    # domain = 0

    # domains = list(self.domain_basis.values())

    def call(self, h):
        if self.similarity_measure == 'projected':
            domain_probability = self.compute_alpha(h)

        elif self.similarity_measure == 'cosine_similarity':
            domain_probability = self.cosine_similarity_softmax(h)

        else:
            domain_probability = self.mmd_softmax(h)

        if self.domain_reg_param > 0.0:
            for domain in range(len(self.domain_basis_reg_dict)):
                domain_reg_key = "domain_reg_{}".format(domain)
                self.domain_basis_reg_dict[domain_reg_key].set_domains([list(self.domain_basis.values())[j] for j in list(set(range(len(self.domain_basis_reg_dict))) - {domain})])
                self.domain_basis_reg_dict[domain_reg_key].set_input(h)
                self.domain_basis_reg_dict[domain_reg_key].set_alpha_coefficients(domain_probability)
                self.domain_basis_reg_dict[domain_reg_key].set_param(self.domain_reg_param)
                #if self.similarity_measure.lower() in ['mmd', 'cosine_similarity']:
                #    self.domain_basis_reg_dict[domain_reg_key].set_C(list(self.C_domains.values()))


        # h_matrix_matmul_old = [mat_mul(h, self.W_domains[domain]) for domain in self.W_domains.keys()]
        # h_matrix_matmul_old = stack([mat_mul(h, self.W_domains[domain]) for domain in self.W_domains.keys()])
        # h_matrix_bias_added_old = tf.transpose(tf.concat(h_matrix_matmul_old, axis=-1))
        # h_matrix_matmul = tf.map_fn(lambda W: mat_mul(h, W), elems=stack(list(self.W_domains.values())))

        h_matrix_matmul = tf.vectorized_map(lambda W:mat_mul(h, W),
                                            elems=stack(list(self.W_domains.values())))




        # h_matrix_matmul_old - h_matrix_matmul

        if self.bias:
            h_matrix_matmul =tf.squeeze(tf.vectorized_map(lambda t: nn.bias_add(tf.expand_dims(t[0], axis=-1), t[1]),
                               elems=[h_matrix_matmul, stack(list(self.B_domains.values()))]
                               ), axis=-1)

            # h_matrix_bias_added_old - h_matrix_matmul

        if self.activation is not None:
            if False:
                h_prob_weighted = tf.map_fn(lambda t:
                                            self.activation(tf.multiply(t[:, -1], t[:, -2])),
                                            elems=tf.transpose(stack([tf.concat(h_matrix_matmul, axis=-1), stack(domain_probability)], axis=0))
                                    )


            h_prob_weighted = tf.vectorized_map(lambda t:
                                        multiply(tf.transpose(self.activation(t[0])), t[1]),
                                        #self.activation(t[0]),
                                        elems=[h_matrix_matmul, tf.transpose(domain_probability)]
                                        )



            if False:
                l_0 = tf.vectorized_map(lambda t:
                                            #multiply(self.activation(t[0]), t[1]),
                                            tf.transpose(self.activation(t[0])),
                                            elems=[h_matrix_matmul, tf.transpose(domain_probability)]
                                            )
                l_1 = tf.vectorized_map(lambda t:
                                            #multiply(self.activation(t[0]), t[1]),
                                            t[1],
                                            elems=[h_matrix_matmul, tf.transpose(domain_probability)]
                                           )

                h_prob_weighted
                a = multiply(l_0[0], l_1[0])
                a-h_prob_weighted[0]

        else:

            if False:
                h_prob_weighted = tf.map_fn(lambda t:
                                            tf.multiply(t[:, -1], t[:, -2]),
                                            elems=tf.transpose(stack([tf.concat(h_matrix_matmul, axis=-1), stack(domain_probability)], axis=0))
                                            )

            h_prob_weighted = tf.vectorized_map(lambda t:
                                                multiply(tf.transpose(t[0]), t[1]),
                                                elems=[h_matrix_matmul, tf.transpose(domain_probability)]
                                                )

        h_out = tf.transpose(reduce_sum(h_prob_weighted, axis=0))
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

    #@tf.function
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

        if self.c_trans:
            h_transformed = [tanh(mat_mul(h, C)) for C in list(self.C_domains.values())]
            mmd = tf.map_fn(lambda d: (diag_part(self.kernel.matrix(h, h)) - 2 * reduce_mean(self.kernel.matrix(h, d), axis=1) + reduce_mean(self.kernel.matrix(d, d))), elems=stack(h_transformed))
        else:
            mmd = tf.map_fn(lambda d: (diag_part(self.kernel.matrix(h, h)) - 2 * reduce_mean(self.kernel.matrix(h, d), axis=1) + reduce_mean(self.kernel.matrix(d, d))), elems=stack(domains))

        domain_probability_mmd = tf.transpose(softmax((-1) * self.softness_param * mmd, axis=0))

        return domain_probability_mmd

    @tf.function
    def cosine_similarity_softmax(self, h, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        if self.c_trans:
            h_transformed = [tanh(mat_mul(h, C)) for C in list(self.C_domains.values())]
            cosine_sim = tf.map_fn(lambda d: self.softness_param * reduce_mean(self.kernel.matrix(h, d), axis=1) / (sqrt(tf.linalg.diag_part(self.kernel.matrix(h, h))) * float(1 / self.domain_dimension) * sqrt(reduce_sum(self.kernel.matrix(d, d)))), elems=stack(h_transformed))
        else:
            cosine_sim = tf.map_fn(lambda d: self.softness_param * reduce_mean(self.kernel.matrix(h, d), axis=1) / (sqrt(tf.linalg.diag_part(self.kernel.matrix(h, h))) * float(1 / self.domain_dimension) * sqrt(reduce_sum(self.kernel.matrix(d, d)))), elems=stack(domains))

        domain_probability_cosine_sim = tf.transpose(softmax(cosine_sim, axis=0))

        return domain_probability_cosine_sim

    @tf.function
    def compute_alpha(self, h, domains=None):
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme = diag_part(self.get_kme_gram(domains=domains))

        if False:
            alpha = tf.squeeze(tf.map_fn(lambda d: reduce_mean(self.kernel.matrix(h, d), axis=-1, keepdims=True),
                                         elems=stack(list(self.domain_basis.values()))), axis=-1)

        alpha = tf.squeeze(tf.vectorized_map(lambda d: reduce_mean(self.kernel.matrix(h, d[0])/d[1], axis=-1, keepdims=True),
                                     elems=[stack(domains), squared_kme]), axis=-1)


        return tf.transpose(alpha)


    @tf.function
    def get_mmd_penalty(self, h, domain_probability=None):
        if domain_probability is not None:
            self.alpha_coefficients = domain_probability
        else:
            self.alpha_coefficients = self.get_domain_prob(h)

        domains_all = list(self.domain_basis.values())
        test_mmd_list = []
        for domain_number in range(len(domains_all)):
            weight_matrix = list(self.domain_basis.values())[domain_number]
            domains = [domains_all[k] for k in list(set(range(self.num_domains)) - {domain_number})]
            domains = domains + [weight_matrix]


            self.alpha_coefficients = self.get_domain_prob(h, domains)

            # (1)
            pen_1 = diag_part(self.kernel.matrix(h, h))

            # (2)
            # pen_2_old = reduce_mean(self.alpha_coefficients[:, domain_number] * reduce_mean(self.kernel.matrix(h, weight_matrix), axis=-1))
            kme_squared_norm = self.get_kme_squared_norm(domains)
            pen_2 = reduce_mean(tf.squeeze(tf.vectorized_map(lambda d: pow(reduce_mean(self.kernel.matrix(h, d[0]), axis=-1, keepdims=True), 2) / d[1] ,elems=[stack(domains), kme_squared_norm]), axis=-1), axis=0)
            #pen_2_old - pen_2
            # (3)
            pen_3 = reduce_sum(tf.vectorized_map(lambda d_j : reduce_sum(tf.vectorized_map(lambda d_k : multiply(d_j[1], d_k[1]) * reduce_mean(self.kernel.matrix(d_j[0], d_k[0])), elems=[domains, self.alpha_coefficients]), axis=0), elems=[domains, self.alpha_coefficients]), axis=-1)

            #domain_inner_products = [reduce_mean(self.kernel.matrix(weight_matrix, domain)) for domain in domains + [weight_matrix]]

           #pen_3 = reduce_sum(
           #    [ reduce_mean(tf.matmul(domain_inner_products[k] * tf.expand_dims(self.alpha_coefficients[:, domain_number], axis=-1), tf.expand_dims(self.alpha_coefficients[:, k], axis=-1), transpose_b=True)) for k in range(len(domains))] \
           #    +
           #    [reduce_mean(domain_inner_products[-1] * tf.matmul(tf.expand_dims(self.alpha_coefficients[:, domain_number], axis=-1), tf.expand_dims(self.alpha_coefficients[:, domain_number], axis=-1), transpose_b=True))]
           #)

            mmd_penalty = sqrt(reduce_mean(pen_1 + (-2) * pen_2 + pen_3))

            test_mmd_list.append(mmd_penalty)

        return mmd_penalty














    ####################
    ###     OLD     ####
    ####################
    # not used
    #@tf.function
    def get_batch_mmd(self, h, domain='domain_0'):
        if type(domain) == str:
            temp_1 = tf.linalg.diag_part(self.kernel.matrix(h, h))
            temp_2 = reduce_mean(self.kernel.matrix(h, self.domain_basis[domain]), axis=1)
            temp_3 = tf.squeeze(scalar_mul(tf.reduce_mean(self.kernel.matrix(self.domain_basis[domain], self.domain_basis[domain])), tf.ones(shape=(h.shape[0], 1))), axis=-1)
            mmd = temp_1 - 2 * temp_2 + temp_3
        else:
            temp_1 = tf.linalg.diag_part(self.kernel.matrix(h, h))
            temp_2 = reduce_mean(self.kernel.matrix(h, domain), axis=1)
            if h.shape[0] is None:
                temp_3 = tf.squeeze(scalar_mul(tf.reduce_mean(self.kernel.matrix(domain, domain)), tf.ones(shape=(1, 1))), axis=-1)
            else:
                temp_3 = tf.squeeze(scalar_mul(tf.reduce_mean(self.kernel.matrix(domain, domain)), tf.ones(shape=(h.shape[0], 1))), axis=-1)

            mmd = temp_1 - 2 * temp_2 + temp_3
        return mmd
    # not used
    def cosine_similarity(self, h, domain):
        reduce_mean(self.kernel.matrix(h, domain), axis=1) / (sqrt(diag_part(self.kernel.matrix(h, h))) * float(1 / self.domain_dimension) * sqrt(reduce_sum(self.kernel.matrix(domain, domain))))

    # not used
    @tf.function
    def get_batch_cosine_similarity(self, h, domain='domain_0', cosine_sim=True):
        if type(domain) == str:
            if cosine_sim:
                return reduce_mean(self.kernel.matrix(h, self.domain_basis[domain]), axis=1) / (sqrt(tf.linalg.diag_part(self.kernel.matrix(h, h))) * float(1/self.domain_dimension) * sqrt(reduce_sum(self.kernel.matrix(self.domain_basis[domain], self.domain_basis[domain]))))
            else:
                return reduce_mean(self.kernel.matrix(h, self.domain_basis[domain]), axis=1)
        else:
            if cosine_sim:
                return reduce_mean(self.kernel.matrix(h, domain), axis=1) / (sqrt(tf.linalg.diag_part(self.kernel.matrix(h, h))) * float(1/self.domain_dimension) * sqrt(reduce_sum(self.kernel.matrix(domain, domain))))
            else:
                return reduce_mean(self.kernel.matrix(h, domain), axis=1)
    # not used in here
    @tf.function
    def inner_product_domain(self, domain_1, domain_2):
        inner_product = reduce_mean(self.kernel.matrix(domain_1, domain_2))
        return inner_product

    @tf.function
    def squared_norm_domain(self, domain):
        squared_norm = reduce_mean(self.kernel.matrix(domain, domain))
        return squared_norm

    # not used here
    @tf.function
    def inner_product_batch_domain(self, h, domain):
        inner_product = reduce_mean(self.kernel.matrix(h, domain), axis=-1, keepdims=True)
        return inner_product


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


    @tf.function
    def get_domain_approximation_error(self, h):

        domain_adaptation_penalty = self.get_mmd_penalty(h)

        return np.sqrt(domain_adaptation_penalty.numpy())

    @tf.function
    def get_domain_gram_matrix(self):
        domains = list(self.domain_basis.values())
        # placeholder for the inner products between the domains
        domain_gram_matrix_list = [[0 for i in range(self.num_domains)] for j in range(self.num_domains)]
        for i in range(self.num_domains):
            for j in range(i, self.num_domains):
                domain_gram_matrix_list[i][j] = reduce_mean(self.kernel.matrix(domains[i], domains[j]))
                domain_gram_matrix_list[j][i] = reduce_mean(self.kernel.matrix(domains[i], domains[j]))

        domain_gram_matrix = tf.reshape(tf.concat(domain_gram_matrix_list, axis=-1), shape=(self.num_domains, self.num_domains))

        return domain_gram_matrix

    @tf.function
    def get_domain_distributional_variance(self):
        domain_gram_matrix = self.get_domain_gram_matrix()
        domain_distributional_variance = reduce_mean(tf.linalg.diag_part(domain_gram_matrix)) - reduce_mean(domain_gram_matrix)

        return domain_distributional_variance

    #@tf.function
    def domain_orthogonality_penalty(self):
        domains = list(self.domain_basis.values())
        domain_vectors = tf.concat(domains, axis=0)
        domain_orthogonal_res_dict = {}

        domain_gram = self.kernel.matrix(domain_vectors, domain_vectors)

        penalty_name = "SRIP"
        #domain_orthogonality_penalty = (tf.linalg.svd(tf.matmul(domain_gram, tf.transpose(domain_gram)) - tf.eye(domain_gram.shape[0]), compute_uv=False)[0])
        domain_orthogonality_penalty = (tf.linalg.svd(domain_gram - tf.eye(domain_gram.shape[0]), compute_uv=False)[0])
        domain_orthogonal_res_dict.update({penalty_name: np.round(domain_orthogonality_penalty.numpy(), 4)})

        penalty_name = "ICP"
        domain_orthogonality_penalty = reduce_mean([self.kernel.matrix(self.domain_basis["domain_"+str(i)], self.domain_basis["domain_"+str(j)]) for i in range(len(self.domain_basis)) for j in range(len(self.domain_basis)-1) if i!=j])
        domain_orthogonal_res_dict.update({penalty_name: np.round(domain_orthogonality_penalty.numpy(), 4)})

        penalty_name = "SO"
        domain_orthogonality_penalty = (tf.norm(domain_gram - tf.eye(domain_vectors.shape[0]), ord='fro', axis=(0, 1)))
        domain_orthogonal_res_dict.update({penalty_name: np.round(domain_orthogonality_penalty.numpy(), 4)})

        return domain_orthogonal_res_dict


    @tf.function
    def get_SRIP(self):
        domains = list(self.domain_basis.values())

        domain_vectors = tf.concat(domains, axis=0)
        domain_orthogonality_penalty = (1/domain_vectors.shape[0])*(tf.linalg.svd(tf.matmul(domain_vectors, tf.transpose(domain_vectors)) - tf.eye(domain_vectors.shape[0]), compute_uv=False)[0])

        return domain_orthogonality_penalty

    @tf.function
    def get_ICP(self):
        domain_gram_matrix = self.get_domain_gram_matrix()
        domain_orthogonality_penalty = reduce_mean(domain_gram_matrix)
        return domain_orthogonality_penalty

    #kme_matrix = tf.map_fn(fn= lambda d: tf.map_fn(fn=lambda t: reduce_mean(self.kernel.matrix(t, d)), elems=domains), elems=domains, parallel_iterations=10)


    def get_kme_inner_product(self, domain_i, domain_j):
        return reduce_mean(self.kernel.matrix(domain_i, domain_j))

        def get_kme_inner_product(domains):
            domain_i, domain_j = domains
            return  reduce_mean(self.kernel.matrix(domain_i, domain_j))


    def kme_gram(self):

        domains = np.array([[domain_i, domain_j] for domain_i in self.domains for domain_j in self.domains])

        from itertools import product
        a = list(product(list(self.domain_basis.values()), list(self.domain_basis.values())))

        self.kernel.matrix(domains, domains)

        b = tf.map_fn(fn=lambda domain: reduce_mean(self.kernel.matrix(domain, domain)), elems=stack(domains), swap_memory=True, infer_shape=False, parallel_iterations=10)

    def get_domain_basis(self):
        return {domain: self.domain_basis[domain].numpy() for domain in self.domain_basis.keys()}

    #@tf.function
    def get_domain_probability(self, h):

        if self.similarity_measure.lower() == 'cosine_similarity':
            if self.c_trans:
                domain_probability = softmax([self.softness_param * reduce_sum(self.kernel.matrix(tf.tanh(tf.matmul(h, self.C_domains[domain])), self.domain_basis[domain]), axis=-1) for domain in self.domain_basis.keys()], axis=0).numpy()
            else:
                domain_probability = softmax([self.softness_param * reduce_sum(self.kernel.matrix(h, self.domain_basis[domain]), axis=-1) for domain in self.domain_basis.keys()], axis=0).numpy()

        elif self.similarity_measure == 'projected':
            domain_probability = tf.concat([self.calculate_alpha(h, domain) for domain in self.domain_basis.values()], axis=-1)
            #domain_probability = tf.math.divide(domain_probability, reduce_sum(domain_probability, axis=-1, keepdims=True))
            #domain_probability = reduce_mean(domain_probability, axis=-1)

            domain_probability = reduce_mean(domain_probability, axis=0).numpy()

        else:

            if self.c_trans:
                domain_probability = softmax([self.softness_param * (tf.linalg.diag_part(self.kernel.matrix(h,  h))
                                    - (2 / self.domain_dimension) * reduce_sum(self.kernel.matrix(tf.tanh(tf.matmul(h, self.C_domains[domain])), self.domain_basis[domain]), axis=-1)
                                    + reduce_sum(self.kernel.matrix(self.domain_basis[domain], self.domain_basis[domain])).numpy() * tf.ones(shape=(h.shape[0],)))
                                                                    for domain in self.domain_basis.keys()], axis=0).numpy()
            else:
                domain_probability = softmax([self.softness_param * (tf.linalg.diag_part(self.kernel.matrix(h,  h))
                                    - (2 / self.domain_dimension) * reduce_sum(self.kernel.matrix(h, self.domain_basis[domain]), axis=-1)
                                    + reduce_sum(self.kernel.matrix(self.domain_basis[domain], self.domain_basis[domain])).numpy() * tf.ones(shape=(h.shape[0],)))
                                                                    for domain in self.domain_basis.keys()], axis=0).numpy()

        return domain_probability


    def set_domain_reg_param(self, param):
        self.domain_reg_param = param




class NormalizedKernel(tfp.math.psd_kernels.ExponentiatedQuadratic):
    def __init__(self, sigma, **kwargs):
        super(NormalizedKernel).__init__(**kwargs)
        self.sigma = sigma
        self.kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=self.sigma)

    #@tf.function
    def matrix(self, x1, x2):
        K = self.kernel.matrix(x1, x2)
        D1 = tf.math.reciprocal(tf.math.sqrt(self.kernel.matrix(x1, x1)))
        D2 = tf.math.reciprocal(tf.math.sqrt(self.kernel.matrix(x2, x2)))
        K = tf.matmul(D1, K)
        K = tf.matmul(K, D2)
        return K




class KernelSum(tfp.math.psd_kernels.ExponentiatedQuadratic):
    def __init__(self, sigma_list, **kwargs):
        super(KernelSum).__init__(**kwargs)
        self.sigma_list = sigma_list

        self.kernels = [tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=sigma) for sigma in sigma_list]

    @tf.function
    def matrix(self, x1, x2):
        return tf.math.reduce_mean([self.kernels[i].matrix(x1, x2) for i in range(len(self.sigma_list))], axis=0)
