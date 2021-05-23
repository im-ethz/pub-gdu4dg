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

import tensorflow_probability as tfp
from tensorflow_probability.python.math.psd_kernels import ExponentiatedQuadratic, RationalQuadratic, MaternFiveHalves


from Model.DomainAdaptation.domain_adaptation_regularization import DomainRegularizer


class DGLayer(Layer):
    """This layer performs a unsupervised domain adaptation. The layer contains domains in the form of weighs ('domain_basis).
          Each time the layer is called the input os compared

    Parameters
    ----------

    domain_units : `int`
        number of domain units (elementary domains)

    N : `int`
        number of basis vector included for each domain basis

    kernel : `tfp.math.psd_kernels`, optional
        kernel function for the domains and the acossiated RKHS,
        Gaussian kernel is used as a default

    sigma : `double`, optional
        bandwidth parameter for the Gaussian kernel

    similarity_measure : `string`, in [`projected`, `MMD`, 'cosine_similarity']
        method how each domain is weighted in the computation of the output (Default value = "projection")
            - projected: coefficients are determined by the projection method
            - MMD: softmax of the MMD feature mapping of the input and domain basis
            - cosine_similarity: feature mapping of the input and domain basis

    softness_param : `double`,
        softmax parameter for the mmd and cosine_similarity method (Default value = 1)

    units : `int`
        output dimension of the layer; if None, than it is the same as the input dimension (Default value = None)

    bias : `boolean`
        if `True`, the model will include bias (Default value = True)


    orth_pen_method : `string`
        method, used as orthogonal penalty  (Default value = "SO")

    domain_reg_param : `double`
        how strong the regularization is taken into account



    Notes
    ------

    Examples
    --------
    >>># simple example how the layer can be included in a Sequential model
            >>> model = Sequential(Input())
            >>> model.add(Dense(100))
            >>> model.add(DGLayer(num_domains=42,
            >>>                     domain_dimension=25,
            >>>                     softness_param=5,
            >>>                     units=10,
            >>>                     activation='sigmoid',
            >>>                     sigma=1.2,
            >>>                     similarity_measure='projected',
            >>>                     domain_reg_method='SO'))
    """
    def __init__(self,

                 # domain params
                 domain_units,
                 N,  # basis size
                 kernel=None,  # Gaussian kernel is set as default
                 sigma=0.5,


                 # domain inference parameter
                 similarity_measure='projected',  #`MMD` 'cosine_similarity'
                 softness_param=1.0,  # softness parameter, which is used in case of mmd- or cosine-similarity

                 # network parameter
                 units=None,
                 activation=None,
                 bias=True,

                 # regularization params
                 orth_pen_method="SO",
                 lambda_OLS=1e-3,
                 lambda_orth=1e-3,
                 lambda_sparse=1e-3,

                 **kwargs):
        super(DGLayer, self).__init__(**kwargs)

        self.num_domains = domain_units
        self.N = N
        self.activation = activations.get(activation)

        # kernel attributes (Gaussian kernel is set as default)
        self.sigma = sigma
        if kernel is None:
            self.kernel = ExponentiatedQuadratic(name='gaussian_kernel', length_scale=self.sigma, feature_ndims=1)
            #self.kernel = RationalQuadratic(name='rational_quadratic', scale_mixture_rate=None, amplitude=self.amplitude, length_scale=self.sigma, feature_ndims=1)
            #self.kernel = MaternFiveHalves(length_scale=sigma, feature_ndims=1, amplitude=amplitude, name="generalized_matern")

        else:
            self.kernel = kernel

        self.similarity_measure = similarity_measure
        self.softness_param = softness_param

        self.orth_pen_method = orth_pen_method

        self.bias = bias

        self.domain_reg = None
        self.units = units

        self.lambda_OLS = lambda_OLS
        self.lambda_sparse = lambda_sparse
        self.lambda_orth = lambda_orth if self.similarity_measure.lower() == "projected" else 0

        self.domain_reg = bool(max(self.lambda_OLS, self.lambda_orth, self.lambda_orth) > 0.0)

    def build(self, input_shape):
        """ build-method of the layer

        Parameters
        ----------
        input_shape : `tuple`
            shape of the input

        """

        if self.units is None:
            self.units = input_shape[-1]

        if self.domain_reg:
            self.domain_basis_reg_dict = {"domain_reg_{}".format(domain_num): DomainRegularizer(domains=None,
                                                                                                kernel=self.kernel,
                                                                                                domain_number=domain_num,
                                                                                                lambda_sparse=self.lambda_sparse,
                                                                                                softness_param=self.softness_param,
                                                                                                lambda_OLS=self.lambda_OLS,
                                                                                                lambda_orth=self.lambda_orth,
                                                                                                similarity_measure=self.similarity_measure,
                                                                                                orth_pen_method=self.orth_pen_method) for domain_num in range(self.num_domains)}

        else:
            self.domain_basis_reg_dict = {"domain_reg_{}".format(domain_num): None for domain_num in range(self.num_domains)}



        self.domain_basis = {'domain_{}'.format(domain): self.add_weight(name="domain_basis_" + str(domain),
                                                                         shape=(self.N, input_shape[-1],),
                                                                         trainable=True,
                                                                         regularizer=self.domain_basis_reg_dict["domain_reg_{}".format(domain)],
                                                                         initializer=tf.keras.initializers.RandomNormal(mean=domain*2*(-1)**(domain), stddev=(domain+1)*0.05)) for domain in range(self.num_domains)}

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

        super(DGLayer, self).build(input_shape)

    # for debugging purpose
    # h = tf.random.normal(shape=(42, input_shape[-1]))
    # h = tf.random.normal(shape=input_shape)
    # h = tf.Tensor(input_shape, dtype=np.float32)


    def call(self, h):
        """ call-method of the layer (forward-propagation)

        Parameters
        ----------
        h : `tf.Tensor`
            input of the layer
            

        Returns
        -------
        h_out : `tf.Tensor`
            propagated input

        """
        if self.similarity_measure == 'projected':
            domain_probability = self.get_projection_coef(h)

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
            h_matrix_matmul = squeeze(vectorized_map(lambda t: nn.bias_add(expand_dims(t[0], axis=-1), t[1]), elems=[h_matrix_matmul, stack(list(self.B_domains.values()))]), axis=-1)

        if self.activation is not None:
            h_prob_weighted = vectorized_map(lambda t: multiply(transpose(self.activation(t[0])), t[1]), elems=[h_matrix_matmul, transpose(domain_probability)])

        else:
            h_prob_weighted = vectorized_map(lambda t: multiply(transpose(t[0]), t[1]), elems=[h_matrix_matmul, transpose(domain_probability)])

        h_out = transpose(reduce_sum(h_prob_weighted, axis=0))
        return h_out

    @tf.function
    def get_kme_gram(self, domains=None):
        """

        Parameters
        ----------
        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        kme_gram_matrix : `tf.Tensor`
            KME-Gram matrix of the domains

        """
        if domains is None:
            domains = list(self.domain_basis.values())

        kme_gram_matrix = tf.map_fn(fn=lambda d_i: tf.map_fn(fn=lambda d_j: reduce_mean(self.kernel.matrix(d_i, d_j)), elems=stack(domains)), elems=stack(domains), parallel_iterations=10)

        return kme_gram_matrix

    @tf.function
    def get_kme_squared_norm(self, domains=None):
        """

        Parameters
        ----------
        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        squared_kme_norm : `tf.Tensor`
            returns the squared KME norms of each domain


        """
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme_norm = tf.map_fn(fn=lambda d_j: reduce_mean(self.kernel.matrix(d_j, d_j)), elems=stack(domains))

        return squared_kme_norm

    @tf.function
    def get_domain_prob(self, h, domains=None):
        """

        Parameters
        ----------
        h : `tf.Tensor`
            input of the layer
            
        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        domain_coefficients : `tf.Tensor`
            tensorf of coefficients for each sample of each domain

        """
        if self.similarity_measure == 'projected':
            domain_coefficients = self.get_projection_coef(h, domains=domains)
        elif self.similarity_measure == 'cosine_similarity':
            domain_coefficients = self.cosine_similarity_softmax(h, domains=domains)
        else:
            domain_coefficients = self.mmd_softmax(h, domains=domains)

        return domain_coefficients


    @tf.function
    def mmd_softmax(self, h, domains=None):
        """

        Parameters
        ----------
        h : `tf.Tensor`
            input of the layer
            
        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        domain_probability_mmd : `tf.Tensor`

        """
        if domains is None:
            domains = list(self.domain_basis.values())

        mmd = tf.map_fn(lambda d: (diag_part(self.kernel.matrix(h, h)) - 2 * reduce_mean(self.kernel.matrix(h, d), axis=1) + reduce_mean(self.kernel.matrix(d, d))), elems=stack(domains))

        domain_probability_mmd = transpose(softmax((-1) * self.softness_param * mmd, axis=0))

        return domain_probability_mmd

    @tf.function
    def cosine_similarity_softmax(self, h, domains=None):
        """

        Parameters
        ----------
        h : `tf.Tensor`
            input of the layer

        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        domain_probability_cosine_sim : `tf.Tensor`

        """
        if domains is None:
            domains = list(self.domain_basis.values())

        cosine_sim = tf.map_fn(lambda d: self.softness_param * reduce_mean(self.kernel.matrix(h, d), axis=1) / (sqrt(tf.linalg.diag_part(self.kernel.matrix(h, h))) * float(1 / self.N) * sqrt(reduce_sum(self.kernel.matrix(d, d)))), elems=stack(domains))

        domain_probability_cosine_sim = transpose(softmax(cosine_sim, axis=0))

        return domain_probability_cosine_sim

    @tf.function
    def get_projection_coef(self, h, domains=None):
        """

        Parameters
        ----------
        h : `tf.Tensor`
            input of the layer

        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        domain_probability_cosine_sim : `tf.Tensor`

        """
        if domains is None:
            domains = list(self.domain_basis.values())

        squared_kme = diag_part(self.get_kme_gram(domains=domains))

        alpha = squeeze(vectorized_map(lambda d: reduce_mean(self.kernel.matrix(h, d[0])/d[1], axis=-1, keepdims=True), elems=[stack(domains), squared_kme]), axis=-1)

        return transpose(alpha)


    #@tf.function
    def get_OLS_penalty(self, h, domains=None):
        """

        Parameters
        ----------
        h : `tf.Tensor`
            input of the layer

        domains : `list` [`tf.Tensor`]
             if domains are `None`, the method use the domain basis (Default value = None)

        Returns
        -------
        ols_penalty : `tf.Tensor`
            ols penalty, based on given domains and input h

        """
        domains = list(self.domain_basis.values())
        alpha_coefficients = transpose(self.get_domain_prob(h, domains))

        # (1)
        pen_1 = reduce_mean(diag_part(self.kernel.matrix(h, h)))

        # (2)
        pen_2 = reduce_mean(vectorized_map(lambda d:  multiply(d[1], reduce_mean(self.kernel.matrix(d[0], h), axis=0)), elems=[stack(domains), alpha_coefficients]))

        # (3)
        pen_3 = reduce_mean(reduce_sum(vectorized_map(lambda d_j: d_j[1] * reduce_sum(transpose(vectorized_map(lambda d_k: d_k[1] * reduce_mean(self.kernel.matrix(d_k[0], d_j[0])), elems=[stack(domains), alpha_coefficients])), axis=-1), elems=[stack(domains), alpha_coefficients]), axis=0))


        ols_penalty = sqrt(pen_1 + (-2) * pen_2 + pen_3)
        return ols_penalty

    @tf.function
    def get_domain_distributional_variance(self):
        """Computes the distributional variance of the domains basis

        Returns
        -------
        domain_distributional_variance : `tf.Tensor`
            distributional variance of the domain basis

        """
        domain_gram_matrix = self.get_kme_gram()
        domain_distributional_variance = reduce_mean(diag_part(domain_gram_matrix)) - reduce_mean(domain_gram_matrix)
        return domain_distributional_variance


    def get_orth_penalty(self):
        """
        Returns
        -------
        orth_pen_dict : `dict` {`string`: `float'}
            returns dictionary which includes the values of the orthogonality penalty functions for the domain basis

        """
        self.gram_matrix = gram_matrix = self.get_kme_gram()

        orth_pen_dict = {}

        self.orth_pen_srip = orth_pen_srip = (tf.linalg.svd(gram_matrix - diag(diag_part(gram_matrix)), compute_uv=False)[0])
        orth_pen_dict.update({"SRIP": orth_pen_srip.numpy()})

        self.orth_pen_so = orth_pen_so = tf.norm(gram_matrix - diag(diag_part(gram_matrix)), ord='fro', axis=(0, 1))
        orth_pen_dict.update({"SO": orth_pen_so.numpy()})

        self.orth_pen_mc = orth_pen_mc = tf.norm(gram_matrix - diag(diag_part(gram_matrix)), ord=np.inf, axis=(0, 1))
        orth_pen_dict.update({"MC": orth_pen_mc.numpy()})

        self.orth_pen_icp = orth_pen_icp = reduce_sum(gram_matrix - diag(diag_part(gram_matrix)))
        orth_pen_dict.update({"ICP": orth_pen_icp.numpy()})

        return orth_pen_dict


    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_domain_basis(self):
        return {domain: self.domain_basis[domain].numpy() for domain in self.domain_basis.keys()}


