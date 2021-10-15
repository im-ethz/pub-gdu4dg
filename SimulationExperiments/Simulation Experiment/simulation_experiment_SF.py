from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import os


import numpy as np
import pandas as pd
import tensorflow as tf
import array_to_latex as a2l
import matplotlib.pyplot as plt
plt.style.use("seaborn")
import tensorflow_probability as tfp


from datetime import datetime
from numpy.random import multivariate_normal
from sklearn.datasets import make_spd_matrix
from tensorflow.python.keras.models import Sequential
from sklearn.metrics.pairwise import euclidean_distances

# domain adaptation modules
from Model.DomainAdaptation.domain_adaptation_layer import DGLayer
from tensorflow.python.keras.layers import Dense, BatchNormalization
from Model.DomainAdaptation.DomainAdaptationModel import DomainAdaptationModel
from tensorflow_probability.python.math.psd_kernels.positive_semidefinite_kernel import _ProductKernel, _SumKernel
from Model.DomainAdaptation.domain_adaptation_callback import DomainCallback, DomainRegularizationCallback, FreezeFeatureExtractor


def init_gpu(gpu, memory):
    used_gpu = gpu
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.experimental.set_visible_devices(gpus[used_gpu], 'GPU')
            tf.config.experimental.set_virtual_device_configuration(gpus[used_gpu], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
        except RuntimeError as e:
            print(e)

init_gpu(gpu=1, memory=10000)


dir_path = "/local/home/sfoell/NeurIPS/results/Simulation_Experiment"
plot_file_dir = "/local/home/sfoell/NeurIPS/results/Simulation_Experiment"
results_dir = "/local/home/sfoell/NeurIPS/results/Simulation_Experiment"


def mult_norm_noise(sample_size, noise_level=0.5):
    return np.random.multivariate_normal(mean=np.zeros(shape=(3,)), cov=np.eye(3)*noise_level, size=sample_size).astype(np.double)


def toy_example_classification(sample_size=1000, source_domians=[1, 2, 3, 4], target_domain=5, save_results=True):

    target_index = target_domain-1

    ######################
    ####  domain 1    ####
    ######################
    mu_d1 = [-1.5, 5.2, -8.8]

    cov_d1 = [[1.0, 0.5, -0.2],
              [0.5, 5.0, -0.1],
              [-0.2, -0.1, 5.0]]

    ######################
    ####  domain 2    ####
    ######################
    mu_d2 = [3.5, 4.8, 4.2]

    cov_d2 = [[2.0, 0.2, 1.0],
              [0.2, 1.8, 0.3],
              [1.0, 0.3, 2.0]]

    ######################
    ####  domain 3    ####
    ######################
    mu_d3 = [3.5, -8.8, 10.2 ]

    cov_d3 = [[2.0, 0.2, 1.0],
              [0.2, 1.8, 0.3],
              [1.0, 0.3, 2.0]]

    ######################
    ####  domain  4   ####
    ######################
    mu_d4 = [2.5, 4.9, 3.2 ]

    cov_d4 = [[1.0, 0.0, 0.1],
              [0.0, 2.0, 0.2],
              [0.1, 0.2, 1.0]]

    ######################
    ####  domain  5   ####
    ######################
    mu_d5 = [4.5, 5.5, 5.2]

    cov_d5 = [[1.0, 0.0, 0.1],
              [0.0, 2.0, 0.2],
              [0.1, 0.2, 1.0]]

    parameter_list = [(mu_d1, cov_d1),
                       (mu_d2, cov_d2),
                       (mu_d3, cov_d3),
                       (mu_d4, cov_d4),
                       (mu_d5, cov_d5)
                      ]

    source_parameter_list = [parameter_list[i-1] for i in source_domians]

    #############################################
    ###      FEATURE DATA GENERATION          ###
    #############################################

    # source domain features
    x_data = [multivariate_normal(mu, cov, sample_size) +
              mult_norm_noise(sample_size) for (mu, cov) in source_parameter_list]

    x_threshold = [np.mean(x) for x in x_data]
    print(x_threshold)
    x_var = [np.var(x) for x in x_data]

    # target feature data
    # target_parameter_list = [parameter_list[target_domain - 1]]
    mu_target, cov_target = parameter_list[target_index]
    x_te = multivariate_normal(mu_target, cov_target, sample_size).astype(np.float32)
    #############################################
    ###         CAUSALITY FUNCTION            ###
    #############################################
    f_1 = lambda x: np.mean(x, axis=1) < np.mean(x)
    f_2 = lambda x: np.mean(x, axis=1) < np.mean(x)
    f_3 = lambda x: np.mean(x, axis=1) < np.mean(x)
    f_4 = lambda x: np.mean(x, axis=1) < np.mean(x)
    f_5 = lambda x: np.mean(x, axis=1) < np.mean(x)


    causality_funtion_list = [f_1, f_2, f_3, f_4, f_5]

    #############################################
    ###        LABEL DATA GENERATION          ###
    #############################################

    # source domain labels
    y_data = [causality_funtion_list[d](x_data[d]).astype(np.int) for d in range(len(x_data))]
    y_true_false_proportion = [[np.mean(y), 1 - np.mean(y)] for y in y_data]

    y_te = causality_funtion_list[target_index](x_te).astype(np.int)

    #############################################
    ###         PREPARE TRAINING              ###
    #############################################

    # merge source domains
    x_tr = np.concatenate([x_data[i] for i in range(len(x_data))], axis=0).astype(np.float32)

    y_tr = np.concatenate([y_data[i] for i in range(len(x_data))], axis=0).astype(np.float32)

    plot_scatter = True
    if plot_scatter:
        markers = ['*', 'v', 's', "p", "h", "D", 'd', "p"]
        n = 100
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#f2fffe")
        for k in range(len(x_data)):
            m = markers[k]
            x_source = x_data[k]
            if True:

                y_source = y_data[k]
                x_true = x_source[y_source==1]
                x_False = x_source[y_source==0]

                ax.scatter(x_true[:n, 0], x_true[:n, 1], x_true[:n, 2], c='r', alpha=0.5,
                           linewidth=1.5,
                           marker=m,
                           #label='$X_{}: y=0$'.format(k+1)
                           )

                ax.scatter(x_False[:n, 0], x_False[:n, 1], x_False[:n, 2],  c='b',
                           alpha=0.5,
                           linewidth=1,
                           marker=m,
                           #label='$X_{}: y=0$'.format(k+1)
                           )

        if save_results:
            plot_file_path = os.path.join(plot_file_dir, "3d_scatter_Labeled.eps")
            plt.savefig(plot_file_path, format="eps")

        plt.show()
        plt.close()


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor("#f2fffe")
        domain_colors = ['aqua', 'tab:green','deeppink', 'royalblue', 'goldenrod']


        for k in range(len(x_data)):
            m = markers[k]
            x_source = x_data[k]
            dom_c = domain_colors[k]
            ax.scatter(x_source[:n, 0], x_source[:n, 1], x_source[:n, 2],
                       c=dom_c, alpha=0.6,  marker=m, label='$X_{}$'.format(k+1))

            ax.legend(bbox_to_anchor=(1, 1), loc='upper left', borderaxespad=0.1)

        plot_file_path = os.path.join(plot_file_dir, "3d_scatter.eps")
        plt.savefig(plot_file_path,  format="eps")

        plt.show()
        plt.close()

    print(y_true_false_proportion)
    print(x_threshold)
    convert_to_latex = False
    if convert_to_latex:

        mu_source_list = [mu_d1, mu_d2, mu_d3, mu_d4, mu_d5]
        cov_source_list = [cov_d1, cov_d2, cov_d3, cov_d4, cov_d5]

        for mu in mu_source_list:
            a2l.to_ltx(pd.DataFrame(mu), frmt='{:6.1f}', arraytype='pmatrix', print_out=True)

        for cov in cov_source_list:
            a2l.to_ltx(pd.DataFrame(cov), frmt='{:6.1f}', arraytype='pmatrix', print_out=True)


        simulation_df = pd.DataFrame(y_true_false_proportion, columns=["$Y=1$", "$Y=0$"])
        simulation_df = simulation_df.join(pd.DataFrame(x_threshold, columns=["bar{x}"]))
        simulation_df = simulation_df.join(pd.DataFrame(x_var, columns=["$S^{2}$"]))

        simulation_df.index = ["$X_{}$".format(i) for i in range(1, len(x_data) + 1)]

        sim_latex = a2l.to_ltx(simulation_df,
                                frmt='{:6.3f}',
                                arraytype='tabular',
                                mathform=False,
                                print_out=True
                                )

    ##################################
    ###         TRAINING          ####
    ##################################
    # training parameter
    epochs = 300
    lr = 1e-3
    batch_size = 128

    ##################################
    ###       BASELINE MODEL      ####
    ##################################
    baseline = True
    if baseline:
        activation = "tanh"
        bias = True
        baseline_model = Sequential([
            Dense(9, activation=activation, use_bias=bias),
            #Dense(9, activation=activation, use_bias=bias),
            #Dense(3, activation=activation, use_bias=bias),
            Dense(1, activation='sigmoid')
        ])

        baseline_model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                               loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.BinaryCrossentropy(from_logits=False)])

        baseline_training_start = datetime.now()
        hist = baseline_model.fit(x_tr, y_tr, epochs=epochs,
                                            batch_size=batch_size,
                                            validation_data=(x_te, y_te),
                                            shuffle=True,
                                            verbose=2)

        baseline_training_start_duration = str(datetime.now() - baseline_training_start)


        if save_results:
            # creating evaluation dataframe
            eval_df_baseline = pd.DataFrame(hist.history.values(), index=hist.history.keys()).transpose()
            eval_df_baseline['duration'] = baseline_training_start_duration
            eval_df_baseline['method'] = "baseline"

            eval_df_file_path = os.path.join(results_dir, "baseline_results.csv")
            eval_df_baseline.to_csv(eval_df_file_path)


    ##################################
    ###          DG MODEL         ####
    ##################################
    for method in ['mmd', 'cosine_similarity', 'projection']:
    #for method in ['projection']:
        feature_extractor = Sequential([
        Dense(9, activation=activation, use_bias=bias),
        # Dense(3, activation=activation, use_bias=bias),
        #Dense(9, activation=activation, use_bias=bias),
        #Dense(1, activation='sigmoid')
    ])
        #num_domains = len(source_domians)
        num_domains = 8
        sigma_median = np.median(euclidean_distances(x_tr, x_tr))
        print(sigma_median)
        #kernel = tfp.math.psd_kernels.ExponentiatedQuadratic(length_scale=sigma_median, amplitude=2, feature_ndims=1)

        prediction_layer = Sequential()

        prediction_layer.add(DGLayer(domain_units=num_domains,
                                     N=30,
                                     softness_param=5,
                                     units=1,
                                     #kernel=kernel,
                                     sigma=sigma_median,
                                     activation="sigmoid",
                                     bias=False,
                                     similarity_measure=method,
                                     orth_reg_method='SRIP',
                                     lambda_orth=1e-4,
                                     lambda_OLS=1e-3,
                                     lambda_sparse=1e-3
                                     ))

        model = DomainAdaptationModel(feature_extractor=feature_extractor,
                                      prediction_layer=prediction_layer)


        model.build(input_shape=x_tr.shape)

        model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                      loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.BinaryAccuracy(),
                               tf.keras.metrics.BinaryCrossentropy(from_logits=False)])

        domain_callback = DomainCallback(test_data=x_te, train_data=x_tr, print_res=True, max_sample_size=500)
        domain_reg_cb = [domain_callback]

        start = datetime.now()
        hist = model.fit(x_tr, y_tr, epochs=epochs, batch_size=batch_size, validation_data=(x_te, y_te),
                                     shuffle=True, verbose=0,
                                     callbacks=domain_reg_cb)
        duration = str(datetime.now() - start)

        if save_results:
            # creating evaluation dataframe
            eval_df = pd.DataFrame(hist.history.values(), index=hist.history.keys()).transpose()
            eval_df['duration'] = duration
            eval_df['method'] = method
            eval_df['sigma_median'] = sigma_median
            eval_df_file_path = os.path.join(results_dir, "DG_{}_results.csv".format(method))
            eval_df.to_csv(eval_df_file_path)

        ##################################
        ###         MMD MATRIX        ####
        ##################################
        dg_layer = model.get_dg_layer()


        if save_results:
            model_param_list = ['domain_units', 'N', 'softness_param', 'units', 'sigma', 'activation', 'bias',
                                'similarity_measure', 'orth_reg_method', 'lambda_orth', 'lambda_OLS', 'lambda_sparse']
            dg_layer_params_dict = dg_layer.__dict__
            dg_layer_params = [dg_layer_params_dict[param] for param in model_param_list]
            dg_layer_params_df = pd.DataFrame(dg_layer_params).transpose()
            dg_layer_params_df.columns = model_param_list

            dg_layer_params_file_path = os.path.join(results_dir, "dg_layer_params_{}.csv".format(method))
            dg_layer_params_df.to_csv(dg_layer_params_file_path)

        x_df = np.concatenate((x_tr, x_te))
        x_index = np.array(["domain_1"] * 1000)
        for domain in ["domain_2", "domain_3", "domain_4", "domain_5"]:
            x_index = np.concatenate((x_index, np.array([domain] * 1000)))

        x_df = pd.DataFrame(x_data)
        x_df['index'] = x_index

        domain_basis = dg_layer.get_domain_basis()
        domain_basis.update({"x_{}".format(i+1): x_data[i] for i in range(len(x_data))})

        domains_values = list(domain_basis.values())
        doamain_keys = list(domain_basis.keys())

        kernel = dg_layer.kernel

        kme_matrix = get_kme_matrix(domains_values, kernel=kernel)
        kme_matrix_df = pd.DataFrame(kme_matrix, columns=doamain_keys, index=doamain_keys)
        print(kme_matrix_df)

        mmd_matrix = get_mmd_matrix(domains_values, kernel=kernel)
        mmd_matrix_df = pd.DataFrame(mmd_matrix, columns=doamain_keys, index=doamain_keys)

        print(mmd_matrix_df)

        if save_results:
            kme_matrix_file_path = os.path.join(results_dir, "kme_matrix_{}.csv".format(method))
            kme_matrix_df.to_csv(kme_matrix_file_path)

            mmd_matrix_file_path = os.path.join(results_dir, "mmd_matrix_{}.csv".format(method))
            mmd_matrix_df.to_csv(mmd_matrix_file_path)





def random_sample_dist(sample_size=5000, dim=3, mean_magnitude=1, cov_magnitude=1):
    random_cov_matrix = cov_magnitude * make_spd_matrix(dim)
    random_mean = mean_magnitude * np.random.randn(dim)
    random_sample = np.random.multivariate_normal(mean=random_mean,
                                                  cov=random_cov_matrix,
                                                  size=sample_size).astype(np.double)

    return [random_sample, random_mean, random_cov_matrix]



def sigma_median(x_data):
    sigma_median = np.median(euclidean_distances(x_data, x_data))
    return sigma_median

def MMD(x1, x2, kernel):
    return np.mean(kernel.matrix(x1, x1)) - 2 * np.mean(kernel.matrix(x1, x2)) + np.mean(kernel.matrix(x2, x2))


def get_kme_matrix(x_data, kernel):
    num_ds = len(x_data) if type(x_data) == list else 1
    kme_matrix = np.zeros((num_ds, num_ds))
    for i in range(num_ds):
        x_i = x_data[i]
        for j in range(i, num_ds):
            x_j = x_data[j]
            kme_matrix[i, j] = kme_matrix[j, i] = np.mean(kernel.matrix(x_i, x_j))

    return kme_matrix

def get_mmd_matrix(x_data, kernel):
    num_ds = len(x_data) if type(x_data) == list else 1
    mmd_matrix = np.zeros((num_ds, num_ds))
    for i in range(num_ds):
        x_i = x_data[i]
        for j in range(num_ds):
            x_j = x_data[j]
            mmd_matrix[i, j] = mmd_matrix[j, i] = MMD(x_i, x_j, kernel=kernel)

    return mmd_matrix



if __name__ == "__main__":
    toy_example_classification()




