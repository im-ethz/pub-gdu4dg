import numpy as np
import pandas as pd
from numpy.linalg import norm
from matplotlib import cm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import LinearLocator
plt.style.use("seaborn")

import glob
import os
import array_to_latex as a2l









plot_file_dir = "/local/home/euernst/pub-gdu4dg/SimulationExperiments/toy_example/experiment_plots"
results_file_dir = "/local/home/euernst/pub-gdu4dg/SimulationExperiments/toy_example/results_toy_example/"

def plot_3d_kernel_function():
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_2d_kernel_function():


    r = np.linspace(-5, 5, num=2000)

    gaussian_kernel = lambda r, sigma: np.exp(-sigma * r ** 2)
    laplacian_kernel = lambda r, sigma: np.exp(-sigma * np.abs(r))

    kernel_function_dict ={
    "Gaussian kernel" : gaussian_kernel,
    "Laplacian Kernel" : laplacian_kernel
    }

    for key in kernel_function_dict.keys():
        for sigma in [10, 1, 0.5, 0.2]:
            kernel = kernel_function_dict[key]
            kernel(r, sigma)
            a = kernel(r, sigma)
            plt.plot(a, label="$\sigma={}$".format(sigma))
        plt.legend()
        plt.grid()
        plt.xlabel("$r = x - x'$")
        plt.title(key)
        #plt.ylabel('$t-SNE_{2}$')
        plt.show()
        plt.close()


    Ratio_quad_kernel = lambda r, c, beta: (r**2 + c)**(-beta)
    for beta in [1, 0.2]:
        for c in [10, 1, 0.2]:
            a = Ratio_quad_kernel(r, c, beta)
            plt.plot(a, label="$ Beta={beta}, c={c}$".format(beta=beta, c=c))
        plt.legend()
        plt.grid()
        plt.xlabel("$r = x - x'$")
        plt.title("Rational Quadratic Kernel")
        #plt.ylabel('$t-SNE_{2}$')
        plt.show()
        plt.close()


# "DOMAIN_VARIANCE", "binary_crossentropy", "val_binary_crossentropy",
def plot_results_toy_example(metrics=['binary_accuracy', 'binary_crossentropy',"DOMAIN_VARIANCE", 'val_binary_accuracy', 'val_binary_crossentropy',]):
    result_file_names = glob.glob(results_file_dir+"**result**")
    result_file_names.reverse()


    for metric in metrics:
        for result_file in result_file_names:
            file_path = os.path.join(results_file_dir, result_file)
            result_df = pd.read_csv(file_path, index_col=0)
            eval_columns = result_df.columns.tolist()

            if metric in eval_columns:
                method = result_df['method'].unique().tolist()[0]
                metric_values = result_df[metric]
                plt.plot(metric_values, label=method, linewidth=1.5, alpha=0.8)

        plt.legend()
        plt.title(metric.replace("_", " ").replace("val", "test").upper())

        model_res_dir = os.path.join(plot_file_dir, "model_res")
        create_dir_if_not_exists(model_res_dir)
        save_file_path = os.path.join(model_res_dir, "training_{metric}_{method}.jpg".format(method=method, metric=metric))
        plt.savefig(save_file_path, dpi=500)

        plt.show()
        plt.close()




def create_eval_tabel():
    results_file_dir = "/local/home/euernst/pub-gdu4dg/SimulationExperiments/toy_example/results_toy_example/"
    result_file_names = glob.glob(results_file_dir+"**result**")

    table_df = pd.DataFrame()

    table_values = ['loss',
                     'binary_accuracy',
                     'binary_crossentropy',
                     'val_loss',
                     'val_binary_accuracy',
                     'val_binary_crossentropy',
                     'DOMAIN_VARIANCE',
                     'MMD_TRAIN',
                     'MMD_TEST',
                     'SRIP',
                     'SO',
                     'MC',
                     'ICP',
                     'sigma_median'
                    ]

    methods_list = []
    for result_file in result_file_names:
        file_path = os.path.join(results_file_dir, result_file)
        result_df = pd.read_csv(file_path, index_col=0)

        for col in result_df.columns:
            try:
                result_df[col] = result_df[col].round(4).astype(float)
            except:
                pass

        method = result_df['method'].unique().tolist()[0]
        methods_list.append(method)
        eval_columns = result_df.columns.tolist()

        result_entries = [col for col in table_values if col in eval_columns]
        result_df = result_df[result_entries].iloc[-1]

        table_df = pd.concat([table_df, result_df], axis=1)

    table_df.columns = methods_list
    #table_df.fillna(0, inplace=True)


    index_list = table_df.index.tolist()
    new_index = [idx.replace("_", " ").replace("val", "test").lower() for idx in index_list]

    table_df.index = new_index

    latex_code = a2l.to_ltx(table_df,
                            frmt='{:6.3f}',
                            arraytype='tabular',
                            mathform=False,
                            print_out=True
                            )

def create_coefficient_tabel():
    results_file_dir = "/local/home/euernst/pub-gdu4dg/SimulationExperiments/toy_example/results_toy_example/"
    result_file_names = glob.glob(results_file_dir+"**result**")

    result_file_names = [file for file in result_file_names if "DG" in file]
    table_df = pd.DataFrame()


    table_values = ['DOMAIN_PROB_TRAIN', 'PROB_STD_TRAIN', 'DOMAIN_PROB_TEST', 'PROB_STD_TEST']

    methods_list = []
    for result_file in result_file_names:
        file_path = os.path.join(results_file_dir, result_file)
        result_df = pd.read_csv(file_path, index_col=0)

        for col in result_df.columns:
            try:
                result_df[col] = result_df[col].round(4).astype(float)
            except:
                pass

        method = result_df['method'].unique().tolist()[0]
        methods_list.append(method)
        eval_columns = result_df.columns.tolist()

        result_entries = [col for col in table_values if col in eval_columns]
        res_ent = result_df[result_entries].iloc[-1].tolist()
        res_ent = [res.replace("  ", " ").replace("[", "").replace("]", "").split(" ") for res in res_ent]
        res_ent = [[elmt for elmt in elmt_list if elmt not in ['', ' ']] for elmt_list in res_ent]


        res_end_df = pd.DataFrame(res_ent).astype(float)

        res_end_df.columns = [r"$\beta_{}$".format(j+1) for j in range(len(res_end_df.transpose()))]
        res_end_df.index = [ent.replace("TRAIN", "SOURCE").replace("TEST", "TARGET").replace("_", " ") for ent in result_entries]

        table_df = pd.concat([table_df, res_end_df], axis=0)


    latex_code = a2l.to_ltx(table_df,
                            frmt='{:6.3f}',
                            arraytype='tabular',
                            mathform=False,
                            print_out=True
                            )



def create_mmd_heat_plot():
    mmd_mat_file_names = glob.glob(results_file_dir+"**mmd_matrix**")

    for file in mmd_mat_file_names:
        file_path = os.path.join(results_file_dir, file)
        mmd_mat = pd.read_csv(file_path, index_col=0)
        #mmd_mat.sort_index()
        method = file_path.split("/")[-1].split("mmd_matrix_")[-1].split(".csv")[0]
        mmd_cols = mmd_mat.columns.tolist()

        new_cols = []
        for i in range(len(mmd_cols)):
            col = mmd_cols[i]
            if "V" in col:
                new_cols.append(r'$\mu_{V_{' + str(i+1)  + '}}$')
            if "x" in col:
                new_cols.append(r'$\mu_{\mathbb{P}_' + col[-1] + '}$')



        mmd_mat.columns = new_cols
        mmd_mat.index = new_cols

        mask = np.triu(mmd_mat)

        heat_map = sns.heatmap(mmd_mat, annot=True, cbar_kws={'label': '$MMD$'}, cmap='Reds_r',
                               #linewidths=1, linecolor='black',
                                mask=mask
                               )
        heat_map.set_yticklabels(heat_map.get_yticklabels(), rotation=0, fontsize=15)
        heat_map.set_xticklabels(heat_map.get_yticklabels(),  fontsize=15)

        plt.title("MMD Matrix: {}".format(method.replace("_", " ")))

        mmd_heatmap_dir = os.path.join(plot_file_dir, "MMD_heatmaps")
        create_dir_if_not_exists(mmd_heatmap_dir)
        save_file_path = os.path.join(mmd_heatmap_dir, "MMD_heatmap_{}.jpg".format(method) )
        plt.savefig(save_file_path, dpi=500)

        plt.show()
        plt.close()







def create_dir_if_not_exists(dir_path):
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        print("\n \n CREATED DIRECTORY: {}".format(dir_path))




if __name__ == "__main__":
    create_mmd_heat_plot()
    plot_results_toy_example()
    #plot_2d_kernel_function()

