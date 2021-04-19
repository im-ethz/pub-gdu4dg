import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('seaborn')

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection import train_test_split

from sklearn.manifold import TSNE
from ModelEvaluation.classifier import model_evaluation
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.decomposition import PCA, KernelPCA


cmaps = ['spring_r', 'winter_r', 'YlGn', 'RdPu', 'magma', 'PuBu',  'Blues',    'BuGn',
         'BuPu', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r',
         'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1',
         'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r',
         'PuRd',]


markers = ['^', 'x', 'o', '*', '+', 's', '4', 'd', "p", "h", "D"]

figure_height = 12
figure_width = 12



def plot_time_series_prediction(model, test_data, columns, figsize=(18, 18), plot_accuracy=False, labels=None, dpi=150):
    num_plots = test_data.shape[2]
    random_segment_index = random.randint(1, test_data.shape[0])
    random_test_segment = test_data[random_segment_index-1:random_segment_index, ]
    test_segment_prediction = model.predict(random_test_segment)

    if plot_accuracy:
        num_plots += 1
        embeddings = model.encode(test_data)
        accuracy_df = model_evaluation(embeddings, labels, model=model)

    if num_plots > 1 and plot_accuracy:
        fig, ax = plt.subplots(num_plots, 1, figsize=figsize, dpi=dpi)
        for i in range(num_plots-1):
            ax[i].plot(random_test_segment[0, :, i], label="GT")
            ax[i].plot(test_segment_prediction[0, :, i], label="predicted")
            ax[i].set_title(columns[i])
            ax[i].legend()
            ax[i].grid()

        ax[num_plots-1].bar(list(range(len(accuracy_df))), list(accuracy_df.accuracy))
        for patch in ax[num_plots - 1].patches:
            ax[num_plots - 1].text(patch.get_x() + 0.26, patch.get_height()/2, str(round(patch.get_height(), 3)), color='white', fontsize=10)

        ax[num_plots-1].set_xticks(list(range(len(accuracy_df))))
        ax[num_plots-1].set_ylim(0, 1)
        ax[num_plots-1].set_xticklabels(accuracy_df.classifier)
        ax[num_plots-1].grid(which='minor')
        ax[num_plots-1].set_title('Accuracy')

    elif num_plots > 1 and not plot_accuracy:
        fig, ax = plt.subplots(num_plots, 1, figsize=figsize, dpi=dpi)
        for i in range(num_plots):
            ax[i].plot(random_test_segment[0, :, i], label="GT")
            ax[i].plot(test_segment_prediction[0, :, i], label="predicted")
            ax[i].set_title(columns[i])
            ax[i].legend()
            ax[i].grid()

    else:

        fig, ax = plt.subplots(num_plots, figsize=(18, 10), dpi=100)
        ax.plot(random_test_segment[0, : ], label="GT")
        ax.plot(test_segment_prediction[0, :], label="predicted")
        ax.set_title('Time Series Encoding:' + columns[0])
        ax.legend()
        ax.grid()

    return ax


def plot_model_evaluation(embeddings, labels):
    accuracy_df = model_evaluation(embeddings, labels)

    fig, ax = plt.subplots()
    ax.bar(list(range(len(accuracy_df))), list(accuracy_df.accuracy))
    ax.set_xticks(list(range(len(accuracy_df))))
    ax.set_ylim(0,1)
    ax.set_xticklabels(accuracy_df.classifier)
    ax.grid(which='minor')
    ax.set_title('Accuracy')
    return ax


def plot_feature_error(model, test_data, columns):
    test_prediction = model.predict(test_data.copy().values)
    test_data.std()
    abs_error = pd.DataFrame(np.abs(test_data - test_prediction), columns=columns)
    error_mean = abs_error.mean()
    num_features = list(range(len(error_mean)))
    fig, ax = plt.subplots(1, figsize=(18, 18))
    ax.bar(num_features, height=error_mean.values, alpha=0.5, align='center')
    plt.xticks(num_features, columns, rotation=90, fontsize=13)
    ax.set_title("Feature Error")
    ax.set_ylabel('error')

    return ax




def colors_from_values(values, palette_name):
    # normalize the values to range [0, 1]
    normalized = (values - min(values)) / (max(values) - min(values))
    # convert to indices
    indices = np.round(normalized * (len(values) - 1)).astype(np.int32)
    # use the indices to get the colors
    palette = sns.color_palette(palette_name, len(values))
    return np.array(palette).take(indices, axis=0)


def plot_TSNE(data_X, data_Y, title="$t-SNE$", labels=None, plot_kde=False, file_path=None, show_plot=True):

    if labels is None:
        try:
            labels = data_Y.ravel()
        except:
            labels = data_Y

    tsne = TSNE(n_components=2, random_state=0)
    tsne_obj = tsne.fit_transform(data_X)
    tsne_df = pd.DataFrame({'X': tsne_obj[:, 0],
                            'Y': tsne_obj[:, 1],
                            'label': labels})

    if plot_kde:
        make_KDE(tsne_df, title=title)

    else:
        tsne_plot = sns.scatterplot(x="X", y="Y", palette="coolwarm", hue='label', legend='full',
                                    marker=".", data=tsne_df, edgecolor='black'
                                    )

        plt.xlabel('$t-SNE_{1}$')
        plt.ylabel('$t-SNE_{2}$')
        tsne_plot.set_title(title)

        if file_path:
            plt.savefig(file_path, bbox_inches='tight')
            print("\n\n TSNE-PLOT SAVED IN: {}".format(file_path))

        if show_plot:
            plt.show()

        plt.close()


def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()



def plot_roc(y_pred, y_true):
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_true, y_pred)
    AUC = roc_auc_score(y_pred=y_pred, y_true=y_true, average='macro')
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC(area = {:.3f})'.format(AUC))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



def plot_PCA(data_X, y_true, plot_kde=False, title="$PCA$"):
    pca_mod = PCA(n_components=2)

    pca_components = pca_mod.fit_transform(data_X)

    PCA_df = pd.DataFrame(data=pca_components, columns=['PCA-component_{1}', 'PCA-component_{2}'])

    pca_col_1, pca_col_2  = PCA_df.columns.tolist()

    PCA_df['label'] = y_true

    plt.figure()
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    if plot_kde:
        make_KDE(PCA_df, title=title)

    else:

        colors = [main_color(cmaps[label]) for label in range(len(np.unique(y_true)))]

        pca_plot = sns.scatterplot(x=pca_col_1, y=pca_col_2, palette=colors, hue='label', legend='full', data=PCA_df)

        plt.xlabel('$PCA-component_{1}$')
        plt.ylabel('$PCA-component_{2}$')
        plt.title(title)

        plt.legend()
        plt.grid()

        plt.xlim(min(PCA_df[pca_col_1].values) - 0.5, max(PCA_df[pca_col_1].values) + 0.5)
        plt.ylim(min(PCA_df[pca_col_2].values) - 0.5, max(PCA_df[pca_col_2].values) + 0.5)

        plt.gcf().set_size_inches(figure_height, figure_width)
        plt.show()

def plot_calibration_curve(name, fig_index, y_pred, y_true, threshold=0.5, bins=10):


    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    num_classes = y_true.shape[-1]

    for i in range(num_classes):
        y_treu_class = y_true[:, i]
        y_pred_class = y_pred[:, i]

        fraction_of_positives, mean_predicted_value = calibration_curve(y_treu_class, y_pred_class, n_bins=bins)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="label " + str(i+1))

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (reliability curve)')
        ax1.grid()

        ax2.hist(y_pred_class, range=(0, 1), bins=bins, label="emp. prob. label " + str(i+1), histtype="step", lw=2)

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)
        ax2.grid()
        plt.tight_layout()

    plt.sho


def make_KDE(data, target_col='label', columns=None, title='KDE Plot', rug_plot=True, kernel='gau'):

    df_columns = columns if columns is not None else data.drop([target_col], axis=1).columns.tolist()
    target_label_list = data[target_col].unique().tolist()

    f, ax = plt.subplots()
    for k in range(len(target_label_list)):
        target_label = target_label_list[k]
        target_data = data[data[target_col] == target_label_list[k]]
        x_data = target_data[df_columns[0]].values
        y_data = target_data[df_columns[1]].values
        sns.kdeplot(x_data, y_data, label=str(target_label), cmap=cmaps[k],
                    shade=True, shade_lowest=False, alpha=0.5,
                    kernel=kernel, legend='full')
        if rug_plot:
            sns.rugplot(x_data, axis='x',  color=sns.color_palette(cmaps[k])[-1], height=0.02, alpha=0.2)
            sns.rugplot(y_data, axis='y',  color=sns.color_palette(cmaps[k])[-1], height=0.02, alpha=0.2)

    plt.title(title)
    ax.legend()

    plt.xlabel("$"+df_columns[0]+"$")
    plt.ylabel("$"+df_columns[1]+"$")

    plt.xlim(min(data[df_columns[0]].values) - 0.5, max(data[df_columns[0]].values) + 0.5)
    plt.ylim(min(data[df_columns[1]].values) - 0.5, max(data[df_columns[1]].values) + 0.5)
    plt.gcf().set_size_inches(figure_height, figure_width)
    plt.show()


def main_color(cmap, format='hex'):
    if format=='hex':
        return rgb2hex(sns.color_palette(cmap)[-1])
    else:
        return sns.color_palette(cmap)[-1]



# TODO: clean this part here
def plot_KPCA(x_data, labels, title="KPCA"):

    sigma = get_kernel_width(x_data)

    kpca = KernelPCA(kernel="linear", fit_inverse_transform=True, gamma=(1/sigma**2), n_components=2, degree=3)
    X_kpca = kpca.fit_transform(x_data)
    X_back = kpca.inverse_transform(X_kpca)

    KPCA_df = pd.DataFrame(data=X_kpca, columns=['KPCA-component_{1}', 'KPCA-component_{2}'])

    kpca_col_1, kpca_col_2 = KPCA_df.columns.tolist()

    KPCA_df['label'] = labels

    plt.figure()
    plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')

    colors = [sns.color_palette() * 4 for label in range(len(np.unique(y)))]
    kpca_plot = sns.scatterplot(x=kpca_col_1, y=kpca_col_2, palette='winter_r',
                                hue='label', legend='full',
                                data=KPCA_df)

    plt.xlabel(str(kpca_col_1))
    plt.ylabel(kpca_col_2)
    plt.grid()
    plt.title(title)
    plt.show()


def rgb2hex(rgb_trple):

    if max(rgb_trple) <= 1:
        r = int(rgb_trple[0]*255)
        g = int(rgb_trple[1]*255)
        b = int(rgb_trple[2]*255)

    else:
        r = rgb_trple[0]
        g = rgb_trple[1]
        b = rgb_trple[2]

    return "#{:02x}{:02x}{:02x}".format(r, g, b)




def plot_LOSO_results():

    if False:

        sample1 = np.random.normal(-50, 5, 2000)
        sample2 = np.random.normal(-45, 7, 2000)
        #sample3 = np.random.normal(170, 20, 2000)

        sns.distplot(sample1, label=r"$P_{1}$")
        sns.distplot(sample2, label=r"$P_{2}$")
        #sns.distplot(sample3, label=r"$P_{3}$")
        plt.legend()
        plt.show()


    loso_res = pd.read_csv('/local/home/euernst/mt-eugen-ernst/loso_wesad.csv')

    val_column = "epoch_AUC_FT.1"
    subject_cols = 'val_subjects'

    #subject_columns = loso_res['val_subjects'].astype(int)

    auc_scores = loso_res[val_column]

    x = list(range(len(auc_scores)))

    plt.figure(figsize=(8, 8))
    #plt.scatter(x=loso_res['val_subjects'].values, y=loso_res[val_column].values, marker="^", label='AUC')
    plt.scatter(x=x, y=loso_res[val_column].values, marker="x", c='r', label='AUC')
    plt.hlines(np.mean(loso_res[val_column]), xmin=x[0], xmax=x[-1], label='avg_AUC {}'. format(np.round(np.mean(loso_res[val_column]), 3)))
    #plt.xticks(loso_res[subject_columns].values)
    #plt.ylim([0, 1])
    plt.legend()
    plt.title('AUC Scores: WESAD')
    plt.show()

    print(' ihszvfl')



def imp_sampling_plot():
    eval_df = pd.read_csv("/local/home/euernst/mt-eugen-ernst/H2Vec/domain_adaptation_results/imp_test_imp_2021-02-11 23:07.csv", index_col=0)

    top10 = eval_df.groupby(['METHOD']).sum()['AUC'].sort_values(ascending=False).index[:10].tolist()

    subjects = eval_df.index.unique().tolist()
    eval_df.columns

    eval_col = 'METHOD'

    methods = eval_df[eval_col].unique().tolist()
    methods = top10
    for method in methods:
        available_subjects = eval_df[eval_df[eval_col]==method].index.unique().tolist()
        #subjects = list(set(available_subjects) & set(subjects))

    plt.figure(figsize=(figure_height, figure_height))
    for i in range(len(methods)):
        method = methods[i]
        try:
            marker = markers[i]
        except:
            marker = '.'
        color = cmaps[i]
        subject_data = eval_df[(eval_df[eval_col] == method) & (eval_df.index.isin(subjects))]

        try:
            method = str(method) + '_' + str(subject_data['TRANSFOMATION'].iloc[0])+'_' +str(subject_data['KERNEL'].iloc[0])+' '+str(subject_data['similarity_measure'].iloc[0])
        except:
            method = str(method) + '_' + str(subject_data['TRANSFOMATION'].iloc[0])


        plt.scatter(x=subject_data.index.tolist(), y=subject_data["AUC"].values, color=sns.color_palette(color)[-1], marker=marker, label=method)
        plt.hlines(np.mean(subject_data['AUC']), xmin=min(subjects), xmax=max(subjects), color=sns.color_palette(color)[-1], label='AVG_' + str(method) + '_'+str(np.round(np.mean(subject_data['AUC']), 3))
                   )
    plt.legend()
    plt.title('Domain Adaption')
    #plt.ylim([0, 1])
    plt.show()





if __name__ == "__main__":


    imp_sampling_plot()
    pass


