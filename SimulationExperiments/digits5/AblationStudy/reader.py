import glob as glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_e2e_accuracy(file_name):
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [el.decode("utf-8") for el in lines]
        return float(lines[-1].split()[8])


def get_ft_accuracy(file_name, method):
    counter = 0
    with open(file_name, 'rb') as f:
        lines = f.readlines()
        lines = [el.decode("utf-8") for el in lines]
        previous_line = None
        for line in lines:
            if method == 'cosine_similarity':
                if line.startswith('MMD'):
                    return float(previous_line.split()[8])
            elif method == 'MMD':
                if line.startswith('projected'):
                    return float(previous_line.split()[8])
            else:
                return float(lines[-1].split()[8])

            previous_line = line


if __name__ == '__main__':
    files = glob.glob('/wave/odin/results/2022-06-04_AblationStudyDigits5/*.out')
    results = []
    for file in files:
        experiment = file.split('_')[2]

        if experiment == 'domaindimfixed':
            M = file.split('_')[3]
            N = 10

            training = "E2E" if file.split('_')[-3] != "None" else "FT"

            if training == "E2E":
                similarity = 'cosine_similarity' if file.split('_')[4] == 'cosine' else file.split('_')[4].lower()
                accuracy = get_e2e_accuracy(file)
                results.append([experiment, M, N, similarity, accuracy, training])
            if training == "FT":
                for similarity in ['cosine_similarity', 'MMD', 'projected']:
                    accuracy = get_ft_accuracy(file, similarity)
                    results.append([experiment, M, N, similarity.lower(), accuracy, training])

        elif experiment == 'numdomainfixed':
            M = 5
            N = file.split('_')[3]
            training = "E2E" if file.split('_')[-3] != "None" else "FT"

            if training == "E2E":
                accuracy = get_e2e_accuracy(file)
                similarity = 'cosine_similarity' if file.split('_')[4] == 'cosine' else file.split('_')[4].lower()
                results.append([experiment, M, N, similarity, accuracy, training])
            if training == "FT":
                for similarity in ['cosine_similarity', 'MMD', 'projected']:
                    accuracy = get_ft_accuracy(file, similarity)
                    results.append([experiment, M, N, similarity.lower(), accuracy, training])


    titles = {'cosine_similarity': "CS", 'mmd': 'MMD', 'projected':'Projection'}

    results = pd.DataFrame(results, columns = ['experiment', 'M', 'N', 'similarity measure', 'Accuracy', 'Training'])
    fig, axs = plt.subplots(2, 3, figsize=(15,8), sharey=True)
    ax_x = 0
    for i in results['experiment'].unique():
        ax_y = 0

        for j in ['cosine_similarity', 'mmd', 'projected']:
            if ax_x == 0:
                axs[ax_x, ax_y].set_title(titles[j])

            df = results.loc[(results['experiment'] == i) & (results['similarity measure'] == j)]
            value_list = ["2" , "4" ,"5", "10"] if i == 'domaindimfixed' else ["2" , "4" ,"10", "20"]
            select = df.M.isin(value_list) if i == 'domaindimfixed' else df.N.isin(value_list)
            df = df[select]
            sns.boxplot(x="M" if i == 'domaindimfixed' else "N",
                        y="Accuracy", hue="Training", data = df,
                        hue_order =['FT', 'E2E'],
                        showfliers=False,
                        order = ["2" , "4" ,"5", "10"] if i == 'domaindimfixed' else ["2" , "4" ,"10", "20"],
                        ax = axs[ax_x, ax_y])
            axs[ax_x, ax_y].set_ylim([0.65, 0.72])
            axs[ax_x, ax_y].legend(loc = "upper right")

            ax_y+=1
        ax_x+=1

    #axs[0, 0].set(xlabel=None)
    #axs[0, 1].set(xlabel=None)
    #axs[0, 2].set(xlabel=None)


    axs[0, 1].set(ylabel=None)
    axs[0, 2].set(ylabel=None)
    axs[1, 1].set(ylabel=None)
    axs[1, 2].set(ylabel=None)


    fig.savefig('/wave/odin/results/figures/MN_ablation_study.eps', format='eps')
    fig.show()

    cluster_scores = {'Calinski-Harabasz Score': [25184.06569106759, 23976.427011731048, 23799.19328490193, 23119.45591922271,
                                                  24110.066161795414,25130.55223261555, 26240.41729274808, 27876.733074090112, 31333.912283869206 ],
                      'Davies Bouldin Score': [2.0650990639414495, 1.5307532489850681, 1.7260386978727866, 1.5011838866808553,
                                               1.5742864675994468, 1.4066158348647277, 1.3604824628341003, 1.3169936266749354, 1.077163340979712],
                      'Silhouette Scores': [0.25240794, 0.2607105, 0.19672757, 0.21483384, 0.225491, 0.25213462, 0.2721163,
                                            0.28520724, 0.3217156]
                      }


    x = range(2, 11)
    methods = results['similarity measure'].unique()
    for method in methods:
        df = results.loc[(results['experiment'] == 'domaindimfixed') & (results['similarity measure'] == method)]
        df["M"] = df["M"].astype(int)

        for key in cluster_scores.keys():

            df2 = pd.DataFrame(data=np.transpose([cluster_scores[key], x]).astype(float), columns = [key, 'M'])

            df = df.merge(df2,  on='M', how='right')

            fig, ax = plt.subplots( figsize=(10,5))

            sns.pointplot(x = "M", y= key, data=df, ax = ax)

            ax2 = ax.twinx()
            #ax2.set_xticks(range(2, 11))
            sns.boxplot(x=df["M"], y="Accuracy", hue="Training", data=df,
                        hue_order=['FT', 'E2E'],
                        showfliers=False, ax=ax2, boxprops=dict(alpha=.5))
            ax.set_title(method)
            ax.set_ylabel(key)
            ax2.set_ylabel('Accuracy')
            ax2.set_ylim([0.65, 0.74])
            fig.savefig('/wave/odin/results/figures/'+key+'_'+method+'.pdf', format='pdf')
            fig.show()

    df = results.loc[(results['experiment'] == 'domaindimfixed')]
    df["M"] = df["M"].astype(int)

    for key in cluster_scores.keys():
        df2 = pd.DataFrame(data=np.transpose([cluster_scores[key], x]).astype(float), columns=[key, 'M'])

        df = df.merge(df2, on='M', how='right')

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.pointplot(x="M", y=key, data=df, ax=ax)

        ax2 = ax.twinx()
        # ax2.set_xticks(range(2, 11))
        sns.boxplot(x=df["M"], y="Accuracy", hue="Training", data=df,
                    hue_order=['FT', 'E2E'],
                    showfliers=False, ax=ax2, boxprops=dict(alpha=.5))
        ax.set_title("All methods")
        ax.set_ylabel(key)
        ax2.set_ylabel('Accuracy')
        ax2.set_ylim([0.65, 0.74])
        fig.savefig('/wave/odin/results/figures/average.pdf', format='pdf')
        fig.show()

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.pointplot(x=cluster_scores['Calinski-Harabasz Score'], y=range(2,11), ax=ax)
    ax2 = ax.twinx()
    sns.pointplot(x=cluster_scores['Silhouette Scores'], y=range(2,11), ax=ax2)
    ax.set_title("All methods")
    ax.set_ylabel(key)
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0.65, 0.74])
    fig.savefig('/wave/odin/results/figures/cluster_score.pdf', format='pdf')
    fig.show()
