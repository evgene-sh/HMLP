from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns


def learning_results(real_data, predicted_data, labels: list[str]):
    cm = confusion_matrix(real_data, predicted_data, labels=labels)

    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(1,1,1)
    sns.heatmap(cm, annot=True, cmap="Blues", ax=ax, fmt='g');

    ax.set_xlabel('Predicted labels');
    ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels);
    plt.setp(ax.get_yticklabels(), rotation=30, horizontalalignment='right')
    plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')     
    plt.show()

    print(classification_report(real_data, predicted_data, labels=labels, zero_division=0))


def vectorization_results(matrix, labels):
    tsne_results = TSNE(n_components=2, init='random', random_state=0, learning_rate=200.0).fit_transform(matrix)

    plt.figure(figsize=(16,10))
    palette = sns.hls_palette(17, l=.6, s=.9)
    sns.scatterplot(
        x=tsne_results[:,0], 
        y=tsne_results[:,1],
        hue=labels,
        palette=palette,
        legend="full",
        alpha=0.3
    )
    plt.show()

