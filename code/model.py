from sklearn.linear_model import LogisticRegression
from sklearn import naive_bayes
from sklearn.neighbors import NearestCentroid
from result_analysis import plot_result, tight_layout

def models(total_datasets, params_set, params):
    Logits = LogisticRegression()
    Logits.fit(total_datasets.train.x, total_datasets.train.y)
    # predict the labels on dev dataset
    predictions_Logits = Logits.predict(total_datasets.dev.x)
    # Show result
    plot_result("Logits - {}".format(params_set), total_datasets.dev.y, predictions_Logits, params.labels, params.train.log_dir)
    tight_layout(Logits, total_datasets.dev.x, total_datasets.dev.y, params.labels, params.train.fig_dir+"/{}-lg.png".format(params_set))

    Naive = naive_bayes.BernoulliNB()
    Naive.fit(total_datasets.train.x, total_datasets.train.y)
    # predict the labels on dev dataset
    predictions_NB = Naive.predict(total_datasets.dev.x)
    # Show result
    plot_result("Naive Bayes - {}".format(params_set), total_datasets.dev.y, predictions_NB, params.labels, params.train.log_dir)
    tight_layout(Naive, total_datasets.dev.x, total_datasets.dev.y, params.labels, params.train.fig_dir+"/{}-nb.png".format(params_set))

    clf = NearestCentroid()
    clf.fit(total_datasets.train.x, total_datasets.train.y)
    # predict the labels on dev dataset
    predictions_clf = clf.predict(total_datasets.dev.x)
    # Show result
    plot_result("Nearest Centroid - {}".format(params_set), total_datasets.dev.y, predictions_clf, params.labels, params.train.log_dir)
    tight_layout(clf, total_datasets.dev.x, total_datasets.dev.y, params.labels, params.train.fig_dir+"/{}-nc.png".format(params_set))