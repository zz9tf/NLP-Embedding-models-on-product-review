import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from sklearn import naive_bayes
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_params, load_data, processing
from result_analysis import plot_result, tight_layout

# Load parameters
params = load_params("configs.yml")

# Load data
df = None
# if not os.path.exists(params.preprocessing.processed_data_dir):
df = load_data(params)

log = open(params.train.log_dir + "/log.txt", "w")
log.close()

# preprocessing data
for vector_set in params.preprocessing.verctorizers:
    for mf in params.preprocessing.max_feature:
        for ngram in params.preprocessing.ngram_range:
            total_datasets = processing(
                df=df,
                params=params,
                vector_set=vector_set,
                max_feature=mf,
                ngram_range=ngram
            )
            params_set = "v{}-mf{}-ng{}".format(vector_set, mf, ngram)

            Naive = naive_bayes.BernoulliNB()
            Naive.fit(total_datasets.train.x, total_datasets.train.y)
            # predict the labels on dev dataset
            predictions_NB = Naive.predict(total_datasets.dev.x)
            # Show result
            plot_result("Naive Bayes - {}".format(params_set), total_datasets.dev.y, predictions_NB, params.labels, params.train.log_dir)
            tight_layout(Naive, total_datasets.dev.x, total_datasets.dev.y, params.labels, params.train.fig_dir+"/{}-nb.png".format(params_set))

            clf = RandomForestClassifier(n_estimators=10, max_leaf_nodes=50)
            clf.fit(total_datasets.train.x, total_datasets.train.y)
            predictions_rf = Naive.predict(total_datasets.dev.x)
            plot_result("Random Forest - {}".format(params_set), total_datasets.dev.y, predictions_rf, params.labels, params.train.log_dir)
            tight_layout(clf, total_datasets.dev.x, total_datasets.dev.y, params.labels, params.train.fig_dir+"/{}-rf.png".format(params_set))


                
            
                





