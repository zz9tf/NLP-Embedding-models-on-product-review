import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')


from preprocessing import load_params, load_data, processing
from model import models

# Load parameters
params = load_params("configs.yml")

# Load data
df = load_data(params)

log = open(params.train.log_dir + "/log.txt", "w")
log.close()

# Experiment 1
if params.train.log_dir != None:
    log = open(params.train.log_dir + "/log.txt", "a+")
    log.write("Experiment 1\n")
    log.close()
for vector_set in params.feature.verctorizers:
    mf = 50
    ngram = "(1,1)"
    total_datasets = processing(
        df=df,
        params=params,
        vector_set=vector_set,
        max_feature=mf,
        ngram_range=ngram
    )
    params_set = "v{}-mf{}-ng{}".format(vector_set, mf, ngram)
    
    models(total_datasets, params_set, params)

# Experiment 2
if params.train.log_dir != None:
    log = open(params.train.log_dir + "/log.txt", "a+")
    log.write("\n\nExperiment 2\n")
    log.close()
for ngram in params.feature.ngram_range:
    vector_set = "TfidfVectorizer"
    mf = 50
    total_datasets = processing(
        df=df,
        params=params,
        vector_set=vector_set,
        max_feature=mf,
        ngram_range=ngram
    )
    params_set = "v{}-mf{}-ng{}".format(vector_set, mf, ngram)
    models(total_datasets, params_set, params)

# Experiment 3
if params.train.log_dir != None:
    log = open(params.train.log_dir + "/log.txt", "a+")
    log.write("\n\nExperiment 3\n")
    log.close()
for mf in params.feature.max_feature:
    vector_set = "TfidfVectorizer"
    ngram="(1,1)"
    total_datasets = processing(
        df=df,
        params=params,
        vector_set=vector_set,
        max_feature=mf,
        ngram_range=ngram
    )
    params_set = "v{}-mf{}-ng{}".format(vector_set, mf, ngram)
    models(total_datasets, params_set, params)

                
            
                





