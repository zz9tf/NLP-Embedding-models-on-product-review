from sklearn.preprocessing import LabelEncoder
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
from sklearn import model_selection, naive_bayes, svm


from preprocessing import load_params, load_data, processing

# Load parameters
params = load_params("configs.yml")

# Load data
df = load_data(params)

# preprocessing data
total_datasets = processing(params, df)




