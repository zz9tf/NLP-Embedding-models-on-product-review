import yaml, gzip
import pandas as pd
from easydict import EasyDict as edict
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter, defaultdict



def load_params(path: str):
    """
    This method will load all global parameters in this project.
    
    Args:
        path (str): dir for a yaml file with all parameters.

    Returns:
        dict: A dictionary with all parameters in yaml file.
    """
    with open(path) as file:
        return edict(yaml.safe_load(file))


def load_data(params: dict):
    """
    A method loads all data in this project.

    Args:
        params (dict): A dictionary with all parameters.

    Returns:
        DataFrame: A dataframe includes all data in this project.
    """
    
    gz = gzip.open(params.data.path, 'rb')
    df = dict()
    for rowId, data_point in enumerate(gz):
        df[rowId] = eval(data_point)
        stars = '*'*int(50*(rowId+1)/params.data.point_num)
        print("Loading data points: |{:50s}| {:.2f}% [{}|{}]".format(
            stars, 
            100*(rowId+1)/params.data.point_num, 
            rowId+1, 
            params.data.point_num)
            , end="\r")
        if rowId+1 == params.load.small_dataset_size and params.load.small_dataset:
            break
    print()
    
    return pd.DataFrame.from_dict(df, orient='index')


def get_tokenlizer(dataset):
    def tokenize(text):
        stemmer = SnowballStemmer("english")
        global id, processing_step, total_num
        id += 1
        stars = '*'*int(50*id/total_num)
        print("{} data points: |{:50s}| {:.2f}% [{}|{}]".format(
            processing_step,
            stars, 
            100*id/total_num, 
            id, 
            total_num)
            , end="\r")
        return str([stemmer.stem(word) for word in word_tokenize(text.lower()) 
                    if word not in stopwords.words("english")])
    
    Tfidf_vect = TfidfVectorizer(
        tokenizer=tokenize,
        max_features=5000
    )
    
    global id, processing_step, total_num
    id = 0
    processing_step = "Fitting"
    total_num = len(dataset)
    Tfidf_vect.fit(dataset)
    print()
    
    del id, processing_step, total_num
    
    return Tfidf_vect


def split_data(df, parames):
    train_dataset, other_dataset = train_test_split(
        df, 
        train_size=0.8, 
        random_state=parames.preprocessing.rand_seed,
        shuffle=True
    )
    dev_dataset, test_dataset = train_test_split(
        other_dataset, 
        test_size=0.5, 
        random_state=parames.preprocessing.rand_seed,
        shuffle=True
    )
    categories = Counter(df["overall"])
    print("\nLoaded data points {}".format(len(df)))
    print("{} categories =>".format(len(categories.keys())), end="")
    for k in range(5,0,-1): print("    {}: {}".format(k, categories[k]), end="")
    print()

    categories = Counter(train_dataset["overall"])
    print("     - [{}] Train data ".format(len(train_dataset)), end="")
    for k in range(5,0,-1): print("    {}: {}".format(k, categories[k]), end="")
    print()

    categories = Counter(dev_dataset["overall"])
    print("     - [{}] Develop data ".format(len(dev_dataset)), end="")
    for k in range(5,0,-1): print("    {}: {}".format(k, categories[k]), end="")
    print()

    categories = Counter(test_dataset["overall"])
    print("     - [{}] Test data ".format(len(test_dataset), ), end="")
    for k in range(5,0,-1): print("    {}: {}".format(k, categories[k]), end="")
    print()

    return train_dataset, dev_dataset, test_dataset


def tokenlize(Tfidf_vect, dataset):
    global id, processing_step, total_num
    id = 0
    processing_step = "Transforming"
    total_num = len(dataset)
    transform_result = Tfidf_vect.transform(dataset)
    print()
    
    del id, processing_step, total_num
    
    return transform_result


def processing(parames: dict, df: pd.core.frame.DataFrame):
    
    # Select the target data and drop out blank link
    df = df[parames.data.use_data_type].dropna()
    # Lower and tokenize review text to reduce the complexity of text.
    
    df["overall"] = [int(rating) for rating in df["overall"]]
    
    Tfidf_vect = get_tokenlizer(df["reviewText"])
    
    datasets = split_data(df, parames)
    print()
    
    total_datasets = {}
    dataset_name = ["train", "dev", "test"]
    for name, dataset in zip(dataset_name, datasets):
        dataset_x = tokenlize(Tfidf_vect, dataset["reviewText"])
        total_datasets[name] = (dataset_x, df["overall"])
    
    return total_datasets

if __name__ == "__main__":
    # Load parameters
    params = load_params("configs.yml")
    # Dict update method
    # dict.update({"test":None})
    
    # Load data
    df = load_data(params)
    total_datasets = processing(params, df)
    

    