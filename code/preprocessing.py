import yaml
import pandas as pd
import numpy as np
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from collections import Counter
from feature_extraction import get_vectorizer, vectorize
from scipy import sparse
import os


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
        pd.core.frame.DataFrame: A dataframe includes all data in this project.
    """
    df = pd.read_csv(params.data.path)
    if params.data.small_dataset:
        df = df[['0', '1', '2']][:params.data.small_dataset_size]
    print("Loading data {}".format(len(df)))
    return df.dropna()


def split_data(df: pd.core.frame.DataFrame, params: dict):
    """
    This method split data into train dataset(80%), dev dataset(10%), and test dataset(10%).

    Args:
        df (pd.core.frame.DataFrame): A dataset contains all data in this project.
        params (dict): A dictionary contains all parameters in this project.

    Returns:
        train_dataset (pd.core.frame.DataFrame): train dataset
        dev_dataset (pd.core.frame.DataFrame): development dataset
        test_dataset (pd.core.frame.DataFrame): test dataset
    """
    train_dataset, other_dataset = train_test_split(
        df, 
        train_size=0.8, 
        random_state=params.rand_seed,
        shuffle=True
    )
    dev_dataset, test_dataset = train_test_split(
        other_dataset, 
        test_size=0.5, 
        random_state=params.rand_seed,
        shuffle=True
    )
    categories = Counter(df["0"])
    print("\nLoaded data points {}".format(len(df)))
    print("{} categories =>".format(len(categories.keys())), end="")
    for k in params.labels: print("    {}: {}".format(k, categories[k]), end="")
    print()

    categories = Counter(train_dataset["0"])
    print("     - [{}] Train data ".format(len(train_dataset)), end="")
    for k in params.labels: print("    {}: {}".format(k, categories[k]), end="")
    print()

    categories = Counter(dev_dataset["0"])
    print("     - [{}] Develop data ".format(len(dev_dataset)), end="")
    for k in params.labels: print("    {}: {}".format(k, categories[k]), end="")
    print()

    categories = Counter(test_dataset["0"])
    print("     - [{}] Test data ".format(len(test_dataset), ), end="")
    for k in params.labels: print("    {}: {}".format(k, categories[k]), end="")
    print()

    return train_dataset, dev_dataset, test_dataset

def processing(df: pd.core.frame.DataFrame, params: dict, vector_set: str="CountVectorizer", max_feature: int=5000, ngram_range: str="(1,1)"):
    """
    This method processes the whole data with special configurations and generates 
    processed dataset which is suitable to be feed into models. It will load processed data,
    if it exists.

    Args:
        df (pd.core.frame.DataFrame): A dataset contains all data in this project.
        data_type (list): _description_
        vector_set (str, optional): _description_. Defaults to "CountVectorizer".
        ngram_range (tuple, optional): _description_. Defaults to (1,1).
    """

    total_datasets = edict()
    dataset_name = ["train", "dev", "test"]
    params_set = "v{}-mf{}-ng{}".format(vector_set, max_feature, ngram_range)
    file_dir = "{}/{}".format(params.processed_data_dir, params_set)
    
    if not os.path.exists(params.processed_data_dir):
        os.mkdir(params.processed_data_dir)

    if os.path.exists(file_dir):
        for name in dataset_name:
            dataset_x = sparse.load_npz(file_dir+"/{}_x.npz".format(name))
            dataset_y = pd.read_pickle(file_dir+"/{}_y.pkl".format(name))
            total_datasets.update({name: {"x": dataset_x, "y": dataset_y}})
    else:
        os.mkdir(file_dir)
        # Select the target data and drop out blank link
        df["0"] = [int(rating) for rating in df["0"]]
            
        vectorizer = get_vectorizer(vector_set, max_feature, ngram_range, df["2"], params)
        datasets = split_data(df, params)
        print()
            
        for name, dataset in zip(dataset_name, datasets):
            dataset_x = vectorize(vectorizer, dataset["2"], params)
            # save processed data for program accelerating
            sparse.save_npz(file_dir+"/{}_x.npz".format(name), dataset_x)
            dataset_y = dataset["0"]
            dataset_y.to_pickle(file_dir+"/{}_y.pkl".format(name))
            total_datasets.update({name: {"x": dataset_x, "y": dataset_y}})
    return total_datasets
        

if __name__ == "__main__":
    # Load parameters
    params = load_params("configs.yml")
    # Dict update method
    # dict.update({"test":None})
    
    # Load data
    df = load_data(params)
    total_datasets = processing(df, params)
    

    