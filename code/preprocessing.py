import yaml
from easydict import EasyDict as edict
import gzip
import os
import pandas as pd
from sklearn.model_selection import train_test_split

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
        stars = '*'*int(50*(rowId+1)/params.data.point_num) # not finish yet
        print("Loading data points: |{:50s}| {:.2f}% [{}|{}]".format(
            stars, 
            (rowId+1)/params.data.point_num, 
            rowId+1, 
            params.data.point_num)
            , end="\r")
        if rowId+1 == params.load.small_dataset_size and params.load.small_dataset:
            break
    
    return pd.DataFrame.from_dict(df, orient='index')

def processing(parames: dict, df: pd.core.frame.DataFrame):
    
    
    def split_data(df):
        train_dataset, other_dataset = train_test_split(
            df[parames.data.use_data_type], 
            train_size=0.8, 
            random_state=parames.rand_seed,
            shuffle=True
        )
        dev_dataset, test_dataset = train_test_split(
            other_dataset, 
            test_size=0.5, 
            random_state=parames.rand_seed,
            shuffle=True
        )
        print("\nLoaded data points {}".format(len(df)))
        print("     - [{}] Train data".format(len(train_dataset)))
        print("     - [{}] Develop data".format(len(dev_dataset)))
        print("     - [{}] Test data".format(len(test_dataset)))
        
        return train_dataset, dev_dataset, test_dataset

    return split_data(df)
    
if __name__ == "__main__":
    # Load parameters
    params = load_params("configs.yml")
    # Dict update method
    # dict.update({"here":None})
    
    # Load data
    df = load_data(params)
    train_dataset, test_dataset, valid_dataset = processing(params, df)
    

    