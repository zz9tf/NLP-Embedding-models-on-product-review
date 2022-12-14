from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def tokenize(text):
    """
    The method for vectorizer to tokenize before vectorizer to default tokenize the text.

    Args:
        text (str): A string to be tokenized in vectorizer.

    Returns:
        text (str): A string has been tokenized.
    """
    global id, processing_step, total_num #, is_stem, is_stopwords
    id += 1
    stars = '*'*int(50*id/total_num)
    print("{} data points: |{:50s}| {:.2f}% [{}|{}]".format(
        processing_step,
        stars, 
        100*id/total_num, 
        id, 
        total_num)
        , end="\r")
    
    # stemmer = SnowballStemmer("english")
    # text = " ".join([stemmer.stem(word) if is_stem else word
    #             for word in word_tokenize(text.lower()) 
    #             if is_stopwords and word not in stopwords.words("english")])

    return text
    


def get_vectorizer(vector_set, max_feature, ngram_range, dataset, params):
    """
    The method to generate vectorizer.

    Args:
        vector_set (str): Vectorizer to be used.
        max_feature (int): Max feature for vectorizer.
        ngram_range (tuple): ngram range for vectorizer.
        dataset (pd.core.frame.DataFrame): A dataset contains all data in this project.
        params (dict): A dictionary contains all parameters in this project.

    Returns:
        Vectorizer: the vectorizer
    """
    verctorizer = None
    if vector_set == "CountVectorizer":
        verctorizer = CountVectorizer(
            tokenizer=tokenize, 
            ngram_range=eval(ngram_range), 
            max_features=max_feature)
    elif vector_set == "TfidfVectorizer":
        verctorizer = TfidfVectorizer(
            tokenizer=tokenize, 
            ngram_range=eval(ngram_range), 
            max_features=max_feature)
    else:
        assert False, "Not valid vectorizer {}.".format(vector_set)
    global id, processing_step, total_num, is_stem, is_stopwords
    id = 0
    processing_step = "Fitting"
    total_num = len(dataset)
    is_stem = params.feature.is_stem
    is_stopwords = params.feature.is_stopwords

    verctorizer.fit(dataset)
    print()
    
    del id, processing_step, total_num, is_stem, is_stopwords
    
    return verctorizer


def vectorize(vect_trans, dataset, params):
    """
    This method use [vect_trans] to vectorize [dataset]

    Args:
        vect_trans : the vectorizer
        dataset (pd.core.frame.DataFrame): A dataset to be vectorized.
        params (dict): A dictionary contains all parameters in this project.

    Returns:
        transform_result (pd.core.frame.DataFrame): A dataset has been vectorized.
    """
    global id, processing_step, total_num, is_stem, is_stopwords
    id = 0
    processing_step = "Transforming"
    total_num = len(dataset)
    is_stem = params.feature.is_stem
    is_stopwords = params.feature.is_stem

    transform_result = vect_trans.transform(dataset)
    print()
    
    del id, processing_step, total_num, is_stem, is_stopwords
    
    return transform_result