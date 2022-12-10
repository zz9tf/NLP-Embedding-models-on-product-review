<div align=center>
  <h1>Final project: Human’s attitude in models logic</h1>
  <b>Zheng Zheng<br/></b>
  <i>Michtom School of Computer, Brandies, Waltham, MA 02453 USA zhengzheng@brandeis.edu</i>
</div>

## 1. Introduction
### 1.1 What lets you explore the problem you did?
Rating prediction is an important topic in the Recommended System (RS). With more efficient prediction, companies can better understand customers' behaviors, and offer better recommendations to obtain benefits. Some related works have combined rating prediction with customer reviews. Therefore, the rating prediction problem becomes a Natural Language Processing (NLP) classification problem. The main reason I explore this problem is because I want to learn the effects of NLP solutions on RS problems.

### 1.2 Why is it interesting to you?
Related works cover many traditional models on this RS problem. Some works applied many traditional models to 
improve the model's performance. Through this topic, I can analyze how our models think about the relationship 
between human reviews and a human's attitude.

## 2. Data

### 2.1 What is the dataset you are using?

The [dataset](https://drive.google.com/file/d/1CfYn5Kmcz86pcL5kz9dOvDaA-_INQUZK/view) used in this experiment is Amazon reviews, which are 3,999,913 reviews and ratings collected by Amazon. This dataset includes product information, user information, ratings, and plaintext reviews. And this dataset is a balanced dataset posted on Kaggle (Figure 1). Although it has been split into a training dataset and a test dataset on Kaggle, I merge two datasets and split the dataset again in this project.

<center>
    <table>
        <tr>
            <td>Polarity</td>
            <td>Review Title</td>
            <td>Review Body</td>
        </tr>
        <tr>
            <td>1 (Negative)</td>
            <td>Batteries died within a year ..</td>
            <td>I bought this charger in Jul 2003 and it worked OK for a while. The design is nice and convenient …</td>
        </tr>
        <tr>
            <td>2 (Positive)</td>
            <td>One of the best game music soundtracks - for a game I didn't really play</td>
            <td>Despite the fact that I have only played a small portion of the game, …</td>
        </tr>
    </table>
</center>


### 2.2 Data processing

In this project, I used the function train_test_split(), which is built into the package scikit-learn, to split data into train dataset, development dataset, and test dataset (Tabel 1).

<div align="center">
  <table>
     <tr>
        <td>Dataset</td>
        <td>Train dataset</td>
        <td>Development dataset</td>
        <td>Test dataset</td>
     </tr>
     <tr>
        <td>All</td>
        <td>3,199,930</td>
        <td>399,991</td>
        <td>399,992</td>
     </tr>
     <tr>
        <td>Positive</td>
        <td>1,600,314</td>
        <td>199,855</td>
        <td>199,793</td>
     </tr>
     <tr>
        <td>Negative</td>
        <td>1,599,616</td>
        <td>200,136</td>
        <td>200,199</td>
     </tr>
  </table>
  <div style="width:70%">
    <h3><b>Tabel 1: Datasets basic information</b></h3>
    <p style="text-align: left;">
      There are two categories. Ratings 5 and 4 are supposed to be positive data. And ratings 2 and 1 are supposed to be negative data. Rating 3 data points are ignored and not applied in this dataset
    </p>
  </div>
</div>

These datasets have been lowered and stemmed. Stop words have also been removed.

## 3. Results

In this section, I have used the logistic regression (LR) model, the Naive Bayes (NB) model, and the Nearest Centroid (NC) model built into the scikit-learn package, combining different feature vectorizer configurations and different hyperparameters to generate some fine-tuned models for Reviews rating prediction.

### 3.1 Feature vectorizer experience

In the experience, I mainly set up different configurations on feature vectorizers to figure out good configurations for vectorizers. These differences are different vectorizers, different n-gram ranges, and different max features. Different vectorizers use different strategies to vectorize input data. Different n-grams indicate that different-length items will be vectorized. The different max feature means it limits different maximum features a vectorizer can have.

#### 3.1.1 Vectorizer difference

This experiment kept extracting at most 50 features from reviews. It processed reviews to 1 item length. All models are default models. It only used two different vectorizers to analyze the vectorizer effect on model performances. One is CountVectorizer. Another is TfidfVectorizer. The following are the results. 

<div align="center">
    <table>
        <tr>
            <td>Model</td>
            <td>Vectorizer</td>
            <td>Accuracy</td>
            <td>Precision</td>
            <td>Recall</td>
            <td>Sensitivity</td>
            <td>Specificity</td>
            <td>F1</td>
        </tr>
        <tr>
            <td rowspan="2">LR</td>
            <td>Count</td>
            <td>63.61%</td>
            <td>64.62%</td>
            <td>60.25%</td>
            <td>62.72%</td>
            <td>66.96%</td>
            <td>0.6236</td>
        </tr>
        <tr>
            <td>Tfidf</td>
            <td>63.59%</td>
            <td>63.46%</td>
            <td>64.20%</td>
            <td>63.73%</td>
            <td>62.99%</td>
            <td>0.6383</td>
        </tr>
        <tr>
            <td rowspan="2">NB</td>
            <td>Count</td>
            <td>56.36%</td>
            <td>57.25%</td>
            <td>50.47%</td>
            <td>55.66%</td>
            <td>62.25%</td>
            <td>0.5364</td>
        </tr>
        <tr>
            <td>Tfidf</td>
            <td>56.36%</td>
            <td>57.25%</td>
            <td>50.47%</td>
            <td>55.66%</td>
            <td>62.25%</td>
            <td>0.5364</td>
        </tr>
        <tr>
            <td rowspan="2">NC</td>
            <td>Count</td>
            <td>53.18%</td>
            <td>53.71%</td>
            <td>46.59%</td>
            <td>52.78%</td>
            <td>59.79%</td>
            <td>0.4989</td>
        </tr>
        <tr>
            <td>Tfidf</td>
            <td>60.64%</td>
            <td>60.56%</td>
            <td>61.18%</td>
            <td>60.72%</td>
            <td>60.10%</td>
            <td>0.6087</td>
        </tr>
    </table>
    <div style="width:70%">
        <h3><b>Tabel 2: Vectorizer</b></h3>
        <p style="text-align: left; font-size:11px">
            The negative category is supposed to be the main category for this binary classification task. This table focuses on the difference caused by the vectorizer feature.
        </p>
    </div>
</div>

As a result, TfidfVectorizer is a better vectorizer for the LR model and the NC model. For the LR model, TfidfVectorizer is harmful to precision and specificity. But for the NC model, all performances are improved. As for the NB model, both CountVectorizer and TfidfVectorizer are workable since there is no difference between the results based on the two vectorizers.

#### 3.1.2 N-grams difference

This experiment kept extracting at most 50 features from reviews. It applied TfidfVectorizer. All models are default models. It only used three different n-gram strategies. (1, 1) is for one item length, (2, 2) is for two items length, (5, 5) is for five items length.

<div align="center">
    <table>
    <tr>
        <td>Model</td>
        <td>N-grams</td>
        <td>Accuracy</td>
        <td>Precision</td>
        <td>Recall</td>
        <td>Sensitivity</td>
        <td>Specificity</td>
        <td>F1</td>
    </tr>
    <tr>
        <td rowspan="3">LR</td>
        <td>(1,1)</td>
        <td>63.59%</td>
        <td>63.46%</td>
        <td>64.20%</td>
        <td>63.73%</td>
        <td>62.99%</td>
        <td>0.6383</td>
    </tr>
    <tr>
        <td>(2,2)</td>
        <td>66.87%</td>
        <td>66.29%</td>
        <td>68.75%</td>
        <td>67.50% </td>
        <td>64.98%</td>
        <td>0.675</td>
    </tr>
    <tr>
        <td>(5,5)</td>
        <td>65.24%</td>
        <td>65.63%</td>
        <td>64.10%</td>
        <td>64.87%</td>
        <td>66.38%</td>
        <td>0.6486</td>
    </tr>
    <tr>
        <td rowspan="3">NB</td>
        <td>(1,1)</td>
        <td>56.36%</td>
        <td>57.25%</td>
        <td>50.47%</td>
        <td>55.66%</td>
        <td>62.25%</td>
        <td>0.5364</td>
    </tr>
    <tr>
        <td>(2,2)</td>
        <td>56.10%</td>
        <td>54.86%</td>
        <td>69.28%</td>
        <td>58.24%</td>
        <td>42.90% </td>
        <td>0.6123</td>
    </tr>
    <tr>
        <td>(5,5)</td>
        <td>60.83%</td>
        <td>61.23%</td>
        <td>59.17%</td>
        <td>60.45%</td>
        <td>62.49%</td>
        <td>0.6018</td>
    </tr>
    <tr>
        <td rowspan="3">NC</td>
        <td>(1,1)</td>
        <td>60.64%</td>
        <td>60.56%</td>
        <td>61.18%</td>
        <td>60.72%</td>
        <td>60.10%</td>
        <td>0.6087</td>
    </tr>
    <tr>
        <td>(2,2)</td>
        <td>63.90%</td>
        <td>63.35%</td>
        <td>66.10%</td>
        <td>64.51%</td>
        <td>61.70%</td>
        <td>0.6469</td>
    </tr>
    <tr>
        <td>(5,5)</td>
        <td>63.78%</td>
        <td>63.81%</td>
        <td>63.78%</td>
        <td>63.75%</td>
        <td>63.78%</td>
        <td>0.638</td>
    </tr>
    </table>
    <div style="width:70%">
        <h3><b>Table 3: N-grams</b></h3>
        <p style="text-align: left; font-size:11px">
            The negative category is supposed to be the main category for this binary classification task. This table focuses on the difference caused by the N-grams feature.
        </p>
    </div>
</div>

As a result, (2,2) strategies have better performance on the LR model and the NC model. The NB model has improved recall and sensitivity but lower accuracy, precision, and specificity.

#### 3.1.3 Max feature difference

This experiment applied the TfidfVectorizer and kept using (2,2) from reviews. All models are default models. It only used four different max feature strategies. This experiment applied 5, 50, 200, and 1000 max feature strategies.

<div align="center">
    <table>
        <tr>
            <td>Model</td>
            <td>Max feature</td>
            <td>Accuracy</td>
            <td>Precision</td>
            <td>Recall</td>
            <td>Sensitivity</td>
            <td>Specificity</td>
            <td>F1</td>
        </tr>
        <tr>
            <td rowspan="4">LR</td>
            <td>5</td>
            <td>57.05%</td>
            <td>56.63%</td>
            <td>60.50%</td>
            <td>57.54%</td>
            <td>53.61%</td>
            <td>0.585</td>
        </tr>
        <tr>
            <td>50</td>
            <td>66.87%</td>
            <td>66.29% </td>
            <td>68.75%</td>
            <td>67.50% </td>
            <td>64.98%</td>
            <td>0.675</td>
        </tr>
        <tr>
            <td>200</td>
            <td>74.37%</td>
            <td>74.07%</td>
            <td>75.06%</td>
            <td>74.68% </td>
            <td>73.69%</td>
            <td>0.7456</td>
        </tr>
        <tr>
            <td>1000</td>
            <td>78.03%</td>
            <td>77.66%</td>
            <td>78.73%</td>
            <td>78.40%</td>
            <td>77.32%</td>
            <td>0.7819</td>
        </tr>
        <tr>
            <td rowspan="4">NB</td>
            <td>5</td>
            <td>50.51%</td>
            <td>50.28%</td>
            <td>98.34%</td>
            <td>61.05%</td>
            <td>2.61%</td>
            <td>0.6654</td>
        </tr>
        <tr>
            <td>50</td>
            <td>56.10%</td>
            <td>54.86%</td>
            <td>69.28%</td>
            <td>58.24%</td>
            <td>42.90% </td>
            <td>0.6123</td>
        </tr>
        <tr>
            <td>200</td>
            <td>59.16%</td>
            <td>58.47%</td>
            <td>63.41%</td>
            <td>59.97% </td>
            <td>54.90%</td>
            <td>0.6084</td>
        </tr>
        <tr>
            <td>1000</td>
            <td>61.82%</td>
            <td>61.56%</td>
            <td>63.06% </td>
            <td>62.09%</td>
            <td>60.58%</td>
            <td>0.623</td>
        </tr>
        <tr>
            <td rowspan="4">NC</td>
            <td>5</td>
            <td>57.00%</td>
            <td>56.48%</td>
            <td>61.32%</td>
            <td>57.63%</td>
            <td>52.68%</td>
            <td>0.588</td>
        </tr>
        <tr>
            <td>50</td>
            <td>63.90% </td>
            <td>63.35%</td>
            <td>66.10%</td>
            <td>64.51%</td>
            <td>61.70%</td>
            <td>0.6469</td>
        </tr>
        <tr>
            <td>200</td>
            <td>69.74%</td>
            <td>69.21%</td>
            <td>71.19%</td>
            <td>70.30%</td>
            <td>68.29%</td>
            <td>0.7019</td>
        </tr>
        <tr>
            <td>1000</td>
            <td>70.78%</td>
            <td>70.23%</td>
            <td>72.23%</td>
            <td>71.37%</td>
            <td>69.33%</td>
            <td>0.7122</td>
        </tr>
    </table>
    <div style="width:70%">
        <h3><b>Tabel 4: Max feature</b></h3>
        <p style="text-align: left; font-size:11px">
            The negative category is supposed to be the main category for this binary classification task. This table focuses on the difference caused by the different max feature strategies.
        </p>
    </div>
</div>

A higher max feature is always a good feature for our models to capture detailed information existing in data points.

### 3.3 Test result

In this experiment, I applied fine-tuned LR model, BN model, and NC model 
into the test dataset, to visualize their effectiveness.

<div align="center">
    <table>
        <tr>
            <td>Model</td>
            <td>Accuracy</td>
            <td>Precision</td>
            <td>Recall</td>
            <td>Sensitivity</td>
            <td>Specificity</td>
            <td>F1</td>
        </tr>
        <tr>
            <td>LR</td>
            <td>78.12%</td>
            <td>77.79%</td>
            <td>78.78%</td>
            <td>78.46%</td>
            <td>77.46%</td>
            <td>0.7828</td>
        </tr>
        <tr>
            <td>NB</td>
            <td>64.66%</td>
            <td>65.60%</td>
            <td>61.79%</td>
            <td>63.82%</td>
            <td>67.53%</td>
            <td>0.6364</td>
        </tr>
        <tr>
            <td>NC</td>
            <td>69.80%</td>
            <td>69.34%</td>
            <td>71.08%</td>
            <td>70.28%</td>
            <td>68.51%</td>
            <td>0.702</td>
        </tr>
    </table>
    <div style="width:70%">
        <h3><b>Tabel 5: Test - All model performances</b></h3>
        <p style="text-align: left; font-size:11px">
            The performance of models LR, BN, and NC is shown in this table.
        </p>
    </div>
</div>

For the model LR, I applied TfidfVectorizer, (2,2) n-grams range, and 1000 max features. For the model NB, I applied TfidfVectorizer, (5,5) n-grams range, and 200 max features. For the mdoel NC, I applied TfidfVectorizer, (2,2) n-grams range, and 200 max features.

<div align="center">
    <table>
        <tr>
            <td></td>
            <td>precision</td>
            <td>recall</td>
            <td>f1-score</td>
            <td>support</td>
        </tr>
        <tr>
            <td>1 (Negative)</td>
            <td>77.79%</td>
            <td>78.78%</td>
            <td>72.28%</td>
            <td>200199</td>
        </tr>
        <tr>
            <td>2 (Positive)</td>
            <td>78.46%</td>
            <td>77.46%</td>
            <td>77.96%</td>
            <td>199793</td>
        </tr>
        <tr>
            <td>accuracy</td>
            <td></td>
            <td></td>
            <td>78.12%</td>
            <td>399992</td>
        </tr>
        <tr>
            <td>macro avg</td>
            <td>78.12%</td>
            <td>78.12%</td>
            <td>78.12%</td>
            <td>399992</td>
        </tr>
        <tr>
            <td>weighted avg</td>
            <td>78.12%</td>
            <td>78.12%</td>
            <td>78.12%</td>
            <td>399992</td>
        </tr>
    </table>
    <div style="width:70%">
        <h3><b>Tabel 5: Test - Best performance</b></h3>
        <p style="text-align: left; font-size:11px">
            This table shows the best performance of models on the test dataset. This performance is generated by the LR model with TfidfVectorizer, (2,2) n-grams range, and 1000 max features.
        </p>
    </div>
</div>

Finally, the LR model has the best performance for this classification task.

## 4. Discussion

Based on the experiment results, different features have different effects on different models. For example, TfidfVectorizer has mathematical advantages because it considers inverse term frequency. However, the result shows the LR and NC models are sensitive to it, the NB model is not impacted by this change. The reason for this phenomenon is that the Tf-idf method allows term vectors more similar in mathematics, which improves LR and NC models performances. However, for the NB model, the possibility of term appearance doesn’t change too much. Therefore, the NB model is not sensitive to this change. 

Longer item sequence is better for models performance on sequence classification. But a too long sequence is also harmful for some models, which happened when I used 5 items as an input sequence. The reason for this problem is that longer sequences reduce the similarity between input vectors. Therefore, it’s harmful for some models to capture the characteristics of different categories of input vectors.

Larger max features are always beneficial for models to capture details that exist in data. However, it has the obvious best range of max features. Since when the max feature changed from 200 to 1000, for some models, its benefits are not proportionate to the change it did. For example, for the model NC, it just improved about 1 present. 

I’m surprised that models’ performance isn’t always good with TfidfVectorizer. And I’m also surprised that improving the max feature is always an effective method for my models. But, also, the more features my models tried to capture, the slower my models train process is. Overview this analysis, all models have acceptable performance on this classification task.



## 5. Conclusion

In conclusion, this research explores the impact of different configurations on the NB, LR, and NC models’ performances. The hardest part is deciding the next research step and explaining the phenomenon. If I had more time, I would spend more time doing more experiments to verify the conclusion of this research.
