# NLP-Embedding-models-on-product-review

This project applied traditional models and embedding models on production ratings prodiction problem. Customer's reviews will be classified according to their ratings(1-5).

## Data
The dataset used in this experiment is [Amazon reviews data](https://www.kaggle.com/datasets/kritanjalijain/amazon-reviews?select=test.csv). This dataset comes from kaggle, which contains 3,999,913 Amazon reviews from 6,643,669 users on 2,441,053 products, from the Stanford Network Analysis Project (SNAP).

### Origin

The Amazon reviews dataset consists of reviews from amazon. The data span a period of 18 years, including ~35 million reviews up to March 2013. Reviews include product and user information, ratings, and a plaintext review. For more information, please refer to the following paper: J. McAuley and J. Leskovec. Hidden factors and hidden topics: understanding rating dimensions with review text. RecSys, 2013.

### Description

The Amazon reviews polarity dataset is constructed by taking review score 1 and 2 as negative, and 4 and 5 as positive. Samples of score 3 is ignored. In the dataset, class 1 is the negative and class 2 is the positive. Each class has about 2,000,000 samples used [train.csv](https://storage.googleapis.com/kaggle-data-sets/1340369/2233682/compressed/train.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221206%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221206T151611Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=86833fa95473c1915bb74eda7056538c5ddf7c936bdb69485fbd44ffa807601702412f8238048edd49f7068c257b0b6385c3fe486d335b035df9daddd81ab79838ec1c39711e658c7439403508ea61bb299db62301253606758e465da68aa5db94e2f6d3136aea575eb7670c50dedd14f16fe84ac6f9a541efb297328d16e3e6efc81e449890006e8c9b9262643a3cfdaded20a4557e127b36429faeb82953c51485204effce94b8ba31acf30dbb32c4b1440fd72f895544a824605ba918409484dbed8c6c97eada48a8b11a561dcfd9fc5b2151ae5d96f42fab892e5a40e1e6597007bf9436143473d25082ad3bcea4986cbc2513d1aea4d074c3f13d92817b) and [test.csv](https://storage.googleapis.com/kaggle-data-sets/1340369/2233682/compressed/test.csv.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20221206%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20221206T151509Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=1a39b87b745d673a770d70f7e3a82aef3ee3b9124f4f78e8c0c5973ab2f0217bbee90f8a318dc10fa8a79a221cb12c654b28b9565ab2ead27c0aaeb64950b8b5d8c4a11ce7876c79a287dd1eb7d0f5ff0861764077419661e9a8253bf4fe129924f52257681053354ccdcafcd8879b31361532383e5cffaa69532803b6845b49913efbba3ce6aab097513deb282f5242ef8eee3411e7c5b06c94e0b0978a45d6b5ce02ea0b176bf0dcec7245e073002f1bfa4e5713afbd218d7222211facae10eb3c70b9181b617a4a062ee306d673772b7d3903cc24544b0f94be5858fdab2a03a1f8bbcbbff8d0a93ca81b0428fede4f53438703991db3826d68269f95a229) from kaggle. You need to merge this two datasets into one dataset [amazon_reviews.csv]() and this project will automatically split them into train, dev, test dataset.

#### Example
![image](https://user-images.githubusercontent.com/77183284/205990846-f86fac68-2a11-4562-806d-be001cc5f3e1.png)

