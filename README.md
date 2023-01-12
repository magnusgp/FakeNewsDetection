Arian Dapouyeh (s204158)\
Benjamin Fazal (s200431)\
Kasper Helverskov Petersen (s203294)
Magnus Guldberg Petersen (s204075)\



MLOPS project description - Classifying real and fake news
==============================

**Overall goal of the project**\
In a digital world where news travel fast over the internet, it is important to know the difference between what is true and what is made up. We have seen a surge of fake news fabricated to distribute untruthful content that can alter people opinions on important topics such as climate change, politics etc. The overall goal of this project is to classify news articles as either fake or real. 

**What framework are you going to use (Kornia, Transformer, Pytorch-Geometrics)**\
We will use the Huggingface Transformer framework to classify the news articles.

**How to you intend to include the framework into your project:**\
Since the Transformers repository from the Huggingface group is focused on NLP models, it is a perfect fit for our project. We intend to use some of the pre-trained models on our dataset, and if possible, try to improve classification performance even further. 

**What data are you going to run on (initially, may change)**\
We are going to use a dataset obtained from kaggle.com (https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset?select=Fake.csv&fbclid=IwAR2vaemWRgwEjObp8rx8ZzsVkavwuq0shny9SroYDrqe2J9pls7WAejAnpY) containing both fake and real news articles. 
The dataset contains the following attributes: title, text, subject and date and we are going to add labels to the data with respect to their origin (whether they are true or false).

**What deep learning models to you expect to use:**\
We intend to use some of the pre-trained models from the Transformer framework. We have not yet determined which, but looked at a Transformer-based language model, RoBERTa, which was intended for sequence classification. 
