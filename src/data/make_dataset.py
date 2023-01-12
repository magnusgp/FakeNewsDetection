import hydra

# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
import re,string,unicodedata
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, AutoTokenizer
import torch

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def makedata(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


MAX_LEN = 256
MODEL_NAME = 'roberta-base'

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def makedata(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # load the true.csv dataset from the raw folder (input_filepath)
    # truedata = pd.read_csv('{}/true.csv'.format(input_filepath))
    # load the fake.csv dataset from the raw folder (input_filepath)
    # fakedata = pd.read_csv('{}/fake.csv'.format(input_filepath))
    
    # load the csv dataset using numpy
    # truedata = np.loadtxt('{}/true.csv'.format(input_filepath), delimiter=',', skiprows=1) # encoding="latin1"
    # fakedata = np.loadtxt('{}/fake.csv'.format(input_filepath), delimiter=',', skiprows=1)

    true = pd.read_csv('{}/true.csv'.format(input_filepath), delimiter=',')
    false = pd.read_csv('{}/fake.csv'.format(input_filepath), delimiter=',')

    true['category'] = 1
    false['category'] = 0

    ### Remove the unbalance

    if len(true) > len(false):
        true = true.sample(n=len(false), random_state=42)
    else:
        false = false.sample(n=len(true), random_state=42)


    df = pd.concat([true, false])

    ### Also the true category only has two subjects run:
    ### true.subject.value_counts()
    ### false.subject.value_counts()

    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']

    nltk.download('stopwords')
    #stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    #stop.update(punctuation)

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    # Removing URL's
    def remove_between_square_brackets(text):
        return re.sub(r'http\S+', '', text)

    """
    # Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)
    """

    # Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        #text = remove_stopwords(text)
        return text

    # Apply function on review column
    df['text'] = df['text'].apply(denoise_text)

    X_data = df[['text']].to_numpy().reshape(-1)
    y_data = df[['category']].to_numpy().reshape(-1)



    def roberta_encode(texts, tokenizer):
        ct = len(texts)
        input_ids = np.ones((ct, MAX_LEN), dtype='int32')
        attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
        token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')  # Not used in text classification

        for k, text in enumerate(texts):
            # Tokenize
            tok_text = tokenizer.tokenize(text)

            # Truncate and convert tokens to numerical IDs
            enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN - 2)])

            input_length = len(enc_text) + 2
            input_length = input_length if input_length < MAX_LEN else MAX_LEN

            # Add tokens [CLS] and [SEP] at the beginning and the end
            input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

            # Set to 1s in the attention input
            attention_mask[k, :input_length] = 1

        return {
            'input_word_ids': input_ids,
            'input_mask': attention_mask,
            'input_type_ids': token_type_ids
        }

    category_to_id = {}
    category_to_name = {}

    for index, c in enumerate(y_data):
        if c in category_to_id:
            category_id = category_to_id[c]
        else:
            category_id = len(category_to_id)
            category_to_id[c] = category_id
            # category_to_name[category_id] = c
            category_to_name[category_id] = "politicsNews" if c == 0 else "worldnews"

        y_data[index] = category_id

    # Display dictionary
    # category_to_name

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    # Import tokenizer from HuggingFace

    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    X_train = roberta_encode(X_train, tokenizer)
    X_test = roberta_encode(X_test, tokenizer)

    ### CONSIDER SEEING IF THIS WORKS FINE
    # def tokenize(dataframe):
    #     """Tokenizes text from a string into a list of strings (tokens)"""
    #     # define tokenizer
    #    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    #    return tokenizer(dataframe['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

    # X_train = tokenize(X_train)
    # X_test = tokenize(X_test)

    y_train = np.asarray(y_train, dtype='int32')
    y_test = np.asarray(y_test, dtype='int32')


    ### Convert to pytorch arrays

    y_train = torch.tensor(np.asarray(y_train, dtype='int32'))
    y_test = torch.tensor(np.asarray(y_test, dtype='int32'))

    # TODO: convert both datasets into a torch tensor
    # true_tensor = y_train[y_train==0]
    # fake_tensor = y_train[y_train==1]

    
    # add labels to the true and fake datasets
    # true_labels = torch.ones(true_tensor.shape[0], 1)
    # fake_labels = torch.zeros(fake_tensor.shape[0], 1)
    
    # combine the true and fake datasets into one dataset
    # data = torch.cat((true_tensor, fake_tensor), 0)
    
    # combine the true and fake labels into one dataset
    # labels = torch.cat((true_labels, fake_labels), 0)
    
    # combine data and labels
    # dataset = torch.utils.data.TensorDataset(data, labels)
    
    # split the dataset randomly into a training and test set with 0.2 test size
    # trainset, testset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
    
    # concatenate the training and test sets
    trainset = torch.utils.data.TensorDataset(X_train, y_train)
    testset = torch.utils.data.TensorDataset(X_test, y_test)

    # save the test and train datasets to data/processed folder
    #torch.save(X_train, '{}/X_train.pt'.format(output_filepath))
    #torch.save(X_test, '{}/X_test.pt'.format(output_filepath))
    #torch.save(y_train, '{}/y_train.pt'.format(output_filepath))
    #torch.save(y_test, '{}/y_test.pt'.format(output_filepath))
    
    torch.save(trainset, '{}/trainset.pt'.format(output_filepath))
    torch.save(testset, '{}/testset.pt'.format(output_filepath))

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # call the main function with data in from the raw folder and out to the processed folder
    makedata(r"data/raw",
             r"data/processed")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # call the main function with data in from the raw folder and out to the processed folder
    makedata()
