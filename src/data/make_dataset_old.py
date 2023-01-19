# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import nltk
import numpy as np
import pandas as pd
import torch
import wget
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaTokenizer
from utils import (denoise_text, remove_between_square_brackets,
                   remove_stopwords, roberta_encode, strip_html, tokenize)

MAX_LEN = 256
MODEL_NAME = 'roberta-base'

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def makedata(input_filepath, output_filepath, autoTokenizer=True):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    if '{}/true.csv'.format(input_filepath) not in input_filepath:
        wget.download("Find URL.")
    if '{}/fake.csv'.format(input_filepath) not in input_filepath:
        wget.download("Find URL.")


    # load the true.csv dataset from the raw folder (input_filepath)
    true = pd.read_csv('{}/true.csv'.format(input_filepath), delimiter=',')
    # load the fake.csv dataset from the raw folder (input_filepath)
    false = pd.read_csv('{}/fake.csv'.format(input_filepath), delimiter=',')

    false['category'] = 0
    true['category'] = 1


    ### Remove the class unbalanc
    if len(true) > len(false):
        true = true.sample(n=len(false), random_state=42)
    else:
        false = false.sample(n=len(true), random_state=42)

    ### Blend the true dataset with the false dataset
    df = pd.concat([true, false])

    ### Also the true category only has two subjects run:
    ### true.subject.value_counts()
    ### false.subject.value_counts()
    ### Therefor we remove title, subject and date

    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']

    nltk.download('stopwords')

    # Apply function on review column
    df['text'] = df['text'].apply(denoise_text)

    # Data/label seperation
    X_data = df[['text']].to_numpy().reshape(-1)
    y_data = df[['category']].to_numpy().reshape(-1)

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

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=42)

    # Import tokenizer from HuggingFace

    if(autoTokenizer):
        X_train = tokenize(X_train)
        X_test = tokenize(X_test)
    else:
        tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
        X_train = roberta_encode(X_train, tokenizer, MAX_LEN)
        X_test = roberta_encode(X_test, tokenizer, MAX_LEN)

    y_train = np.asarray(y_train, dtype='int32')
    y_test = np.asarray(y_test, dtype='int32')

    ### Convert to pytorch arrays

    y_train = torch.tensor(np.asarray(y_train, dtype='int32'))
    y_test = torch.tensor(np.asarray(y_test, dtype='int32'))

    # save the test and train datasets to data/processed folder
    torch.save(X_train, '{}/X_train.pt'.format(output_filepath))
    torch.save(X_test, '{}/X_test.pt'.format(output_filepath))
    torch.save(y_train, '{}/y_train.pt'.format(output_filepath))
    torch.save(y_test, '{}/y_test.pt'.format(output_filepath))

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
    # makedata(r"C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/MLOps/Projekt/FakeNewsDetection/data/raw",
    #         r"C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/MLOps/Projekt/FakeNewsDetection/data/processed")