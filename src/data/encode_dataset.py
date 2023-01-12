# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, TFRobertaModel
from sklearn.model_selection import train_test_split

import torch

def tokenize(dataframe, tokenizer, no_token=256):
    """Tokenizes text from a string into a list of strings (tokens)"""
    tokenized = dataframe['text'].apply((lambda x: tokenizer.encode(x, add_special_tokens=True)))
    # pad the tokens
    padded = np.array([i + [no_token]*(no_token-len(i)) for i in tokenized.values])
    # create an attention mask
    attention_mask = np.where(padded != no_token, 1, 0)
    # convert to torch tensors
    input_ids = torch.tensor(padded)  
    attention_mask = torch.tensor(attention_mask)

    return input_ids, attention_mask

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def makedata(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    
    # load data files from data/raw folder
    true = pd.read_csv('{}/true.csv'.format(input_filepath))
    false = pd.read_csv('{}/fake.csv'.format(input_filepath))
    
    # add labels as a column
    true = true.assign(label=1)
    false = false.assign(label=0)
    
    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(true, false, test_size=0.2, random_state=42)
    
    # tokenizer
    tokenizer=RobertaTokenizer.from_pretrained('roberta-base')
    
    X_train = tokenize(X_train, tokenizer)
    X_test = tokenize(X_test, tokenizer)
    
    y_train = np.asarray(y_train, dtype='int32')
    y_test = np.asarray(y_test, dtype='int32')
    
    
    
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
