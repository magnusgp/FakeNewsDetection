# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from transformers import RobertaTokenizer, TFRobertaModel, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split

import torch

def tokenize(dataframe):
    """Tokenizes text from a string into a list of strings (tokens)"""
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer(dataframe['text'].tolist(), padding=True, truncation=True, return_tensors="pt")

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
    
    # if one dataset is larger than the other, drop samples so that they are equally large
    if len(true) > len(false):
            true = true.sample(n=len(false), random_state=42)
    else:
        false = false.sample(n=len(true), random_state=42)
    
    # present lengths of true and false datasets
    print('True dataset length: {}'.format(len(true)))
    print('False dataset length: {}'.format(len(false)))
    
    # concatenate true and false datasets
    X = pd.concat([true['text'], false['text']], ignore_index=True)
    y = pd.concat([true['label'], false['label']], ignore_index=True)
    
    # split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_train = tokenize(X_train)
    X_test = tokenize(X_test)
    
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
