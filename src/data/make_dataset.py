# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np

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
    
    # load the true.csv dataset from the raw folder (input_filepath)
    # truedata = pd.read_csv('{}/true.csv'.format(input_filepath))
    # load the fake.csv dataset from the raw folder (input_filepath)
    # fakedata = pd.read_csv('{}/fake.csv'.format(input_filepath))
    
    # load the csv dataset using numpy
    truedata = np.loadtxt('{}/true.csv'.format(input_filepath), delimiter=',', skiprows=1)
    fakedata = np.loadtxt('{}/fake.csv'.format(input_filepath), delimiter=',', skiprows=1)
    
    # TODO: convert both datasets into a torch tensor
    #true_tensor = ...
    #fake_tensor = ...
    
    # add labels to the true and fake datasets
    true_labels = torch.ones(true_tensor.shape[0], 1)
    fake_labels = torch.zeros(fake_tensor.shape[0], 1)
    
    # combine the true and fake datasets into one dataset
    data = torch.cat((true_tensor, fake_tensor), 0)
    
    # combine the true and fake labels into one dataset
    labels = torch.cat((true_labels, fake_labels), 0)
    
    # combine data and labels
    dataset = torch.utils.data.TensorDataset(data, labels)
    
    # split the dataset randomly into a training and test set with 0.2 test size
    trainset, testset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), int(0.2*len(dataset))])
    
    # save dataset into the processed folder (output_filepath)
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
    makedata()
