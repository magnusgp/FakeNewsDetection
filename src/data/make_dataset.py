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

import torch


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


    df = pd.concat([true, false])

    ### Also the true category only has two subjects run:
    ### true.subject.value_counts()
    ### false.subject.value_counts()

    df['text'] = df['text'] + " " + df['title']
    del df['title']
    del df['subject']
    del df['date']

    nltk.download('stopwords')
    stop = set(stopwords.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    def strip_html(text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(text):
        return re.sub('\[[^]]*\]', '', text)

    # Removing URL's
    def remove_between_square_brackets(text):
        return re.sub(r'http\S+', '', text)

    # Removing the stopwords from text
    def remove_stopwords(text):
        final_text = []
        for i in text.split():
            if i.strip().lower() not in stop:
                final_text.append(i.strip())
        return " ".join(final_text)

    # Removing the noisy text
    def denoise_text(text):
        text = strip_html(text)
        text = remove_between_square_brackets(text)
        text = remove_stopwords(text)
        return text

    # Apply function on review column
    df['text'] = df['text'].apply(denoise_text)


    # true_np = true.to_numpy(dtype=np.float16)
    # flse_np = false.to_numpy(dtype=np.float16)


    # TODO: convert both datasets into a torch tensor
    # true_tensor = torch.from_numpy(true_np)
    # fake_tensor = torch.tensor(false['title', ' text', 'subject', 'date'].values)

    
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
    makedata(r"C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/MLOps/Projekt/FakeNewsDetection/data/raw",
             r"C:/Users/arian/OneDrive/Desktop/Kunstig_Intelligens_og_Data/MLOps/Projekt/FakeNewsDetection/data/processed")
