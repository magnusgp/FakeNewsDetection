# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import nltk
import numpy as np
import pandas as pd
import torch
import wget
from datasets import load_dataset
from dotenv import find_dotenv, load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, RobertaTokenizer
from utils import (
    denoise_text,
    remove_between_square_brackets,
    remove_stopwords,
    roberta_encode,
    strip_html,
    tokenize,
)

MAX_LEN = 256
MODEL_NAME = "roberta-base"

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def makedata(input_filepath, output_filepath, autoTokenizer=True):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # load the true.csv dataset from the raw folder (input_filepath)
    dataset = load_dataset("csv", data_files="{}/dataset.csv".format(input_filepath))[
        "train"
    ]

    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length"),
        batched=True,
    )

    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    torch.save(dataset, "{}/dataset.pt".format(output_filepath))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    # call the main function with data in from the raw folder and out to the processed folder
    makedata(input_filepath="data/raw", output_filepath="data/processed")
