# -*- coding: utf-8 -*-
import logging
import click
import torch
from pathlib import Path
from csveditor import editcsv
from datasets import load_dataset
from transformers import AutoTokenizer
from dotenv import find_dotenv, load_dotenv

MAX_LEN = 256
MODEL_NAME = "roberta-base"

# Define arguments for the input file path and output file path
@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def makedata(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    ### Write to the logger
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")

    # load the dataset from the raw folder (input_filepath)
    # if the dataset does not exist, create it
    if "{}/dataset.csv".format(input_filepath) not in input_filepath:
        editcsv()
    dataset = load_dataset("csv", data_files="{}/dataset.csv".format(input_filepath))[
        "train"
    ]

    # Split the dataset into training and test sets
    dataset = dataset.train_test_split(test_size=0.2, shuffle=True)

    # Tokenizer using roberta
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    # Apply the tokenizer to the dataset
    dataset = dataset.map(
        lambda e: tokenizer(e["text"], truncation=True, padding="max_length"),
        batched=True,
    )
    # Convert dataset to a torch tensor
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # Save the dataset
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
    makedata()
