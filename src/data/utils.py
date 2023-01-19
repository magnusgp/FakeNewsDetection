import re
import string

import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from transformers import AutoTokenizer


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub("\[[^]]*\]", "", text)


# Removing URL's
def remove_between_square_brackets(text):
    return re.sub(r"http\S+", "", text)


# Removing the stopwords from text
def remove_stopwords(text):
    final_text = []
    ### Handle the stop words
    # nltk.download('stopwords')
    stop = set(stopwords.words("english"))
    punctuation = list(string.punctuation)
    stop.update(punctuation)
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


def roberta_encode(texts, tokenizer, MAX_LEN):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype="int32")
    attention_mask = np.zeros((ct, MAX_LEN), dtype="int32")
    token_type_ids = np.zeros(
        (ct, MAX_LEN), dtype="int32"
    )  # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)

        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[: (MAX_LEN - 2)])

        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype="int32")

        # Set to 1s in the attention input
        attention_mask[k, :input_length] = 1

    return {
        "input_word_ids": input_ids,
        "input_mask": attention_mask,
        "input_type_ids": token_type_ids,
    }


def tokenize(dataframe):
    """Tokenizes text from a string into a list of strings (tokens)"""
    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    return tokenizer(
        dataframe.tolist(), padding=True, truncation=True, return_tensors="pt"
    )
