[![Run tests](https://github.com/magnusgp/FakeNewsDetection/actions/workflows/tests.yml/badge.svg)](https://github.com/magnusgp/FakeNewsDetection/actions/workflows/tests.yml)
[![Run isort](https://github.com/magnusgp/FakeNewsDetection/actions/workflows/isort.yml/badge.svg)](https://github.com/magnusgp/FakeNewsDetection/actions/workflows/isort.yml)

Arian Dapouyeh (s204158)\
Benjamin Fazal (s200431)\
Kasper Helverskov Petersen (s203294)\
Magnus Guldberg Petersen (s204075)



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

**Project checklist**\
### Week 1

* [x] Create a git repository
* [x] Make sure that all team members have write access to the github repository
* [x] Create a dedicated environment for you project to keep track of your packages
* [x] Create the initial file structure using cookiecutter
* [x] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [x] Add a model file and a training script and get that running
* [x] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [x] Remember to comply with good coding practices (`pep8`) while doing the project
* [ ] Do a bit of code typing and remember to document essential parts of your code
* [x] Setup version control for your data or part of your data
* [x] Construct one or multiple docker files for your code
* [x] Build the docker files locally and make sure they work as intended
* [x] Write one or multiple configurations files for your experiments
* [x] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [x] Use Weights & Biases to log training progress and other important metrics/artifacts in your code. Additionally,
      consider running a hyperparameter optimization sweep.
* [ ] Use Pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [x] Write unit tests related to the data part of your code
* [ ] Write unit tests related to model construction and or model training
* [x] Calculate the coverage.
* [x] Get some continuous integration running on the github repository
* [x] Create a data storage in GCP Bucket for you data and preferable link this with your data git version control setup 
* [x] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training in GCP using either the Engine or Vertex AI
* [x] Create a FastAPI application that can do inference using your model
* [ ] If applicable, consider deploying the model locally using torchserve
* [ ] Deploy your model in GCP using either Functions or Run as the backend

### Week 3

* [x] Check how robust your model is towards data drifting
* [x] Setup monitoring for the system telemetry of your deployed model
* [ ] Setup monitoring for the performance of your deployed model
* [ ] If applicable, play around with distributed data loading
* [ ] If applicable, play around with distributed model training
* [ ] Play around with quantization, compilation and pruning for you trained models to increase inference speed

### Additional

* [x] Revisit your initial project description. Did the project turn out as you wanted?
* [x] Make sure all group members have a understanding about all parts of the project
* [ ] Uploaded all your code to github
