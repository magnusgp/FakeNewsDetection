import pandas as pd

from sklearn import datasets

from evidently.test_suite import TestSuite
from evidently.test_preset import DataStabilityTestPreset
from evidently.test_preset import DataQualityTestPreset

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch

dataset = torch.load(
        "/Users/magnus/Desktop/DTU/5semester/MLOPS/TrueOrFakeNews/data/processed/dataset.pt")
trainset = dataset["train"]
trainset = trainset.select(range(0, 100))
dataset = trainset

#data_new = pd.DataFrame.from_dict(dataset)

# Check data stability
data_stability= TestSuite(tests=[
    DataStabilityTestPreset(),
])

data_stability.run(current_data=pd.DataFrame(dataset['input_ids'].numpy()[50:]), reference_data=pd.DataFrame(dataset['input_ids'].numpy()[:50]))

data_stability.save_html("file_stability.html")

# Check data drifting
data_drift_report = Report(metrics=[
    DataDriftPreset(),
]) 

data_drift_report.run(current_data=pd.DataFrame(dataset['input_ids'].numpy()[50:]), reference_data=pd.DataFrame(dataset['input_ids'].numpy()[:50]))

data_drift_report.save_html("file_datadrifting.html") 




