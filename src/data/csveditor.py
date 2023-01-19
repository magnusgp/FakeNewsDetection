import pandas as pd
from sklearn.model_selection import train_test_split

### Given the two csv files, merge them.
def editcsv():
    # load data
    csv_true = pd.read_csv(r"data\raw\True.csv")
    csv_fake = pd.read_csv(r"data\raw\Fake.csv")
    # create a new column called label and set it to 0 for fake news
    csv_fake["label"] = 0
    # create a new column called label and set it to 1 for true news
    csv_true["label"] = 1
    # concatenate the two dataframes together
    csv_input = pd.concat([csv_fake, csv_true])
    # delete the title, subject and date columns
    csv_input.drop(["title", "subject", "date"], axis=1, inplace=True)
    # Save the merged file
    csv_input.to_csv(r"data\raw\dataset.csv", index=False)


if __name__ == "__main__":
    editcsv()
