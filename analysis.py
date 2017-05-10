#Code to train a machine learning model for the Fragile Families Challenge

#Data not provided: if you are part of this challenge and have access to the data,
#  update path to the directory with the data files (using their default names)
#  in the first line

#Usage: run this file (python analysis.py or run in IDE)
#Requires: sklearn, numpy, pandas (all installable with pip)

DATA_FOLDER = "data/FFChallenge/"

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def main():
  #Read features (indexed by ID) into data frame
  features_df = pd.read_csv(DATA_FOLDER + "background.csv")
  print "Read in data"

  #Coerce strings to numeric
  features_df = features_df.apply(pd.to_numeric, errors='coerce')
  print "Converted to numeric"

  #Read in labels
  labels_df = pd.read_csv(DATA_FOLDER + "train.csv")
  print "Read in labels"

  #Get data only for which we have labels
  labeled_features_df = features_df.loc[features_df["challengeID"].isin(labels_df.challengeID.tolist())]

  #To train a model only on data we have labels for
  #(and test it, using a train/test split)
  data = convert_to_data_matrix(labeled_features_df)

  #To make predictions on all data
  all_data = convert_to_data_matrix(features_df)

  all_preds_df = pd.DataFrame() #where predictions will be stored
  all_preds_df["challengeID"] = features_df.challengeID.tolist()

  #Make and write out predictions for all data
  outcome_vars = labels_df.columns.tolist()
  print outcome_vars
  outcome_vars.remove("challengeID")
  for outcome_var in outcome_vars:
    print "Predicting outcome variable", outcome_var
    labels = labels_df.as_matrix([outcome_var])
    print("Converted label matrix with %d labels" % labels.size)

    #Replace NaN with another number
    #TODO do better
    labels[np.isnan(labels)] = -2

    #Split into training and test
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.10, random_state=42)
    print "Split into training and test"

    #Train model
    #TODO model selection 
    clf = Lasso()
    clf.fit(train_data, train_labels)
    print "Trained model"

    #Make predictions
    test_predictions = clf.predict(test_data)
    rmse = mean_squared_error(test_predictions, test_labels)
    print "Made predictions with RMSE", rmse

    #TODO could retrain model on all data, train and test?
    all_predictions = clf.predict(all_data)
    print "Made predictions for all data"

    all_preds_df[outcome_var] = all_predictions
    print "Wrote predictions to CSV"

  print "Writing all predictions to CSV"
  all_preds_df.to_csv("preds.csv", index=False)

#Convert dataframe to data matrix (preprocessing, etc)
#Input: Pandas data frame
#Output: NumPy matrix
def convert_to_data_matrix(df):
  data = df.as_matrix()
  data = data[:, 1:] #remove first column which are the IDs
  print("Converted feature matrix with %d observations and %d features" % (data.shape[0], data.shape[1]))

  #Preprocess
  #Convert data to NumPy array
  data = np.asarray(data, dtype = "float64")

  #Replace NaN with 0
  data = np.nan_to_num(data)
  return data

if __name__ == "__main__":
  main()
