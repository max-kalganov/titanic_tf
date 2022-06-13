############################
# COLUMNS
############################
from datetime import datetime
import numpy as np

PASSENGER_ID = "PassengerId"
SURVIVED = "Survived"
PCLASS = "Pclass"
NAME = "Name"
SEX = "Sex"
AGE = "Age"
SIBSP = "SibSp"
PARCH = "Parch"
TICKET = "Ticket"
FARE = "Fare"
CABIN = "Cabin"
EMBARKED = "Embarked"

KEY_COLUMNS = {
    PCLASS,
    SEX,
    AGE,
    SIBSP,
    PARCH,
    FARE,
    EMBARKED
}

NUM_FEATURES_COLUMNS = {
    PCLASS
}

############################
# PATHS
############################

PATH_TO_TRAIN = "data/train.csv"
PATH_TO_TEST = "data/test.csv"
LOG_DIR = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR = "data/models/model"
PATH_TO_PRED = "data/test_predictions/predictions.csv"

############################
# CATEGORICAL VALUES
############################

MALE = "male"
FEMALE = "female"

CHERBOURG = "C"
QUEENSTOWN = "Q"
SOUTHAMPTON = "S"

CATEG_TO_FLOAT = {
    FEMALE: 0.,
    MALE: 1.,
    CHERBOURG: -1.,
    QUEENSTOWN: 0.,
    SOUTHAMPTON: 1.,
    np.nan: -100
}
