################
# COLUMNS
################
from datetime import datetime

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

################
# COMMON
################

PATH_TO_TRAIN = "data/train.csv"
PATH_TO_TEST = "data/test.csv"
LOG_DIR = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
MODEL_DIR = "data/model"
PATH_TO_PRED = "data/predictions.csv"

MALE = "male"
FEMALE = "female"

CHERBOURG = "C"
QUEENSTOWN = "Q"
SOUTHAMPTON = "S"
