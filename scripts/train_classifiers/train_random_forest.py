from classifiers.base_classifier import train_and_save_model
from classifiers.random_forest import RandomForestClassifier

if __name__ == '__main__':
    classifier = RandomForestClassifier()
    train_and_save_model(classifier, do_classify=True)
