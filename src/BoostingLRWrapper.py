from jpype import JClass
from BoostingLR import BoostingLR

class BoostingLRWrapper:
    def __init__(self, max_iterations=50):
        self.max_iterations = max_iterations
        self.boosting_lr = BoostingLR(max_iterations=self.max_iterations)

    def fit(self, train_data):
        """
        Train the BoostingLR model on the provided training data.
        :param train_data: Weka Instances object containing the training data
        """
        # Use the provided training data to build the classifier
        lrt = JClass("weka.classifiers.labelranking.LRT")()
        self.boosting_lr.build_classifier(train_data, lrt)

    def predict(self, test_data):
        """
        Predict rankings for the provided test data.
        :param test_data: Weka Instances object containing the test data
        :return: List of predicted rankings for each instance in the test data
        """
        predictions = []
        for i in range(test_data.numInstances()):
            instance = test_data.instance(i)
            preds = self.boosting_lr.predict(instance)
            predictions.append(preds)
        return predictions