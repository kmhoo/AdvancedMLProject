import collections


class zeroR():
    """
    zeroR is a very simple classifier: for
    every test instance presented to it, the classifier returns
    the label that was seen most often in the training data.
    """

    def __init__(self):
        # if we haven't been trained, assume most frequent class is 1
        self.guess = 1
        self.type = "mostfrequent"

    def fit(self, data, labels):
        """
        Inputs: data: a list of X vectors
        labels: Y, a list of target values

        Find the most common label in the training data, and store it in self.guess
        Not required to return a value.
        """
        # TODO: Your code here
        label_count = collections.Counter()
        label_count.update(labels)
        self.guess = label_count.most_common()[0][0]

    def predict(self, testData):
        """
        Input: testData: a list of X vectors to label.

        Classify all test data as the most common label.
        returns: a list of labels of the same length as testData, where each entry corresponds
        to the model's output on each example.
        """
        # TODO: Your code here
        predictions = []
        for i in testData:
            predictions.append(self.guess)
        # print type(predictions)
        # print "preds: ", predictions
        return predictions
