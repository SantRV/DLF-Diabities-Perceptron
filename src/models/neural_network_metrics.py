
class NNMetrics():
    def __init__(self,
                 model_name: str,
                 loss=[],
                 accuracy=[],
                 precision=[],
                 recall=[],
                 learning_rate=[]
                 ) -> None:

        self.model_name = model_name
        self.loss: list = loss
        self.accuracy: list = accuracy
        self.precision: list = precision
        self.recall: list = recall
        self.learning_rate: list = learning_rate
        pass

    def get_metric(self, metric_name):
        if metric_name == "loss":
            return self.loss
        if metric_name == "accuracy":
            return self.accuracy
        if metric_name == "precision":
            return self.precision
        if metric_name == "recall":
            return self.recall
        if metric_name == "learning_rate":
            return self.learning_rate

        return self.loss
