from utils.file import to_local_data_path, to_unix_path


class Result:
    """A recognition result"""
    __slots__ = ('filename', 'truth', 'predicted', 'is_correct', 'loss')
    filename: str
    truth: str
    predicted: str
    is_correct: bool
    loss: float

    def __init__(self, filename, truth, predicted, loss=None):
        self.filename = filename
        self.truth = truth
        self.predicted = predicted
        self.is_correct = truth == predicted
        self.loss = loss

    def serialize(self):
        return {
            'filename': to_unix_path(to_local_data_path(self.filename)),
            'truth': self.truth,
            'predicted': self.predicted,
            'loss': float(self.loss)
        }
