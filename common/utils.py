import numpy as np
from sklearn.metrics import cohen_kappa_score, matthews_corrcoef, roc_auc_score
from tensorflow.keras.callbacks import Callback


class MetricsHistory(Callback):
    def __init__(self, validation_data, test_data):
        super().__init__()
        self.validation_data = validation_data
        self.mccs = []
        self.kappas = []
        self.auc_scores = []
        self.test_losses = []
        self.test_data = test_data


    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = self.model.predict(self.validation_data[0])
        y_pred_labels = np.argmax(y_pred, axis=1)
        y_test_labels = np.argmax(self.validation_data[1], axis=1)

        mcc = matthews_corrcoef(y_test_labels, y_pred_labels)
        kappa = cohen_kappa_score(y_test_labels, y_pred_labels)
        auc_score = roc_auc_score(self.validation_data[1], y_pred)

        self.mccs.append(mcc)
        self.kappas.append(kappa)
        self.auc_scores.append(auc_score)
        logs['mcc'] = mcc
        logs['kappa'] = kappa
        logs['auc'] = auc_score
        
        test_loss, _ = self.model.evaluate(self.test_data[0], self.test_data[1], verbose=0)
        self.test_losses.append(test_loss)
        logs['test_loss'] = test_loss

        return logs
    