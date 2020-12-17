import numpy as np

def Accuracy(pred_set, t_set):
    return np.sum(pred_set.argmax(axis=1) == t_set) / t_set.shape[0]

def batchAccuracy(model, Loader):
    Acc, Num = 0, 0
    for batch in Loader:
        img = batch['imgs'] / 255
        targets = batch['targets']

        output = model.forward(img, False)
        Acc += np.sum(output.argmax(axis=1) == targets)
        Num += img.shape[0]
    return Acc / Num

def ConfusionMatrix(pred_set, t_set, norm=False):
    pred_labels = pred_set.argmax(axis=1)
    cm = np.zeros((pred_set.shape[1], pred_set.shape[1]))
    for i in range(len(t_set)):
        cm[int(t_set[i])][int(pred_labels[i])] += 1