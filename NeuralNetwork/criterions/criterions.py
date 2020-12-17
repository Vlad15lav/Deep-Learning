import numpy as np

class Criterion:
    def __init__(self, name, reduction=True):
        if name == 'CrossEntropy':
            self.__loss_func = self.__CrossEntropyLoss
        elif name == 'MSELoss':
            self.__loss_func = self.__MSELoss
        elif name == 'L1Loss':
            self.__loss_func = self.__L1Loss
        elif name == 'BCELoss':
            self.__loss_func = self.__BCELoss
        else:
            raise Exception("Invalid criteria!")
        if reduction:
            self.__reduc = np.mean
        else:
            self.__reduc = np.sum
    
    def __CrossEntropyLoss(self, pred_set, t_set):
        return -self.__reduc(np.log(pred_set[np.arange(len(t_set)), t_set.reshape(1, -1)]))
    
    def __MSELoss(self, pred_set, t_set):
        return self.__reduc((pred_set - t_set) ** 2)
    
    def __L1Loss(self, pred_set, t_set):
        return self.__reduc(np.linalg.norm(pred_set - t_set, axis=1))

    def __BCELoss(self, pred_set, t_set):
        return -self.__reduc(np.log(pred_set) * t_set + np.log(1 - pred_set) * (1 - t_set))
    
    def loss(self, pred_set, t_set):
        return self.__loss_func(pred_set, t_set)