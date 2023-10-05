import torch
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_regression(output,target):

     batch_size = target.size(0)

     preds = torch.round(output)

     accuracy = 100*torch.sum(preds==target) / batch_size

     total_correct = 0
     for i,pred in enumerate(preds):
        if pred in [target[i],target[i]-1,target[i]+1]:
            total_correct +=1

     accuracy_plusminus1 = 100*torch.as_tensor(total_correct/ batch_size)

     return accuracy,accuracy_plusminus1