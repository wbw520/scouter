import torch


def evaluateTop1(logits, labels):
    with torch.no_grad():
        pred = logits.argmax(dim=1)
        return torch.eq(pred, labels).sum().float().item()/labels.size(0)


def evaluateTop5(logits, labels):
    with torch.no_grad():
        maxk = max((1, 5))
        labels_resize = labels.view(-1, 1)
        _, pred = logits.topk(maxk, 1, True, True)
        return torch.eq(pred, labels_resize).sum().float().item()/labels.size(0)


class MetricLog():
    def __init__(self):
        self.record = {"train": {"loss": [], "acc": [], "log_loss": [], "att_loss": []},
                       "val": {"loss": [], "acc": [], "log_loss": [], "att_loss": []}}

    def print_metric(self):
        print("train loss:", self.record["train"]["loss"])
        print("val loss:", self.record["val"]["loss"])
        print("train acc:", self.record["train"]["acc"])
        print("val acc:", self.record["val"]["acc"])
        print("train CE loss", self.record["train"]["log_loss"])
        print("val CE loss", self.record["val"]["log_loss"])
        print("train attention loss", self.record["train"]["att_loss"])
        print("val attention loss", self.record["val"]["att_loss"])
