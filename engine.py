import torch
import tools.calculate_tool as cal


def train_one_epoch(model, optimizer, data_loader, device, criterion, record, epoch):
    model.train()
    L = len(data_loader)
    running_loss = 0.0
    running_corrects_1 = 0.0
    running_corrects_5 = 0.0
    for i_batch, sample_batch in enumerate(data_loader):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)
        # zero the gradient parameter
        optimizer.zero_grad()
        logits, loss = model(inputs, labels)

        loss.backward()
        optimizer.step()

        a = loss.item()
        running_loss += a
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}  LR: {}".format(epoch, i_batch, L-1, a, optimizer.param_groups[0]["lr"]))
        running_corrects_1 += cal.evaluateTop1(logits, labels)
        running_corrects_5 += cal.evaluateTop5(logits, labels)
    epoch_loss = round(running_loss/L, 3)
    epoch_acc_1 = round(running_corrects_1/L, 3)
    epoch_acc_5 = round(running_corrects_5/L, 3)
    record["train"]["loss"].append(epoch_loss)
    record["train"]["acc_1"].append(epoch_acc_1)
    record["train"]["acc_5"].append(epoch_acc_5)


@torch.no_grad()
def evaluate(model, data_loader, device, criterion, record, epoch):
    model.eval()
    L = len(data_loader)
    running_loss = 0.0
    running_corrects_1 = 0.0
    running_corrects_5 = 0.0
    print("start evaluate  " + str(epoch))
    for i_batch, sample_batch in enumerate(data_loader):
        # inputs = sample_batch["image"].to(device, dtype=torch.float32)
        # labels = sample_batch["label"].to(device, dtype=torch.int64)
        inputs = sample_batch[0].to(device, dtype=torch.float32)
        labels = sample_batch[1].to(device, dtype=torch.int64)

        logits, loss = model(inputs, labels)

        a = loss.item()
        running_loss += a
        running_corrects_1 += cal.evaluateTop1(logits, labels)
        running_corrects_5 += cal.evaluateTop5(logits, labels)
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}".format(epoch, i_batch, L-1, a))
    epoch_loss = round(running_loss/L, 3)
    epoch_acc_1 = round(running_corrects_1/L, 3)
    epoch_acc_5 = round(running_corrects_5/L, 3)
    record["val"]["loss"].append(epoch_loss)
    record["val"]["acc_1"].append(epoch_acc_1)
    record["val"]["acc_5"].append(epoch_acc_5)