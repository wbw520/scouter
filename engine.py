import torch
import tools.calculate_tool as cal
from tqdm.auto import tqdm


def train_one_epoch(model, data_loader, optimizer, device, record, epoch):
    model.train()
    calculation(model, "train", data_loader, device, record, epoch, optimizer)


@torch.no_grad()
def evaluate(model, data_loader, device, record, epoch):
    model.eval()
    calculation(model, "val", data_loader, device, record, epoch)


def calculation(model, mode, data_loader, device, record, epoch, optimizer=None):
    L = len(data_loader)
    running_loss = 0.0
    running_corrects = 0.0
    print("start " + mode + " :" + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)

        if mode == "train":
            optimizer.zero_grad()
        logits, loss = model(inputs, labels)
        if mode == "train":
            loss.backward()
            optimizer.step()

        a = loss.item()
        running_loss += a
        running_corrects += cal.evaluateTop1(logits, labels)
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}".format(epoch, i_batch, L-1, a))
    epoch_loss = round(running_loss/L, 3)
    epoch_acc = round(running_corrects/L, 3)
    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
