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
    running_att_loss = 0.0
    running_log_loss = 0.0
    print("start " + mode + " :" + str(epoch))
    for i_batch, sample_batch in enumerate(tqdm(data_loader)):
        inputs = sample_batch["image"].to(device, dtype=torch.float32)
        labels = sample_batch["label"].to(device, dtype=torch.int64)

        if mode == "train":
            optimizer.zero_grad()
        logits, loss_list = model(inputs, labels)
        loss = loss_list[0]
        if mode == "train":
            loss.backward()
            # clip_gradient(optimizer, 1.1)
            optimizer.step()

        a = loss.item()
        running_loss += a
        if len(loss_list) > 2: # For slot training only
            running_att_loss += loss_list[2].item()
            running_log_loss += loss_list[1].item()
        running_corrects += cal.evaluateTop1(logits, labels)
        # if i_batch % 10 == 0:
        #     print("epoch: {} {}/{} Loss: {:.4f}".format(epoch, i_batch, L-1, a))
    epoch_loss = round(running_loss/L, 3)
    epoch_loss_log = round(running_log_loss/L, 3)
    epoch_loss_att = round(running_att_loss/L, 3)
    epoch_acc = round(running_corrects/L, 3)
    record[mode]["loss"].append(epoch_loss)
    record[mode]["acc"].append(epoch_acc)
    record[mode]["log_loss"].append(epoch_loss_log)
    record[mode]["att_loss"].append(epoch_loss_att)


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)

