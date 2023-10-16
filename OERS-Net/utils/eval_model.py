import torch
from tqdm import tqdm
from config import nums


def eval(model, testloader, criterion, status, save_path, epoch):
    model.eval()
    print('Evaluating')

    raw_loss_sum = 0
    object_loss_sum = 0
    parts_loss_sum = 0
    total_loss_sum = 0
    raw_correct = 0
    object_correct = 0

    with torch.no_grad():
        for i, data in enumerate(tqdm(testloader)):

            images, labels = data
            images = images.cuda()
            labels = labels.cuda()


            raw_logits, object_logits,  parts_logits = model(images, 'train')

            raw_loss = criterion(raw_logits, labels)
            object_loss = criterion(object_logits, labels)
            parts_loss = criterion(parts_logits,
                                   labels.unsqueeze(1).repeat(1, nums).view(-1))

            total_loss = raw_loss + object_loss + parts_loss

            raw_loss_sum += raw_loss.item()
            object_loss_sum += object_loss.item()
            parts_loss_sum += parts_loss.item()

            total_loss_sum += total_loss.item()





            # raw
            pred = raw_logits.max(1, keepdim=True)[1]
            raw_correct += pred.eq(labels.view_as(pred)).sum().item()
            # object
            pred = object_logits.max(1, keepdim=True)[1]
            object_correct += pred.eq(labels.view_as(pred)).sum().item()


    raw_loss_avg = raw_loss_sum / (i+1)
    object_loss_avg = object_loss_sum / (i+1)
    parts_loss_avg = parts_loss_sum / (i+1)
    total_loss_avg = total_loss_sum / (i+1)

    raw_accuracy = raw_correct / len(testloader.dataset)
    object_accuracy = object_correct / len(testloader.dataset)



    return raw_loss_avg, parts_loss_avg, total_loss_avg, raw_accuracy, object_accuracy, object_loss_avg