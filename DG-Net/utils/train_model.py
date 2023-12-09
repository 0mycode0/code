import os
import glob
import torch
from tqdm import tqdm
from config import save_interval, eval_trainset, max_checkpoint_num, nums
from utils.eval_model import eval

def train(model,
          trainloader,
          testloader,
          criterion,
          optimizer,
          scheduler,
          save_path,
          start_epoch,
          end_epoch):

    max_accuracy = 0
    
    for epoch in range(start_epoch + 1, end_epoch + 1):
        model.train()

        print('Training %d epoch' % epoch)


        lr = next(iter(optimizer.param_groups))['lr']

        for i, data in enumerate(tqdm(trainloader)):

            images, labels = data
            
            images, labels = images.cuda(), labels.cuda()

            optimizer.zero_grad()


            raw_logits, object_logits,  parts_logits  = model(images, 'train')


            raw_loss = criterion(raw_logits, labels)

            object_loss = criterion(object_logits, labels)

            parts_loss = criterion(parts_logits,
                               labels.unsqueeze(1).repeat(1, nums).view(-1))



            total_loss = raw_loss + object_loss + parts_loss

            total_loss.backward()

            optimizer.step()

        scheduler.step()
        
        exp_dir = "./air"
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        # evaluation trainset
        if epoch % eval_trainset == 0:

            raw_loss_avg, parts_loss_avg, total_loss_avg, raw_accuracy, object_accuracy, object_loss_avg \
                = eval(model, trainloader, criterion, 'train', save_path, epoch)

            print(
                'Train set: raw accuracy: {:.2f}%, object accuracy: {:.2f}%'.format(
                    100. * raw_accuracy, 100. * object_accuracy))


            with open(exp_dir + '/6-2-train.txt', 'a') as file:
                file.write('Epoch : %d | Train/learning rate : %.5f | Train/object_accuracy:%.5f | Train/raw_loss_avg:%.5f'
                           '|Train/object_loss_avg:%.5f |Train/parts_loss_avg:%.5f | Train/total_loss_avg:%.5f\n'
                           %(epoch,lr,100. * object_accuracy, raw_loss_avg, object_loss_avg, parts_loss_avg, total_loss_avg))

        # eval testset
        raw_loss_avg, parts_loss_avg, total_loss_avg, raw_accuracy, object_accuracy, \
        object_loss_avg \
            = eval(model, testloader, criterion, 'test', save_path, epoch)

        print(
            ' Test set: raw accuracy: {:.2f}%, object accuracy: {:.2f}%'.format(
                100. * raw_accuracy, 100. * object_accuracy))

        with open(exp_dir + '/6-2-test.txt', 'a') as file:
            file.write(
                'Epoch : %d | Test/object_accuracy: %.5f| Test/raw_loss_avg:%.5f | Test/object_loss_avg:%.5f'
                '| Train/parts_loss_avg:%.5f | Test/total_loss_avg:%.5f \n'
                %(epoch, 100. * object_accuracy, raw_loss_avg, parts_loss_avg, object_loss_avg,
                   total_loss_avg))

        # save checkpoint
        if (epoch % save_interval == 0) or (epoch == end_epoch):
            print('Saving checkpoint')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'learning_rate': lr,
            }, os.path.join(save_path, 'epoch' + str(epoch) + '.pth'))
            
            
            
        checkpoint_list = [os.path.basename(path) for path in glob.glob(os.path.join(save_path, '*.pth'))]
        if len(checkpoint_list) == max_checkpoint_num + 1:
            idx_list = [int(name.replace('epoch', '').replace('.pth', '')) for name in checkpoint_list]
            min_idx = min(idx_list)
            os.remove(os.path.join(save_path, 'epoch' + str(min_idx) + '.pth'))
        
        


