import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
#from tqdm import tqdm
#import shutil
import time
from config import get_args, get_logger
from model import ResNet50, ResNet38, ResNet26
from preprocess import load_data
import file_writer

'''
TODO:
1. Use learning rate decay on optimizer (they did this in paper)
3. Run main function with several different hyper parameters
4. Need to save state of optimizer if reloading with momentum or Adam


Early stopping
'''

#TODO Make training pipeline nice like in RL where logs go to different dir for each run.
#TODO Employ early stopping as well as allowing multiple experiments in one run.
#TODO Add ability for multi gpu
#TODO Add dynamic attention span (span depends also on current input and layer)
def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #After 60% of we trim by factor args.decay_factor and at 80% we do this again
    if epoch == int(0.6*float(args.epochs)) or epoch == int(0.8*float(args.epochs)):
        for param_group in optimizer.param_groups:
            param_group['lr'] *= args.decay_factor


def train(model, train_loader, optimizer, criterion, epoch, args, logger, device):
    model.train()

    train_acc = 0.0
    step = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        span_loss = model.get_span_l1(args)
        loss = criterion(output, target) + args.span_penalty * span_loss
        loss.backward()
        optimizer.step()

        y_pred = output.data.max(1)[1]

        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.
        train_acc += acc
        step += 1
        if step % args.print_interval == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc))
            #logger.info("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc))


def eval(model, test_loader, args, is_valid=True, device=None):
    print('evaluation ...')
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            prediction = output.data.max(1)[1]
            correct += prediction.eq(target.data).sum()

    acc = 100. * float(correct) / len(test_loader.dataset)
    data_set = 'Validation' if is_valid else "Test"
    print(data_set + ' acc: {0:.2f}'.format(acc))
    return acc


def get_model_parameters(model):
    total_parameters = 0
    for layer in list(model.parameters()):
        layer_parameter = 1
        for l in list(layer.size()):
            layer_parameter *= l
        total_parameters += layer_parameter
    return total_parameters


def main(args, logger):
    train_loader, valid_loader, test_loader = load_data(args)
    num_classes = None
    if args.dataset == 'CIFAR10':
        num_classes = 10
    elif args.dataset == 'CIFAR100':
        num_classes = 100
    elif args.dataset == 'IMAGENET':
        num_classes = 1000

    print('img_size: {}, num_classes: {}'.format(args.img_size, num_classes))
    model = None
    print('ARGS: ', args)
    if args.model_name == 'ResNet26':
        print('Model Name: {0}'.format(args.model_name))
        model = ResNet26(num_classes=num_classes, args=args)
    elif args.model_name == 'ResNet38':
        print('Model Name: {0}'.format(args.model_name))
        model = ResNet38(num_classes=num_classes, all_attention=args.all_attention)
    elif args.model_name == 'ResNet50':
        print('Model Name: {0}'.format(args.model_name))
        model = ResNet50(num_classes=num_classes, all_attention=args.all_attention)

    if args.use_adam:
        optimizer = optim.Adam(model.parameters(), lr=args.adam_lr)  # Try altering initial settings of Adam later.
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay, nesterov=True)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                           T_max=args.T_max,
                                                           eta_min=args.eta_min)

    start_epoch = 1
    best_acc = 0.0
    best_epoch = 1

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.pretrained_model or args.test:
        filename = args.xpid + '_model_' + str(args.dataset) + '_' + str(args.model_name)  + '_ckpt.tar'
        print('filename :: ', filename)

        map_location = 'cuda' if args.cuda else None
        checkpoint = torch.load(filename, map_location=map_location)

        model.load_state_dict(checkpoint['state_dict'])

        model = model.to(device)

        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])

        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        best_epoch = start_epoch
        model_parameters = checkpoint['parameters']
        print('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))
        #logger.info('Load model, Parameters: {0}, Start_epoch: {1}, Acc: {2}'.format(model_parameters, start_epoch, best_acc))

        if args.test:
            #Compute test accuracy

            test_acc = eval(model, test_loader, args, is_valid=False)
            print('TEST ACCURACY: ',test_acc)
            return

    if not args.pretrained_model:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model = model.to(device)

    print("Number of model parameters: ", get_model_parameters(model))
    #logger.info("Number of model parameters: {0}".format(get_model_parameters(model)))

    filename = args.xpid + '_model_' + str(args.dataset) + '_' + str(args.model_name) + '_ckpt.tar'
    plogger = file_writer.FileWriter(
        xpid=args.xpid, rootdir=os.path.dirname(os.path.abspath(__file__))
    )
    print('will save model as filename :: ', filename)

    criterion = nn.CrossEntropyLoss()


    for epoch in range(start_epoch, args.epochs + 1):

        if args.all_attention or args.attention_conv:
            if epoch < args.warmup_epochs:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * (epoch + 1) / args.warmup_epochs

            elif epoch >= args.start_scheduler:
                scheduler.step()

        else:
            adjust_learning_rate(optimizer, epoch, args)
        learning_rate = optimizer.param_groups[0]['lr']
        # learning_rate = [x['lr'] for x in optimizer.param_groups]
        print('Updated lr: ', learning_rate)
        start_time = time.time()
        train(model, train_loader, optimizer, criterion, epoch, args, logger, device)
        print('Epoch took: ', time.time()-start_time)

        eval_acc = eval(model, valid_loader, args, is_valid=True, device=device)
        to_log = dict(accuracy=eval_acc, learning_rate=learning_rate)
        plogger.log(to_log)

        is_best = eval_acc > best_acc
        best_acc = max(eval_acc, best_acc)
        if is_best:
            best_epoch = epoch
        elif epoch - best_epoch > 10:
            print('EARLY STOPPING')
            break

        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')

        parameters = get_model_parameters(model)

        if is_best:
            print('Saving best model')
            state = {
                'epoch': epoch,
                'arch': args.model_name,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'parameters': parameters,
                }
            torch.save(state,filename)
    plogger.close()





if __name__ == '__main__':
    args, logger = get_args()
    print('ARGS: ', args)
    main(args, logger)

    #Run on test set
    args.test = True
    main(args, logger)
