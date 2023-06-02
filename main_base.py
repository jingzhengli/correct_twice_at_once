# Some of the code was forked from ***
import os
import time
import numpy as np
import math, random
import datetime
from collections import OrderedDict
import itertools
import gc
# os.chdir(os.path.dirname(__file__))
import matplotlib
matplotlib.use('Agg')
from dataset import get_data, get_dataloader, get_synthetic_idx, DATASETS_BIG, DATASETS_SMALL
from model_getter import get_model
from utils import *
# from PreResNet import *
import torch.backends.cudnn as cudnn
# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

PARAMS_META = {'cifar10'           :{'alpha':0.5, 'beta':2000, 'gamma':1, 'stage1':40,'stage2':120,'stage3':150,'k':10},
               'cifar100'          :{'alpha':0.5, 'beta':5000, 'gamma':1, 'stage1':40,'stage2':120, 'stage3':150,'k':10},
               'clothing1Mbalanced':{'alpha':0.1, 'beta':1000, 'gamma':1, 'stage1':2, 'stage2':15, 'stage3':20, 'k':10}}

def metacorrection(alpha, beta, gamma, stage1, stage2,stage3, K):
    def warmup_training():
        loss = criterion_cce(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    def meta_training():
        prior = torch.ones(NUM_CLASSES)/NUM_CLASSES
        prior = prior.to(device)
        pred_mean = torch.softmax(output, dim=1).mean(0)
        lo = torch.sum(prior*torch.log(prior/pred_mean))

        y_softmaxed = softmax(yy)
        lc = criterion_meta(output, y_softmaxed)
        net.zero_grad()
        grads_yy = torch.autograd.grad(lc, yy, create_graph=False, retain_graph=True, only_inputs=True)
        for grad in grads_yy:
            grad.detach()
        meta_grads = beta*grads_yy[0]
        yy.data.sub_(meta_grads)
        del meta_grads
        y_softmaxed = softmax(yy)

        lc2 = criterion_meta(output, y_softmaxed)  
        grads = torch.autograd.grad(lc2, net.parameters(), create_graph=True, retain_graph=True, only_inputs=True)
        for grad in grads:
            grad.detach()
        fast_weights = OrderedDict((name, param - 0.1*grad) for ((name, param), grad) in zip(net.named_parameters(), grads))  
        fast_out = net.forward(images_meta,fast_weights) 
        

        loss_meta = criterion_cce(fast_out, labels_meta)
        grads_yy2 = torch.autograd.grad(loss_meta, yy, create_graph=False, retain_graph=True, only_inputs=True)
        for grad in grads_yy2:
            grad.detach()
        meta_grads = beta*grads_yy2[0]

        # update labels
        yy.data.sub_(meta_grads)
        new_y[index,:] = yy.data.cpu().numpy()
        del grads, grads_yy, grads_yy2

        # training base network
        y_softmaxed = softmax(yy)
        lc = criterion_meta(output, y_softmaxed)                        # classification loss
        le = -torch.mean(torch.mul(softmax(output), logsoftmax(output)))# entropy loss
        # overall loss
        loss = lc + 0.1 * lo+ 0.1 * le
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss, meta_grads

    print('use_clean:{}, alpha:{}, beta:{}, gamma:{}, stage1:{}, stage2:{}, K:{}'.format(use_clean_data, alpha, beta, gamma, stage1, stage2, K))

    NUM_TRAINDATA = len(train_dataset)
    t_dataset, m_dataset, t_dataloader, m_dataloader = train_dataset, meta_dataset, train_dataloader, meta_dataloader
    # loss functions
    criterion_cce_without_softmax = lambda output, labels: -torch.mean(torch.sum(F.log_softmax(output, dim=1) * labels, dim=1))
    criterion_cce = nn.CrossEntropyLoss() 
    criterion_meta = lambda output, labels: torch.mean(softmax(output)*(logsoftmax(output+1e-10)-torch.log(labels+1e-10))) #KL-DIVER
    criterion_KL = lambda output, labels: torch.mean(labels*(torch.log(labels+1e-10)-softmax(output)))

    # initialize predicted labels with given labels multiplied with a constant K
    y_init_path = '{}/y_{}_{}_{}_{}.npy'.format(dataset,noise_type,noise_ratio,NUM_TRAINDATA,stage1)                                                               
    labels_yy = np.zeros(NUM_TRAINDATA)
    #store the noisy label
    if not os.path.exists(y_init_path):
        new_y = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
        for batch_idx, (images, labels, index) in enumerate(tqdm(t_dataloader)):
            onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), K).cpu().numpy()
            new_y[index, :] = onehot
    else:
        new_y = np.load(y_init_path)
    if not (clean_labels is None):
        data_lables=new_y.copy()
        clean_idx = np.where(np.argmax(data_lables, axis=1)==clean_labels)[0]
        noisy_idx = np.where(np.argmax(data_lables, axis=1)!=clean_labels)[0]
    print('end init new_y')
    test_acc_best = 0
    val_acc_best = 0
    top5_acc_best = 0
    top1_acc_best = 0
    epoch_best = 0

    model_s1_path = '{}/{}_{}_{}_{}_{}.pt'.format(dataset,dataset,noise_type,noise_ratio,NUM_TRAINDATA,stage1)
    if os.path.exists(model_s1_path):
        print("finetune--------------------")
        net.load_state_dict(torch.load(model_s1_path, map_location=device))

    for epoch in range(stage3): 
        start_epoch = time.time()
        train_accuracy = AverageMeter()
        train_loss = AverageMeter()
        train_accuracy_meta = AverageMeter()
        label_similarity = AverageMeter()

        lr = lr_scheduler(epoch)
        set_learningrate(optimizer, lr)
        net.train() 
        grads_dict = OrderedDict((name, 0) for (name, param) in net.named_parameters()) 
        # skip warm-up if there is a pretrained model already
        if os.path.exists(model_s1_path) and epoch < stage1:
            continue
        if epoch == stage1:
            if not os.path.exists(model_s1_path):
                torch.save(net.cpu().state_dict(), model_s1_path)
                net.to(device)
            if use_clean_data == 0:
                t_dataset, m_dataset, t_dataloader, m_dataloader = get_dataloaders_meta()
                NUM_TRAINDATA = len(t_dataset)
                labels_yy = np.zeros(NUM_TRAINDATA)
                new_y = np.zeros([NUM_TRAINDATA,NUM_CLASSES])
                for batch_idx, (images, labels,index) in enumerate(t_dataloader):
                    #index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
                    onehot = torch.zeros(labels.size(0), NUM_CLASSES).scatter_(1, labels.view(-1, 1), K).cpu().numpy()
                    new_y[index, :] = onehot
            t_meta_loader_iter = iter(m_dataloader)
        y_hat = new_y.copy()
        # meta_grads_yy_log = np.zeros((NUM_TRAINDATA,NUM_CLASSES))
        print(epoch)
        for batch_idx, (images, labels,index) in enumerate(tqdm(t_dataloader)):
            start = time.time()
            images, labels = images.to(device), labels.to(device)
            images, labels = torch.autograd.Variable(images), torch.autograd.Variable(labels)
            # predicted underlying true label distribution
            yy = torch.FloatTensor(y_hat[index,:]).to(device)
            yy = torch.autograd.Variable(yy,requires_grad = True)
            output = net(images)
            # Warm-up training
            if epoch < stage1:
                loss = warmup_training()
            # Noisy labels correction training
            elif epoch < stage2:
                try:
                    images_meta, labels_meta,_ = next(t_meta_loader_iter)
                except StopIteration:#yinwei
                    t_meta_loader_iter = iter(m_dataloader)
                    images_meta, labels_meta,_ = next(t_meta_loader_iter)
                    images_meta, labels_meta = images_meta[:labels.size(0)], labels_meta[:labels.size(0)]
                images_meta, labels_meta = images_meta.to(device), labels_meta.to(device)
                images_meta, labels_meta = torch.autograd.Variable(images_meta), torch.autograd.Variable(labels_meta)
                loss, meta_grads_yy = meta_training()
                # meta_grads_yy_log[index] = meta_grads_yy.cpu().detach().numpy()
            # Fine-tuning training
            else:
                y_softmaxed = softmax(yy)
                loss = criterion_meta(output, y_softmaxed)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            _, labels_yy[index] = torch.max(yy.cpu(), 1) # yy:onehot label mutilply a K over batch when warm up labels_yy: ground truth over batch
            _, predicted = torch.max(output.data, 1) # output:prediction over batch
            train_accuracy.update(predicted.eq(labels.data).cpu().sum().item(), labels.size(0)) #labels: labels_yy
            train_loss.update(loss.item())
            train_accuracy_meta.update(predicted.eq(torch.tensor(labels_yy[index]).long().to(device)).cpu().sum().item(), predicted.size(0))
            label_similarity.update(labels.eq(torch.tensor(labels_yy[index]).long().to(device)).cpu().sum().item(), labels.size(0))
            # keep log of gradients
            for tag, parm in net.named_parameters():
                grads_dict[tag] += parm.grad.data.cpu().numpy() * lr
            del yy

            if verbose == 2:
                template = "Progress: {:6.5f}, Accuracy: {:5.4f}, Accuracy Meta: {:5.4f}, Loss: {:5.4f}, Process time:{:5.4f}   \r"
                sys.stdout.write(template.format(batch_idx*BATCH_SIZE/NUM_TRAINDATA, train_accuracy.percentage, train_accuracy_meta.percentage, train_loss.avg, time.time()-start))
        if verbose == 2:
            sys.stdout.flush()           

        if SAVE_LOGS == 1:
            np.save(log_dir + "y.npy", new_y)
        # evaluate on validation and test data
        val_accuracy, val_loss, _, _ = evaluate(net, m_dataloader, criterion_cce)
        test_accuracy, test_loss, idx_top5, accs_top5 = evaluate(net, test_dataloader, criterion_cce)
        if val_accuracy > val_acc_best: 
            val_acc_best = val_accuracy
            test_acc_best = test_accuracy
            top5_acc_best = accs_top5.mean()
            top1_acc_best = accs_top5.max()
            epoch_best = epoch

        if SAVE_LOGS == 1:
            summary_writer.add_scalar('train_loss', train_loss.avg, epoch)
            summary_writer.add_scalar('test_loss', test_loss, epoch)
            summary_writer.add_scalar('train_accuracy', train_accuracy.percentage, epoch)
            summary_writer.add_scalar('test_accuracy', test_accuracy, epoch)
            summary_writer.add_scalar('test_accuracy_best', test_acc_best, epoch)
            summary_writer.add_scalar('val_loss', val_loss, epoch)
            summary_writer.add_scalar('val_accuracy', val_accuracy, epoch)
            summary_writer.add_scalar('top5_accuracy', accs_top5.mean(), epoch)
            summary_writer.add_scalar('top1_accuracy', accs_top5.max(), epoch)
            summary_writer.add_scalar('val_accuracy_best', val_acc_best, epoch)
            summary_writer.add_scalar('label_similarity', label_similarity.percentage, epoch)


        if verbose > 0:
            template = 'Epoch {}, Accuracy(train,meta_train,val,test): {:3.1f}/{:3.1f}/{:3.1f}/{:3.1f}, Accuracy(top5,top1): {:3.1f}/{:3.1f}, Loss(train,val,test): {:4.3f}/{:4.3f}/{:4.3f}, Label similarity: {:6.3f}, Learning rate(lr,yy): {}/{}, Time: {:3.1f}({:3.2f})'
            print(template.format(epoch + 1, 
                                train_accuracy.percentage, train_accuracy_meta.percentage, val_accuracy, test_accuracy,
                                accs_top5.mean(), accs_top5.max(),
                                train_loss.avg, val_loss, test_loss,  
                                label_similarity.percentage, lr, int(beta),
                                time.time()-start_epoch, (time.time()-start_epoch)/3600))

    print('{}({}): Train acc: {:3.1f}, Validation acc: {:3.1f}-{:3.1f}, Test acc: {:3.1f}-{:3.1f}, Top5 acc: {:3.1f}-{:3.1f}, Top1 acc: {:3.1f}-{:3.1f}, Best epoch: {}, Num meta-data: {}'.format(
        noise_type, noise_ratio, train_accuracy.percentage, val_accuracy, val_acc_best, test_accuracy, test_acc_best, accs_top5.mean(), top5_acc_best, accs_top5.max(), top1_acc_best, epoch_best, NUM_METADATA))
    if SAVE_LOGS == 1:
        summary_writer.close()
        # write log for hyperparameters
        hp_writer.add_hparams({'alpha':alpha, 'beta': beta, 'gamma':gamma, 'stage1':stage1, 'K':K, 'use_clean':use_clean_data, 'num_meta':NUM_METADATA}, 
                              {'val_accuracy': val_acc_best, 'test_accuracy': test_acc_best, 'top5_acc':top5_acc_best, 'top1_acc':top1_acc_best, 'epoch_best':epoch_best})
        hp_writer.close()
        torch.save(net.state_dict(), os.path.join(log_dir, 'saved_model.pt'))

def get_dataloaders_meta():
    NUM_TRAINDATA = len(train_dataset)
    num_meta_data_per_class = int(NUM_METADATA/NUM_CLASSES)
    idx_meta = None
    
    loss_values = np.zeros(NUM_TRAINDATA)
    label_values = np.zeros(NUM_TRAINDATA)
    
    c = nn.CrossEntropyLoss(reduction='none').to(device)
    for batch_idx, (images, labels,index) in enumerate(train_dataloader):
        images, labels = images.to(device), labels.to(device)
        #index = np.arange(batch_idx*BATCH_SIZE, (batch_idx)*BATCH_SIZE+labels.size(0))
        output = net(images)
        loss = c(output, labels)
        loss_values[index] = loss.detach().cpu().numpy()
        label_values[index] = labels.cpu().numpy()
    for i in range(NUM_CLASSES):
        idx_i = label_values == i
        idx_i = np.where(idx_i == True)
        loss_values_i = loss_values[idx_i]
        sorted_idx = np.argsort(loss_values_i)
        anchor_idx_i = np.take(idx_i, sorted_idx[:num_meta_data_per_class])
        if idx_meta is None:
            idx_meta = anchor_idx_i
        else:
            idx_meta = np.concatenate((idx_meta,anchor_idx_i))
    idx_train = np.setdiff1d(np.arange(NUM_TRAINDATA),np.array(idx_meta))

    t_dataset = torch.utils.data.Subset(train_dataset, idx_train)
    m_dataset = torch.utils.data.Subset(train_dataset, idx_meta)
    t_dataloader = torch.utils.data.DataLoader(t_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers)
    m_dataloader = torch.utils.data.DataLoader(m_dataset,batch_size=BATCH_SIZE,shuffle=False, num_workers=num_workers, drop_last=True)
    return t_dataset, m_dataset, t_dataloader, m_dataloader

def get_topk(arr, percent=0.01):
    arr_flat = arr.flatten()
    arr_len = int(len(arr_flat)*percent)
    idx = np.argsort(np.absolute(arr_flat))[-arr_len:]
    return arr_flat[idx]

def set_learningrate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_model():
    model = ResNet18(num_classes=NUM_CLASSES)
    model = model.cuda()
    return model

def evaluate(net, dataloader, criterion):
    eval_accuracy = AverageMeter()
    eval_loss = AverageMeter()

    topks = {}
    for i in range(NUM_CLASSES):
        topks[i] = AverageMeter()

    net.eval()
    with torch.no_grad():
        for (inputs, targets, _) in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs) 
            loss = criterion(outputs, targets) 
            _, predicted = torch.max(outputs.data, 1) 
            eval_accuracy.update(predicted.eq(targets.data).cpu().sum().item(), targets.size(0)) 
            eval_loss.update(loss.item())
            for i in range(NUM_CLASSES):
                idx = targets == i
                topks[i].update(predicted[idx].eq(targets[idx].data).cpu().sum().item(), idx.sum().item())  
    # get best 10 accuracies
    accs_per_class = np.array([topks[i].percentage for i in range(NUM_CLASSES)])
    idx = np.argsort(np.absolute(accs_per_class))[-5:]
    return eval_accuracy.percentage, eval_loss.avg, idx, accs_per_class[idx]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', required=False, type=str, default='cifar10',
        help="'cifar10', 'cifar100','clothing1Mbalanced'")
    parser.add_argument('-n', '--noise_type', required=False, type=str, default='symmetric',
        help="Noise type for cifar10: 'feature-dependent', 'symmetric', 'class-dependent'")
    parser.add_argument('-r', '--noise_ratio', required=False, type=int, default=40,
        help="Synthetic noise ratio in percentage between 0-100")
    parser.add_argument('-s', '--batch_size', required=False, type=int,
        help="Number of gpus to be used")
    parser.add_argument('-i', '--gpu_ids', required=False, type=int, nargs='+', action='append',
        help="GPU ids to be used")
    parser.add_argument('-f', '--folder_log', required=False, type=str,
        help="Folder name for logs")
    parser.add_argument('-v', '--verbose', required=False, type=int, default=0,
        help="Details of prints: 0(silent), 1(not silent)")
    parser.add_argument('-w', '--num_workers', required=False, type=int,
        help="Number of parallel workers to parse dataset")
    parser.add_argument('--save_logs', required=False, type=int, default=1,
        help="Either to save log files (1) or not (0)")
    parser.add_argument('--seed', required=False, type=int, default=42,
        help="Random seed to be used in simulation")
    
    parser.add_argument('-c', '--clean_data', required=False, type=int, default=1,
        help="Either to use available clean data (1) or not (0)")
    parser.add_argument('-m', '--metadata_num', required=False, type=int, default=4999,
        help="Number of samples to be used as meta-data")

    parser.add_argument('-a', '--alpha', required=False, type=float,
        help="Learning rate for meta iteration")
    parser.add_argument('-b', '--beta', required=False, type=float,
        help="Beta paramter")
    parser.add_argument('-g', '--gamma', required=False, type=float,
        help="Gamma paramter")
    parser.add_argument('-s1', '--stage1', required=False, type=int,
        help="Epoch num to end stage1 (straight training)")
    parser.add_argument('-s2', '--stage2', required=False, type=int,
        help="Epoch num to end stage2 (meta training)")
    parser.add_argument('-s3', '--stage3', required=False, type=int,
        help="Epoch num to end stage3 (meta training)")        
    parser.add_argument('-k', required=False, type=int, default=10,
        help="")

    args = parser.parse_args()
    #set default variables if they are not given from the command line
    if args.alpha == None: args.alpha = PARAMS_META[args.dataset]['alpha']
    if args.beta == None: args.beta = PARAMS_META[args.dataset]['beta']
    if args.gamma == None: args.gamma = PARAMS_META[args.dataset]['gamma']
    if args.stage1 == None: args.stage1 = PARAMS_META[args.dataset]['stage1']
    if args.stage2 == None: args.stage2 = PARAMS_META[args.dataset]['stage2']
    if args.stage3 == None: args.stage3 = PARAMS_META[args.dataset]['stage3']
    if args.k == None: args.k = PARAMS_META[args.dataset]['k']
    # configuration variables
    framework = 'pytorch'
    dataset = args.dataset
    model_name = 'OURS'
    noise_type = args.noise_type
    noise_ratio = args.noise_ratio/100
    BATCH_SIZE = args.batch_size if args.batch_size != None else PARAMS[dataset]['batch_size']
    NUM_CLASSES = PARAMS[dataset]['num_classes']
    SAVE_LOGS = args.save_logs
    use_clean_data = args.clean_data
    verbose = args.verbose
    #****************************************************************************************************************
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    if args.gpu_ids is None:
        ngpu = torch.cuda.device_count() if device.type == 'cuda' else 0  
        gpu_ids = list(range(ngpu)) 
    else:
        gpu_ids = args.gpu_ids[0]
        ngpu = len(gpu_ids)
        if ngpu == 1: 
            device = torch.device("cuda:{}".format(gpu_ids[0]))
        
    if args.num_workers is None:
        num_workers = 2 if ngpu < 2 else ngpu*2
    else:
        num_workers = args.num_workers
    # ****************************************************************************************************************
    #num_workers=0
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # create necessary folders
    create_folder('{}/dataset'.format(dataset))
    # global variables
    train_dataset, val_dataset, test_dataset, meta_dataset, class_names = get_data(dataset,framework,noise_type,noise_ratio,args.seed,args.metadata_num,0)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=1024,shuffle=False)
    meta_dataloader = torch.utils.data.DataLoader(meta_dataset,batch_size=BATCH_SIZE,shuffle=True, drop_last=True)
    NUM_METADATA = len(meta_dataset)
    _, clean_labels = get_synthetic_idx(dataset,args.seed,args.metadata_num,0,noise_type,noise_ratio,) 
    print('| Building net')
    net = get_model(dataset,framework).to(device)
    lr_scheduler = get_lr_scheduler(dataset)
    optimizer = optim.SGD(net.parameters(), lr=lr_scheduler(0), momentum=0.9, weight_decay=1e-4)
    logsoftmax = nn.LogSoftmax(dim=1).to(device)
    softmax = nn.Softmax(dim=1).to(device)
  
    print("Dataset: {}, Model: {}, Device: {}, Batch size: {}, #GPUS to run: {}".format(dataset, model_name, device, BATCH_SIZE, ngpu))
    if dataset in DATASETS_SMALL:
        print("Noise type: {}, Noise ratio: {}".format(noise_type, noise_ratio))

    # if logging
    if SAVE_LOGS == 1:
        base_folder = model_name if dataset in DATASETS_BIG else noise_type + '/' + str(args.noise_ratio) + '/' + model_name
        log_folder = args.folder_log if args.folder_log else 'c{}_a{}_b{}_g{}_s{}_m{}_main_base_{}'.format(use_clean_data, args.alpha, args.beta, args.gamma, args.stage1, NUM_METADATA, current_time)
        log_base = '{}/logs/{}/'.format(dataset, base_folder)
        log_dir = log_base + log_folder + '/'
        log_dir_hp = '{}/logs_hp/{}/'.format(dataset, base_folder)
        create_folder(log_dir)
        summary_writer = SummaryWriter(log_dir)
        create_folder(log_dir_hp)
        hp_writer = SummaryWriter(log_dir_hp)
    
    start_train = time.time()
    metacorrection(args.alpha, args.beta, args.gamma, args.stage1, args.stage2, args.stage3, args.k)
    print('Total training duration: {:3.2f}h'.format((time.time()-start_train)/3600))