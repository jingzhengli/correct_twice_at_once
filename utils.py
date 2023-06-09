import os
import datetime

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
PARAMS = {'mnist_fashion'     :{'epochs':25, 'batch_size':32, 'num_classes':10},
          'cifar10'           :{'epochs':120, 'batch_size':128, 'num_classes':10},
          'cifar100'          :{'epochs':80, 'batch_size':128, 'num_classes':100},
          'clothing1M'        :{'epochs':100, 'batch_size':64, 'num_classes':14},
          'clothing1M50k'     :{'epochs':10, 'batch_size':32, 'num_classes':14},
          'clothing1Mbalanced':{'epochs':15, 'batch_size':32, 'num_classes':14},
          'food101N'          :{'epochs':15, 'batch_size':32, 'num_classes':101}}#一开始是32

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_lr_scheduler(dataset):
    if dataset == 'mnist_fashion':
        def lr_scheduler_mnist(epoch):
            return 0.01
        return lr_scheduler_mnist
    if dataset == 'cifar10':
        def lr_scheduler_cifar10(epoch):
            if epoch > 80:
            #if epoch > 80:
                return 0.001
            elif epoch > 50:
            #elif epoch > 40:
                return 0.01
            else:
                #return 0.01
                return 0.1
            # if epoch > 100:
            # #if epoch > 80:
            #     return 0.003
            # elif epoch > 40:
            # #elif epoch > 40:
            #     return 0.03
            # else:
            #     #return 0.01
            #     return 0.3
        return lr_scheduler_cifar10
    if dataset == 'cifar100':
        def lr_scheduler_cifar100(epoch):
            # if epoch > 80:
            # #if epoch > 80:
            #     return 0.0001
            # elif epoch > 40:
            # #elif epoch > 40:
            #     return 0.001
            # else:
            #     #return 0.01
            #     return 0.1
            # if epoch > 100:
            # #if epoch > 80:
            #     return 0.1
            # elif epoch > 40:
            # #elif epoch > 40:
            #     return 0.01
            # else:
            #     #return 0.01
            #     return 0.1
            if epoch > 50:
            #if epoch > 80:
                return 0.001
            elif epoch > 30:
            #elif epoch > 40:
                return 0.01
            else:
                #return 0.01
                return 0.1
        return lr_scheduler_cifar100
    elif dataset == 'clothing1M' or dataset == 'clothing1M50k' or dataset == 'clothing1Mbalanced' or dataset == 'food101N':
        def lr_scheduler_clothing1M(epoch):
            if epoch < 5:
                return 1e-3
            elif epoch < 50:
                return 1e-4
            else:
                return 1e-5
            # if epoch < 2:
            #     return 1e-2
            # elif epoch < 7:
            #     return 1e-3
            # else:
            #     return 1e-4
        return lr_scheduler_clothing1M

def get_para_scheduler(stage1, stage2,epoch):
    stage2_duration = stage2 - stage1
    delta = (stage2_duration - (epoch - stage1)) / stage2_duration
    return delta

def get_meta_lr_scheduler(lr_type, stage1, stage2, lambda1, lambda2):
    if lr_type == 'constant':
        def lr_scheduler(epoch):
            return lambda1
        return lr_scheduler
    elif lr_type == 'linear_decrease':
        stage2_duration = stage2 - stage1
        def lr_scheduler(epoch):
            coeff = (stage2_duration - (epoch - stage1 - 1)) / stage2_duration
            return lambda1 * coeff
        return lr_scheduler
    elif lr_type == 'two_phase':
        stage2_duration = stage2 - stage1
        half_duration = stage2_duration/2
        def lr_scheduler(epoch):
            if epoch - (stage1-1) < half_duration:
                return lambda1
            else:
                return lambda2
        return lr_scheduler

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0
        self.percentage = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
        self.percentage = self.avg*100