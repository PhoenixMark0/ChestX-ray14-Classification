import os
import time
import shutil
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics.ranking import roc_auc_score

import torch
import torch.nn as nn
import torchvision.models as models

import resnet_wider
import densenet


# ---------------------------------------Classification model------------------------------------
def Classifier_model(arch_name, num_class, conv=None, weight=None, linear_classifier=False, sobel=False,
                     activation=None):
    if weight is None:
        weight = "none"

    if conv is None:
        try:
            model = resnet_wider.__dict__[arch_name](sobel=sobel)
        except:
            model = models.__dict__[arch_name](pretrained=False)
    else:
        if arch_name.lower().startswith("resnet"):
            model = resnet_wider.__dict__[arch_name + "_layerwise"](conv, sobel=sobel)
        elif arch_name.lower().startswith("densenet"):
            model = densenet.__dict__[arch_name + "_layerwise"](conv)

    if arch_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        if activation is None:
            model.fc = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

        if linear_classifier:
            for name, param in model.named_parameters():
                if name not in ['fc.0.weight', 'fc.0.bias', 'fc.weight', 'fc.bias']:
                    param.requires_grad = False

        # init the fc layer
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        kernelCount = model.classifier.in_features
        if activation is None:
            model.classifier = nn.Linear(kernelCount, num_class)
        elif activation == "Sigmoid":
            model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())

        if linear_classifier:
            for name, param in model.named_parameters():
                if name not in ['classifier.0.weight', 'classifier.0.bias', 'classifier.weight', 'classifier.bias']:
                    param.requires_grad = False

        # init the classifier layer
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)
            model.classifier[0].bias.data.zero_()

    def _weight_loading_check(_arch_name, _activation, _msg):
        if len(_msg.missing_keys) != 0:
            if _arch_name.lower().startswith("resnet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"fc.weight", "fc.bias"}
                else:
                    assert set(_msg.missing_keys) == {"fc.0.weight", "fc.0.bias"}
            elif _arch_name.lower().startswith("densenet"):
                if _activation is None:
                    assert set(_msg.missing_keys) == {"classifier.weight", "classifier.bias"}
                else:
                    assert set(_msg.missing_keys) == {"classifier.0.weight", "classifier.0.bias"}

    state_dict = None
    if weight.lower() == "random" or weight.lower() == "none":
        state_dict = model.state_dict()

    if weight.lower() == "imagenet":
        pretrained_model = models.__dict__[arch_name](pretrained=True)
        state_dict = pretrained_model.state_dict()

        # delete fc layer
        for k in list(state_dict.keys()):
            if k.startswith('fc') or k.startswith('classifier'):
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        _weight_loading_check(arch_name, activation, msg)
        print("=> loaded ImageNet pre-trained model")

    # reinitialize fc layer again
    if arch_name.lower().startswith("resnet"):
        if activation is None:
            model.fc.weight.data.normal_(mean=0.0, std=0.01)
            model.fc.bias.data.zero_()
        else:
            model.fc[0].weight.data.normal_(mean=0.0, std=0.01)
            model.fc[0].bias.data.zero_()
    elif arch_name.lower().startswith("densenet"):
        if activation is None:
            model.classifier.weight.data.normal_(mean=0.0, std=0.01)
            model.classifier.bias.data.zero_()
        else:
            model.classifier[0].weight.data.normal_(mean=0.0, std=0.01)

    return model, state_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def computeAUROC(dataGT, dataPRED, classCount=14):
    outAUROC = []

    datanpGT = dataGT.cpu().numpy()
    datanpPRED = dataPRED.cpu().numpy()

    for i in range(classCount):
        outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))

    return outAUROC


def save_checkpoint(state, is_best, filename='model'):
    torch.save(state, filename + '_checkpoint.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_checkpoint.pth.tar', filename + '.pth.tar')


# ----------------------------------Whether Experiment Exist----------------------------------
def experiment_exist(log_file, exp_name):
    if not os.path.isfile(log_file):
        return False

    with open(log_file, 'r') as f:
        line = f.readline()
        while line:
            # print(line)
            # if line.replace('\n', '') == exp_name:
            if line.startswith(exp_name):
                return True
            line = f.readline()

    return False


# ----------------------------------Get Pretrained Weight------------------------------------
def get_weight_name(log_file, idx, wait_time=0):
    weight_name = None

    while (True):
        # print(log_file)
        if os.path.isfile(log_file):
            with open(log_file, 'r') as f:
                for i in range(idx + 1):
                    line = f.readline()
                    if not line:
                        break
                if line:
                    line = line.replace('\n', '')
                    weight_name = line
                    # if line.endswith(str(idx)):
                    #   weight_name = line
                    #   break

        if weight_name is not None or wait_time == 0:
            break
        else:
            time.sleep(wait_time * 60.)

    return weight_name


# ---------------------------Callback function for OptionParser-------------------------------
def vararg_callback_bool(option, opt_str, value, parser):
    assert value is None

    arg = parser.rargs[0]
    if arg.lower() in ('yes', 'true', 't', 'y', '1'):
        value = True
    elif arg.lower() in ('no', 'false', 'f', 'n', '0'):
        value = False

    del parser.rargs[:1]
    setattr(parser.values, option.dest, value)


def vararg_callback_int(option, opt_str, value, parser):
    assert value is None
    value = []

    def intable(str):
        try:
            int(str)
            return True
        except ValueError:
            return False

    for arg in parser.rargs:
        # stop on --foo like options
        if arg[:2] == "--" and len(arg) > 2:
            break
        # stop on -a, but not on -3 or -3.0
        if arg[:1] == "-" and len(arg) > 1 and not intable(arg):
            break
        value.append(int(arg))

    del parser.rargs[:len(value)]
    setattr(parser.values, option.dest, value)


def display_args(args):
    """Display Configuration values."""
    print("\nConfigurations:")
    for a in dir(args):
        if not a.startswith("__") and not callable(getattr(args, a)):
            print("{:30} {}".format(a, getattr(args, a)))
    print("\n")


if __name__ == '__main__':
    weight = "/mnt/.nfs/ruibinf/MoCo/chestxray_pretrain/checkpoint_0199.pth.tar"
    model = Classifier_model("resnet18", 14, weight=weight, linear_classifier=False)
    print(model)
