# Dataset NIH, classification

import os
import sys
import shutil
import time
import numpy as np
from optparse import OptionParser
from shutil import copyfile
from tqdm import tqdm

from model import experiment_exist, get_weight_name, vararg_callback_bool, vararg_callback_int
from dataloader import Augmentation, ChestX_ray14
from model import Classifier_model, AverageMeter, ProgressMeter, computeAUROC, \
  save_checkpoint

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.setrecursionlimit(40000)

parser = OptionParser()

parser.add_option("--GPU", dest="GPU", help="the index of gpu is used", default=None, action="callback",
                  callback=vararg_callback_int)
# network architecture
parser.add_option("--model", dest="model_name", help="DenseNet121", default="DenseNet121", type="string")
parser.add_option("--init", dest="init", help="Random | ImageNet | MoCo | SimCLR | C2L",
                  default="Random", type="string")
parser.add_option("--num_class", dest="num_class", help="number of the classes in the downstream task",
                  default=14, type="int")
# data loader
parser.add_option("--data_set", dest="data_set", help="ChestX-ray14", default="ChestX-ray14", type="string")
parser.add_option("--normalization", dest="normalization", help="how to normalize data", default="default",
                  type="string")
parser.add_option("--augment", dest="augment", help="full", default="full", type="string")
parser.add_option("--img_size", dest="img_size", help="input image resolution", default=224, type="int")
parser.add_option("--img_depth", dest="img_depth", help="num of image depth", default=3, type="int")
parser.add_option("--train_list", dest="train_list", help="file for training list",
                  default=None, type="string")
parser.add_option("--val_list", dest="val_list", help="file for validating list",
                  default=None, type="string")
parser.add_option("--test_list", dest="test_list", help="file for test list",
                  default=None, type="string")
# training detalis
parser.add_option("--linear_classifier", dest="linear_classifier", help="whether train a linear classifier",
                  default=False, action="callback", callback=vararg_callback_bool)
parser.add_option("--sobel", dest="sobel", help="Sobel filtering", default=False, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--mode", dest="mode", help="train | test | valid", default="train", type="string")
parser.add_option("--batch_size", dest="batch_size", help="batch size", default=32, type="int")
parser.add_option("--num_epoch", dest="num_epoch", help="num of epoches", default=1000, type="int")
parser.add_option("--optimizer", dest="optimizer", help="Adam | SGD", default="Adam", type="string")
parser.add_option("--lr", dest="lr", help="learning rate", default=2e-4, type="float")
parser.add_option("--lr_Scheduler", dest="lr_Scheduler", help="learning schedule", default=None, type="string")
parser.add_option("--patience", dest="patience", help="num of patient epoches", default=10, type="int")
parser.add_option("--early_stop", dest="early_stop", help="whether use early_stop", default=True, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--trial", dest="num_trial", help="number of trials", default=1, type="int")
parser.add_option("--start_index", dest="start_index", help="the start model index", default=0, type="int")
parser.add_option("--clean", dest="clean", help="clean the existing data", default=False, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--resume", dest="resume", help="whether latest checkpoint", default=False, action="callback",
                  callback=vararg_callback_bool)
parser.add_option("--workers", dest="workers", help="number of CPU workers", default=8, type="int")
parser.add_option("--print_freq", dest="print_freq", help="print frequency", default=50, type="int")
parser.add_option("--test_augment", dest="test_augment", help="whether use test time augmentation",
                  default=True, action="callback", callback=vararg_callback_bool)
# pretrained weights
parser.add_option("--proxy_dir", dest="proxy_dir", help="Pretrained model folder", default=None, type="string")
parser.add_option("--proxy_idx", dest="proxy_idx", help="Pretrained model index", default=0, type="int")

(options, args) = parser.parse_args()

if options.GPU is not None:
  os.environ["CUDA_VISIBLE_DEVICES"] = str(options.GPU)[1:-1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

assert options.init in ['Random',
                        'ImageNet',
                        'MoCo',
                        'SimCLR',
                        'C2L',
                        'P2W',
                        'SSL_Transfer'
                        ]

model_path = "../Models/ChestXray14/"
if not os.path.exists(model_path):
  os.makedirs(model_path)
output_path = "../Outputs/ChestXray14/"
if not os.path.exists(output_path):
  os.makedirs(output_path)


class setup_config():
  def __init__(self,
               model_name=None,
               init=None,
               dense_unit=None,
               num_class=None,
               data_set=None,
               linear_classifier=False,
               sobel=False,
               normalization="default",
               augment=None,
               num_meta=None,
               train_list=None,
               val_list=None,
               test_list=None,
               img_size=224,
               img_depth=3,
               batch_size=32,
               num_epoch=1000,
               optimizer=None,
               lr=0.001,
               lr_Scheduler=None,
               early_stop=True,
               patience=10,
               proxy_dir=None,
               proxy_idx=0
               ):
    self.model_name = model_name
    self.dense_unit = dense_unit
    self.num_class = num_class
    self.init = init
    self.exp_name = self.model_name + "_" + self.init
    self.data_set = data_set
    self.augment = augment
    self.img_size = img_size
    self.img_depth = img_depth
    self.batch_size = batch_size
    self.num_epoch = num_epoch
    self.optimizer = optimizer
    self.lr = lr
    self.lr_Scheduler = lr_Scheduler
    self.early_stop = early_stop
    self.patience = patience
    if linear_classifier:
      self.linear_classifier = linear_classifier
    if sobel:
      self.sobel = sobel

    self.data_dir = "../../../Data/ChestX-ray14/images"
    self.class_name = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
                       'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                       'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    self.num_meta = num_meta
    self.train_list = train_list
    self.val_list = val_list
    self.test_list = test_list

    self.activate = "Sigmoid"
    if self.init == "ModelGenesis":
      self.proxy_dir = os.path.join("../Models/ModelGenesis", proxy_dir)
      self.proxy_idx = proxy_idx
    if self.init == "MoCo":
      self.proxy_dir = os.path.join("../Models/MoCo", proxy_dir)
      self.proxy_idx = proxy_idx
    if self.init == "SimCLR":
      self.proxy_dir = os.path.join("../Models/SimCLR", proxy_dir)
      self.proxy_idx = proxy_idx
    if self.init == "C2L":
      self.proxy_dir = os.path.join("../Models/C2L", proxy_dir)
      self.proxy_idx = proxy_idx
    if self.init == "P2W":
      self.proxy_dir = os.path.join("../Models/Part2Whole", proxy_dir)
      self.proxy_idx = proxy_idx
    if self.init == "SSL_Transfer":
      self.proxy_dir = os.path.join("../Models/SSL_Transfer", proxy_dir)
      self.proxy_idx = proxy_idx

    if normalization == "default":
      if self.init.lower() == "random" or self.init.lower() == "part2whole" or self.init.lower() == "moco":
        self.normalization = "chestx-ray"
      elif self.init.lower() == "imagenet" or self.init.lower() == "c2l":
        self.normalization = "imagenet"
      elif self.init.lower() == "modelgenesis":
        self.normalization = "none"
      elif self.init.lower() == "transvw":
        self.normalization = "none"
      elif self.init.lower() == "simclr":  # SimCLR does not use normalization
        self.normalization = "none"
    else:
      self.normalization = normalization

  def display(self):
    """Display Configuration values."""
    print("\nConfigurations:")
    for a in dir(self):
      if not a.startswith("__") and not callable(getattr(self, a)):
        print("{:30} {}".format(a, getattr(self, a)))
    print("\n")

  def log_config(self, model_path, clean=False):
    """Log Configuration values."""
    dup_file = self.check_duplicate(model_path)
    if not clean and dup_file is not None:
      print("Experiment Exists!")
      return dup_file

    if clean and dup_file is not None:
      print("Experiment Exists!")
      print("Delete existing experiment...")
      os.remove(os.path.join(model_path, dup_file + ".log"))
      shutil.rmtree(os.path.join(model_path, dup_file), ignore_errors=True)

    # log configuration
    time_string = time.strftime("%Y%m%d%H%M%S", time.localtime())
    config_log = self.exp_name + '_' + time_string
    os.makedirs(os.path.join(model_path, config_log))
    with open(os.path.join(model_path, config_log + '.log'), "w+") as f:
      f.write("Configurations: \n")
      for a in dir(self):
        if not a.startswith("__") and not callable(getattr(self, a)):
          f.write("{:30} {} \n".format(a, getattr(self, a)))
      f.close()
    return config_log

  def print_config(self, file_path):
    """Log Configuration values."""

    with open(file_path, "w+") as f:
      f.write("Configurations: \n")
      for a in dir(self):
        if not a.startswith("__") and not callable(getattr(self, a)):
          f.write("{:30} {} \n".format(a, getattr(self, a)))
      f.write("--------------------------------------------------\n")
      f.close()

  def check_duplicate(self, model_path):
    current_log = "Configurations: \n"
    for a in dir(self):
      if not a.startswith("__") and not callable(getattr(self, a)):
        current_log += "{:30} {} \n".format(a, getattr(self, a))

    for file in os.listdir(model_path):
      if file.startswith(self.exp_name) and file.endswith(".log"):
        with open(os.path.join(model_path, file), "r") as f:
          log = f.read()
        f.close()
        if current_log == log:
          return file.replace(".log", "")
    return None


config = setup_config(model_name=options.model_name,
                      num_class=options.num_class,
                      init=options.init,
                      data_set=options.data_set,
                      linear_classifier=options.linear_classifier,
                      sobel=options.sobel,
                      normalization=options.normalization,
                      augment=options.augment,
                      train_list=options.train_list,
                      val_list=options.val_list,
                      test_list=options.test_list,
                      img_size=options.img_size,
                      img_depth=options.img_depth,
                      batch_size=options.batch_size,
                      num_epoch=options.num_epoch,
                      optimizer=options.optimizer,
                      lr=options.lr,
                      lr_Scheduler=options.lr_Scheduler,
                      early_stop=options.early_stop,
                      patience=options.patience,
                      proxy_dir=options.proxy_dir,
                      proxy_idx=options.proxy_idx
                      )


def train(train_loader, model, criterion, optimizer, epoch):
  batch_time = AverageMeter('Time', ':6.3f')
  losses = AverageMeter('Loss', ':.4e')
  progress = ProgressMeter(
    len(train_loader),
    [batch_time, losses],
    prefix="Epoch: [{}]".format(epoch))

  # model training
  if options.linear_classifier:
    """
        Switch to eval mode:
        Under the protocol of linear classification on frozen features/models,
        it is not legitimate to change any part of the pre-trained model.
        BatchNorm in train mode may revise running mean/std (even if it receives
        no gradient), which are part of the model parameters too.
        """
    model.eval()
  else:
    model.train()

  end = time.time()
  for i, (input, target) in enumerate(train_loader):
    varInput, varTarget = input.float().to(device), target.float().to(device)

    varOutput = model(varInput)

    loss = criterion(varOutput, varTarget)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.update(loss.item(), varInput.size(0))
    batch_time.update(time.time() - end)
    end = time.time()

    if i % options.print_freq == 0:
      progress.display(i)


def validate(val_loader, model, criterion):
  # model validation
  model.eval()

  with torch.no_grad():
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
      len(val_loader),
      [batch_time, losses], prefix='Val: ')

    end = time.time()
    for i, (input, target) in enumerate(valid_data):
      varInput, varTarget = input.float().to(device), target.float().to(device)

      varOutput = model(varInput)

      loss = criterion(varOutput, varTarget)

      losses.update(loss.item(), varInput.size(0))
      losses.update(loss.item(), varInput.size(0))
      batch_time.update(time.time() - end)
      end = time.time()

      if i % options.print_freq == 0:
        progress.display(i)

  return losses.avg


def test(pathCheckpoint, test_loader, config):
  model, _ = Classifier_model(config.model_name.lower(), config.num_class, linear_classifier=options.linear_classifier,
                              sobel=options.sobel, activation=config.activate)
  print(model)

  modelCheckpoint = torch.load(pathCheckpoint)
  state_dict = modelCheckpoint['state_dict']
  for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith('module.'):
      # remove prefix
      state_dict[k[len("module."):]] = state_dict[k]
      # delete renamed or unused k
      del state_dict[k]

  msg = model.load_state_dict(state_dict)
  assert len(msg.missing_keys) == 0
  print("=> loaded pre-trained model '{}'".format(pathCheckpoint))

  if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
  model.to(device)

  model.eval()

  y_test = torch.FloatTensor().cuda()
  p_test = torch.FloatTensor().cuda()

  with torch.no_grad():
    for i, (input, target) in enumerate(tqdm(test_loader)):
      target = target.cuda()
      y_test = torch.cat((y_test, target), 0)

      if len(input.size()) == 4:
        bs, c, h, w = input.size()
        n_crops = 1
      elif len(input.size()) == 5:
        bs, n_crops, c, h, w = input.size()

      varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())

      out = model(varInput)
      outMean = out.view(bs, n_crops, -1).mean(1)
      p_test = torch.cat((p_test, outMean.data), 0)

  return y_test, p_test

def adjust_learning_rate(optimizer, org_lr, epoch, schedule=[20, 40], decay_rate=0.1):
  """Decay the learning rate based on schedule"""
  lr = org_lr

  for milestone in schedule:
    lr *= decay_rate if epoch >= milestone else 1.

  print("---> learning rate is set to {}".format(lr))

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr

# training phase
if options.mode == "train":
  # Load data
  _exp_name = config.log_config(model_path, clean=options.clean)
  model_path = os.path.join(model_path, _exp_name)
  logs_path = os.path.join(model_path, "Logs")
  if not os.path.exists(logs_path):
    os.makedirs(logs_path)

  log_file = os.path.join(model_path, "models.log")

  augment = Augmentation(normalize=config.normalization).get_augmentation(
    "{}_{}".format(config.augment, config.img_size), "train")

  datasetTrain = ChestX_ray14(pathImageDirectory=config.data_dir, pathDatasetFile=config.train_list,
                              augment=augment)
  train_data = DataLoader(dataset=datasetTrain, batch_size=config.batch_size, shuffle=True,
                          num_workers=options.workers, pin_memory=True)

  if config.val_list is not None:
    augment = Augmentation(normalize=config.normalization).get_augmentation(
      "{}_{}".format(config.augment, config.img_size), "valid")
    datasetVal = ChestX_ray14(pathImageDirectory=config.data_dir, pathDatasetFile=config.val_list,
                              augment=augment)
    valid_data = DataLoader(dataset=datasetVal, batch_size=config.batch_size, shuffle=False,
                            num_workers=options.workers, pin_memory=True)
  else:
    valid_data = None

  # training phase
  for i in range(options.start_index, options.num_trial):
    exp_name = config.exp_name + "_run_" + str(i)
    if experiment_exist(log_file, exp_name):
      print(exp_name + " exists!")
      continue

    init_epoch = 0
    init_loss = 100000

    if config.init.lower() == "moco" or config.init.lower() == "simclr" \
        or config.init.lower() == "c2l" or config.init.lower() == "p2w" \
        or config.init.lower() == "ssl_transfer":
      _exp_name = get_weight_name(os.path.join(config.proxy_dir, "models.log"), config.proxy_idx)

      if _exp_name is None:
        print("Pretrained model does not exist!")
        break

      _model_name = os.path.join(config.proxy_dir, _exp_name + ".pth.tar")
      model, _ = Classifier_model(config.model_name.lower(), config.num_class, weight=_model_name,
                                  linear_classifier=options.linear_classifier, sobel=options.sobel,
                                  activation=config.activate)
    else:
      model, _ = Classifier_model(config.model_name.lower(), config.num_class, weight=config.init,
                                  linear_classifier=options.linear_classifier, activation=config.activate)

    print(model)

    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)
    model.to(device)

    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    if options.linear_classifier:
      assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.Adam(parameters, lr=config.lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=config.patience // 2, mode='min',
                                  threshold=0.0001, min_lr=0, verbose=True)
    loss = torch.nn.BCELoss()

    if options.resume:
      # check whether continue from previous training
      resume = os.path.join(model_path, exp_name + '_checkpoint.pth.tar')
      if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)

        init_epoch = checkpoint['epoch']
        init_loss = checkpoint['lossMIN']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        print("=> loaded checkpoint '{}' (epoch={:04d}, val_loss={:.5f})"
              .format(resume, init_epoch, init_loss))
      else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    lossMIN = init_loss
    patience_cnt = 0
    _model_path = os.path.join(model_path, exp_name)

    for epochID in range(init_epoch, config.num_epoch):
      if config.lr_Scheduler.lower() == "lineardecay":
        adjust_learning_rate(optimizer, config.lr, epochID)

      train(train_data, model, loss, optimizer, epochID)

      if valid_data is not None:
        val_loss = validate(valid_data, model, loss)

      if config.lr_Scheduler is not None and config.lr_Scheduler.lower() == "reducelronplateau":
        scheduler.step(val_loss)

      if config.early_stop:
        if val_loss < lossMIN:
          print(
            "Epoch {:04d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(epochID, lossMIN, val_loss,
                                                                                               _model_path))
          lossMIN = val_loss
          patience_cnt = 0

          if config.lr_Scheduler is not None and config.lr_Scheduler.lower() == "reducelronplateau":
            save_checkpoint({
              'epoch': epochID + 1,
              'lossMIN': lossMIN,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
            }, True, filename=_model_path)
          else:
            save_checkpoint({
              'epoch': epochID + 1,
              'lossMIN': lossMIN,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
            }, True, filename=_model_path)
        else:
          if config.lr_Scheduler is not None and config.lr_Scheduler.lower() == "reducelronplateau":
            save_checkpoint({
              'epoch': epochID + 1,
              'lossMIN': lossMIN,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
              'scheduler': scheduler.state_dict(),
            }, False, filename=_model_path)
          else:
            save_checkpoint({
              'epoch': epochID + 1,
              'lossMIN': lossMIN,
              'state_dict': model.state_dict(),
              'optimizer': optimizer.state_dict(),
            }, False, filename=_model_path)

          patience_cnt += 1

        if patience_cnt > config.patience:
          break
      else:
        save_checkpoint({
          'epoch': epochID + 1,
          'state_dict': model.state_dict(),
          'optimizer': optimizer.state_dict(),
        }, True, filename=_model_path)

    os.remove(_model_path + '_checkpoint.pth.tar')

    # log experiment
    with open(log_file, 'a') as f:
      f.write(exp_name + "\n")
      f.close()

model_path = "../Models/ChestXray14/"
_exp_name = config.check_duplicate(model_path)
if _exp_name is None:
  print("Experiment does not exist!")
  exit(-1)
model_path = os.path.join(model_path, _exp_name)

# load test data
if options.mode == "train" or options.mode == "test":
  output_file = os.path.join(output_path, _exp_name + "_test.txt")

  augment = Augmentation(normalize=config.normalization).get_augmentation(
    "{}_{}".format(config.augment, config.img_size), "test", options.test_augment)

  datasetTest = ChestX_ray14(pathImageDirectory=config.data_dir, pathDatasetFile=config.test_list,
                                 augment=augment)
  test_data = DataLoader(dataset=datasetTest, batch_size=config.batch_size, shuffle=False,
                         num_workers=options.workers, pin_memory=True)

elif "valid" in options.mode:
  output_file = os.path.join(output_path, _exp_name + "_valid.txt")

  augment = Augmentation(normalize=config.normalization).get_augmentation(
    "{}_{}".format(config.augment, config.img_size), "valid")

  datasetTest = ChestX_ray14(pathImageDirectory=config.data_dir, pathDatasetFile=config.val_list,
                                 augment=augment)
  test_data = DataLoader(dataset=datasetTest, batch_size=config.batch_size, shuffle=False,
                         num_workers=options.workers, pin_memory=True)

cudnn.benchmark = True

# testing phase
log_file = os.path.join(model_path, "models.log")
if not os.path.isfile(log_file):
  print("log_file ({}) not exists!".format(log_file))
else:
  mean_auc = []
  config.print_config(output_file)
  with open(log_file, 'r') as f_r, open(output_file, 'a') as f_w:
    exp_name = f_r.readline()
    while exp_name:
      exp_name = exp_name.replace('\n', '')
      pathCheckpoint = os.path.join(model_path, exp_name + ".pth.tar")

      y_test, p_test = test(pathCheckpoint, test_data, config)

      aurocIndividual = computeAUROC(y_test, p_test, config.num_class)
      print(">>{}: AUC = {}".format(exp_name, config.class_name))
      print(">>{}: AUC = {}".format(exp_name, np.array2string(np.array(aurocIndividual), precision=4, separator=',')))
      f_w.write("{}: AUC = {}\n".format(exp_name, config.class_name))
      f_w.write(
        "{}: AUC = {}\n".format(exp_name, np.array2string(np.array(aurocIndividual), precision=4, separator='\t')))

      if 'No_Finding' in config.class_name:
        index = aurocIndividual.index('No_Finding')
        aurocIndividual.pop(index)
      aurocMean = np.array(aurocIndividual).mean()
      print(">>{}: AUC = {:.4f}".format(exp_name, aurocMean))
      f_w.write("{}: ACC = {:.4f}\n".format(exp_name, aurocMean))

      mean_auc.append(aurocMean)
      exp_name = f_r.readline()

    mean_auc = np.array(mean_auc)
    print(">> All trials: mAUC  = {}".format(np.array2string(mean_auc, precision=4, separator=',')))
    f_w.write("All trials: mAUC  = {}\n".format(np.array2string(mean_auc, precision=4, separator='\t')))
    print(">> All trials: mAUC(mean)  = {:.4f}".format(np.mean(mean_auc)))
    f_w.write("All trials: mAUC(mean)  = {:.4f}\n".format(np.mean(mean_auc)))
    print(">> All trials: mAUC(std)  = {:.4f}".format(np.std(mean_auc)))
    f_w.write("All trials: mAUC(std)  = {:.4f}\n".format(np.std(mean_auc)))
