#!/usr/bin/env python
# coding: utf-8


from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from scipy.signal import butter, lfilter, filtfilt, savgol_filter, detrend
from sklearn.model_selection import KFold, GroupKFold
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from contextlib import contextmanager
from joblib import Parallel, delayed
from IPython.display import display
from sklearn import preprocessing
import scipy.stats as stats
import random as rn
import pandas as pd
import numpy as np
import scipy as sp
import itertools
import warnings
import math
import sys
from utils import *
from datetime import datetime

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_toolbelt import losses as L
import torch.nn as nn

import time
import pywt
import os
import gc

warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

def seed_everything(seed):
    rn.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def lag_with_pct_change(df : pd.DataFrame,
                        shift_sizes : Optional[List]=[1, 2],
                        add_pct_change : Optional[bool]=False,
                        add_pct_change_lag : Optional[bool]=False) -> pd.DataFrame:
    
    for shift_size in shift_sizes:    
        df['signal_shift_pos_'+str(shift_size)] = df.groupby('group')['signal'].shift(shift_size).fillna(0)
        df['signal_shift_neg_'+str(shift_size)] = df.groupby('group')['signal'].shift(-1*shift_size).fillna(0)

    if add_pct_change:
        df['pct_change'] = df['signal'].pct_change()
        if add_pct_change_lag:
            df['pct_change_shift_pos_'+str(shift_size)] = df.groupby('group')['pct_change'].shift(shift_size).fillna(0)
            df['pct_change_shift_neg_'+str(shift_size)] = df.groupby('group')['pct_change'].shift(-1*shift_size).fillna(0)
    return df

def run_feat_enginnering(df : pd.DataFrame,
                         create_all_data_feats : bool,
                         batch_size : int) -> pd.DataFrame:
    
    df = batching(df, batch_size=batch_size)
    if create_all_data_feats:
        df = lag_with_pct_change(df, [1, 2, 3],  add_pct_change=False, add_pct_change_lag=False)
    
    return df

def read_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = '/SSD4T/lyh/liverpool/'
    train = pd.read_csv(data_dir + '/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv(data_dir + '/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv(data_dir + '/sample_submission.csv', dtype={'time': np.float32})
    
    return train, test, sub

def normalize(train, test):
    
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal-train_input_mean)/train_input_sigma
    test['signal'] = (test.signal-train_input_mean)/train_input_sigma

    return train, test

def feature_selection(df : pd.DataFrame,
                      df_test : pd.DataFrame) -> Tuple[pd.DataFrame , pd.DataFrame, List]:
    use_cols = [col for col in df.columns if col not in ['index','group', 'open_channels', 'time']]
    df = df.replace([np.inf, -np.inf], np.nan)
    df_test = df_test.replace([np.inf, -np.inf], np.nan)
    for col in use_cols:
        col_mean = pd.concat([df[col], df_test[col]], axis=0).mean()
        df[col] = df[col].fillna(col_mean)
        df_test[col] = df_test[col].fillna(col_mean)
   
    gc.collect()
    return df, df_test, use_cols

def batching(df : pd.DataFrame,
             batch_size : int) -> pd.DataFrame :
    
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
        
    return df

def denoise_signal_simple(x, wavelet='db1', threshold=3):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    #univeral threshold
    threshold = 3
    coeff[1:] = (pywt.threshold(i, value=threshold, mode='hard') for i in coeff[1:])
    # Reconstruct the signal using the thresholded coefficients
    return pywt.waverec(coeff, wavelet, mode='per')

def denoising(train, test, wavelet, threshold):
    col_name = f'signal_denoise_{wavelet}_{threshold}'
    train[col_name] = denoise_signal_simple(train['signal'].values, wavelet, threshold)
    test[col_name] = denoise_signal_simple(test['signal'].values, wavelet, threshold)
    train_input_mean = train[col_name].mean()
    train_input_sigma = train[col_name].std()
    train[col_name] = (train[col_name]-train_input_mean)/train_input_sigma
    test[col_name] = (test[col_name]-train_input_mean)/train_input_sigma

    return train, test

class IonDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        labels = self.labels[idx]
        return [data, labels]
    
class IonTestDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        return data

    

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, drop_prob=0.2):
        super(GRUNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.drop_prob = drop_prob

        self.rnn1 = torch.nn.GRU(input_size=self.input_size, 
                                 hidden_size=self.hidden_size, 
                                 num_layers=self.num_layers, 
                                 bidirectional=self.bidirectional, 
                                 batch_first=True,
                                 dropout=self.drop_prob)
        
        self.rnn2 = torch.nn.GRU(input_size=self.hidden_size * 2, 
                                 hidden_size=self.hidden_size, 
                                 num_layers=self.num_layers, 
                                 bidirectional=self.bidirectional, 
                                 batch_first=True,
                                 dropout=self.drop_prob)
        
#         self.rnn3 = torch.nn.GRU(input_size=self.hidden_size * 2, 
#                                  hidden_size=self.hidden_size, 
#                                  num_layers=self.num_layers, 
#                                  bidirectional=self.bidirectional, 
#                                  batch_first=True,
#                                  dropout=self.drop_prob)
        
        self.last_fc = torch.nn.Linear(hidden_size*2, self.output_size)
        self.activation_fn = torch.relu

        
    def forward(self, x):
        outputs, hidden = self.rnn1(x.permute(0, 2, 1))
        outputs, hidden = self.rnn2(outputs)
        #outputs, hidden = self.rnn3(outputs)
        outputs = self.activation_fn(outputs)
        outputs = self.last_fc(outputs)
        return outputs

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR



def get_constant_schedule(optimizer, last_epoch=-1):
    """ Create a schedule with a constant learning rate.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def get_constant_schedule_with_warmup(optimizer, num_warmup_steps, last_epoch=-1):
    """ Create a schedule with a constant learning rate preceded by a warmup
    period during which the learning rate increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1.0, num_warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function between 0 and `pi * cycles` after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_cosine_with_hard_restarts_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=1.0, last_epoch=-1
):
    """ Create a schedule with a learning rate that decreases following the
    values of the cosine function with several hard restarts, after a warmup
    period during which it increases linearly between 0 and 1.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class AdamW(Optimizer):
    """ Implements Adam algorithm with weight decay fix.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.0, correct_bias=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1.0 - beta2, grad, grad)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.data.add_(-group["lr"] * group["weight_decay"], p.data)

        return loss

parser = argparse.ArgumentParser(description='')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--epochs', type=int, default=128, help='total epochs')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--start_lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--gpu_number', type=str, default="0", help='gpu cuda number')
parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
parser.add_argument('--version', type=str, default="debug",help='experiment name')
parser.add_argument('--pretrained_weights', type=str, default="debug",help='pretrained_weights')

arguments = parser.parse_args()

class CFG:
    batch_size = 16
    initial_lr = 1e-3

args = CFG
args.data_dir = '/SSD4T/lyh/liverpool/'
args.model_dir = '/SSD4T/lyh/liverpool/models/'
args.oof_dir = '/SSD4T/lyh/liverpool/oofs/'
args.sub_dir = '/home/youhanlee/project/kaggle-liverpool/subs/'
args.num_classes = 11
args.seed = 42
args.fold = arguments.fold
args.epochs = arguments.epochs
args.batch_size = arguments.batch_size
args.start_lr = arguments.start_lr
args.gpu_number = arguments.gpu_number
args.num_workers = arguments.num_workers
args.version = arguments.version
args.pretrained_weights = arguments.pretrained_weights

seed_everything(args.seed)

if not os.path.exists('./logs'):
    os.mkdir('./logs')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= f'{args.gpu_number}'  # specify which GPU(s) to be used

if not os.path.exists(os.path.join(args.model_dir, args.version)):
    os.mkdir(os.path.join(args.model_dir, args.version))

if not os.path.exists(os.path.join(args.oof_dir, args.version)):
    os.mkdir(os.path.join(args.oof_dir, args.version))


log = Logger()
log.open(f'./logs/log_oofs_{args.fold}.txt', mode='w')
date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} ##################### Start ... ##################### \n")

fe_config = [
    (True, 4000),
]
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

################################# prepare data
target_col = ['open_channels']

date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} Reading Data Started ... \n")
train, test, sample_submission = read_data()
train, test = normalize(train, test)    

date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} Denoising ... \n")

train, test = denoising(train, test, wavelet='db1', threshold=2)
train, test = denoising(train, test, wavelet='db1', threshold=4)
train, test = denoising(train, test, wavelet='db1', threshold=8)


date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} Creating Features ... \n")

for config in fe_config:
    train = run_feat_enginnering(train, create_all_data_feats=config[0], batch_size=config[1])
    test  = run_feat_enginnering(test,  create_all_data_feats=config[0], batch_size=config[1])

train, test, feats = feature_selection(train, test)

################################# prepare data

oof_ = np.zeros((len(train), 11))
preds_ = np.zeros((len(test), 11))
target = ['open_channels']
group = train['group']
kf = GroupKFold(n_splits=5)
splits = [x for x in kf.split(train, train[target], group)]

new_splits = []
for sp in splits:
    new_split = []
    new_split.append(np.unique(group[sp[0]]))
    new_split.append(np.unique(group[sp[1]]))
    new_split.append(sp[1])    
    new_splits.append(new_split)
#tr = pd.concat([pd.get_dummies(train.open_channels), train[['group']]], axis=1)
tr = pd.concat([train.open_channels, train[['group']]], axis=1)

train_tr = np.array(list(tr.groupby('group').apply(lambda x: x['open_channels'].values))).astype(np.float32)

train = np.array(list(train.groupby('group').apply(lambda x: x[feats].values)))
test = np.array(list(test.groupby('group').apply(lambda x: x[feats].values)))

date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} train.shape {train.shape} train_tr.shape {train_tr.shape} test.shape {test.shape} \n")

for n_fold, (trn_idx, vld_idx, vld_orig_idx) in enumerate(new_splits[0:], start=0):
    if n_fold == args.fold:
        break

train_x, train_y = train[trn_idx], train_tr[trn_idx]
valid_x, valid_y = train[vld_idx], train_tr[vld_idx]

train_x = train_x.transpose((0, 2, 1))
valid_x = valid_x.transpose((0, 2, 1))



model = GRUNet(input_size = train_x.shape[1], hidden_size=128, 
               output_size=11, num_layers=2, bidirectional=True, drop_prob=0.2).to(device)



best_model_name = args.pretrained_weights

final_score = float(best_model_name.split('_')[-1][:-3])

# Get oofs    
date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} Get oofs ... \n")

model.load_state_dict(torch.load(best_model_name))

trn_dataset = IonDataset(train_x, train_y)
trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

vld_dataset = IonDataset(valid_x, valid_y)
vld_loader = DataLoader(vld_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

test_dataset = IonTestDataset(test.transpose((0, 2, 1)))
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

trn_pred = []
model.eval()
with torch.no_grad():
    #for inputs, targets in trn_loader:
    for inputs, targets in tqdm(trn_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        trn_pred.append(torch.softmax(outputs, dim=2).cpu().view(-1, 11).numpy())

vld_pred = []
model.eval()
with torch.no_grad():
    #for inputs, targets in vld_loader:
    for inputs, targets in tqdm(vld_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        vld_pred.append(torch.softmax(outputs, dim=2).cpu().view(-1, 11).numpy())

trn_pred = np.concatenate(trn_pred)
vld_pred = np.concatenate(vld_pred)

np.save(os.path.join(args.oof_dir, args.version) + f'/{args.version}_trn_oofs_fold_{args.fold}_score_{final_score:.4f}.npy', trn_pred)
np.save(os.path.join(args.oof_dir, args.version) + f'/{args.version}_vld_oofs_fold_{args.fold}_score_{final_score:.4f}.npy', vld_pred)


# Get subs
date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} Get subs ... \n")

test_pred = []
test_prob = []
model.eval()
with torch.no_grad():
    #for inputs in test_loader:
    for inputs in tqdm(test_loader):
        inputs = inputs.to(device)

        outputs = model(inputs)

        test_pred.append(outputs.cpu().argmax(dim=2).view(-1).numpy())
        test_prob.append(torch.softmax(outputs, dim=2).cpu().view(-1, 11).numpy())

test_pred = np.concatenate(test_pred)
test_prob = np.concatenate(test_prob)

np.save(os.path.join(args.oof_dir, args.version) + f'/{args.version}_test_oofs_fold_{args.fold}_score_{final_score:.4f}.npy', test_prob)


data = pd.read_csv(os.path.join(args.data_dir, 'test_clean.csv'))

date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} Writing submission ... \n")
submission = pd.DataFrame()
submission['time'] = data['time']
submission['open_channels'] = test_pred
submission.to_csv(args.sub_dir + f'/{args.version}_sub_fold_{args.fold}_score_{final_score:.4f}.csv', index=False, float_format='%.4f')

# Get subs
date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} ##################### Done ... ##################### \n\n\n\n\n\n")

