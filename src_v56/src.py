#!/usr/bin/env python
# coding: utf-8

# In[9]:


from typing import List, NoReturn, Union, Tuple, Optional, Text, Generic, Callable, Dict
from sklearn.metrics import f1_score, cohen_kappa_score, mean_squared_error
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO
from scipy.signal import butter, lfilter, filtfilt, savgol_filter, detrend
from sklearn.model_selection import KFold, GroupKFold
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm
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
from optimizer import *
import shutil
from datetime import datetime

import argparse

import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_toolbelt import losses as L
import torch.nn as nn
from tsfresh.feature_extraction import feature_calculators
import librosa
import pywt

from sklearn.preprocessing import StandardScaler


import time
import pywt
import os
import gc
import shutil

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


def read_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = '/home/lyh/liverpool/'
    train = pd.read_csv(data_dir + '/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv(data_dir + '/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv(data_dir + '/sample_submission.csv', dtype={'time': np.float32})
    
    reverse_train = train[::-1].reset_index(drop=True)
    
    train = pd.concat([train, reverse_train], axis=0).reset_index(drop=True)
    
    Y_train_proba = np.load("/home/lyh/liverpool/Y_train_proba.npy")
    Y_test_proba = np.load("/home/lyh/liverpool/Y_test_proba.npy")
    
    Y_train_proba = np.concatenate([Y_train_proba, Y_train_proba[::-1]])
    #Y_test_proba = np.concatenate([Y_test_proba, Y_test_proba[::-1]])
    
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]
        
        
        
    
    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
    #print(df)
    df['group'] = df.groupby(df.index//batch_size, sort=False)['signal'].agg(['ngroup']).values
    df['group'] = df['group'].astype(np.uint16)
    return df

# normalize the data (standard scaler). We can also try other scalers for a better score!
def normalize(train, test):
    train_input_mean = train.signal.mean()
    train_input_sigma = train.signal.std()
    train['signal'] = (train.signal - train_input_mean) / train_input_sigma
    test['signal'] = (test.signal - train_input_mean) / train_input_sigma
    return train, test

def diff_features(df, shifts):
    for shift in shifts:    
        df['signal_diff_shift_' + str(shift)] = df.groupby('group')['signal'].diff(shift).fillna(0)
    return df

def categorize(df, thres):
    df = df > thres
    return df

def cat_features(df, thresholds):
    for thres in thresholds:    
        df['signal_cat_' + str(thres)] = df.groupby('group')['signal'].apply(lambda x: categorize(x, thres)).astype(float)
    return df

# get lead and lags features
def lag_with_pct_change(df, windows):
    for window in windows:    
        df['signal_shift_pos_' + str(window)] = df.groupby('group')['signal'].shift(window).fillna(0)
        df['signal_shift_neg_' + str(window)] = df.groupby('group')['signal'].shift(-1 * window).fillna(0)
    return df

# main module to run feature engineering. Here you may want to try and add other features and check if your score imporves :).
def run_feat_engineering(df, batch_size):
    # create batches
    df = batching(df, batch_size = batch_size)
    # create leads and lags (1, 2, 3 making them 6 features)
    df = lag_with_pct_change(df, [1, 2, 3])
    df = diff_features(df, [-1, 1])
    df = cat_features(df, [-2, -1, 0, 1, 2, 3])
    # create signal ** 2 (this is the new feature)
    df['signal_2'] = df['signal'] ** 2
    return df

# fillna with the mean and select features for training
def feature_selection(train, test):
    features = [col for col in train.columns if col not in ['index', 'group', 'open_channels', 'time']]
    train = train.replace([np.inf, -np.inf], np.nan)
    test = test.replace([np.inf, -np.inf], np.nan)
    for feature in features:
        if 'cat' in feature:
            continue
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features



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
    def __init__(self, data, labels, mode):
        self.data = torch.tensor(data, dtype=torch.float)
        #self.feats = torch.tensor(feats, dtype=torch.float)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.mode = mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        #feat = self.feats[idx]
        labels = self.labels[idx]
        #if self.mode == 'train':
        #    if np.random.rand() < 0.5:
        #        data = torch.flip(data, dims=[1])
        #        labels = torch.flip(labels, dims=[0])

        return [data, labels]
    
class IonTestDataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data, dtype=torch.float)
        #self.feats = torch.tensor(feats, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        #feat = self.feats[idx]
        return data

class SEModule(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels//reduction, kernel_size=1, padding=0)
        self.conv2 = nn.Conv1d(in_channels//reduction, in_channels, kernel_size=1, padding=0)
        
    def forward(self, x):
        # x: [B, C, H]
        s = F.adaptive_avg_pool1d(x, 1) # [B, C, 1]
        s = self.conv1(s) # [B, C//reduction, 1]
        s = F.relu(s, inplace=True)
        s = self.conv2(s) # [B, C, 1]
        x = x + torch.sigmoid(s)
        return x

class ConvBR1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, dilation=1, stride=1, groups=1, is_activation=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, groups=groups, bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.is_activation = is_activation
        
        if is_activation:
            self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.is_activation:
            x = self.relu(x)
        return x
    
    

class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, bidirectional, drop_prob=0.2):
        super(GRUNet, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        self.drop_prob = drop_prob
        
        self.block0 = nn.Sequential(
                    ConvBR1d(self.input_size, self.hidden_size, kernel_size=1, stride=1, padding=0),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=1, stride=1, padding=0),
                )
        
        self.block1 = nn.Sequential(
                    ConvBR1d(self.input_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=3, stride=1, padding=1),
                )

        self.block2 = nn.Sequential(
                    ConvBR1d(self.input_size, self.hidden_size, kernel_size=5, stride=1, padding=2),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1, padding=2),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=5, stride=1, padding=2),
                )
        self.block3 = nn.Sequential(
                    ConvBR1d(self.input_size, self.hidden_size, kernel_size=7, stride=1, padding=3),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=7, stride=1, padding=3),
                    ConvBR1d(self.hidden_size, self.hidden_size, kernel_size=7, stride=1, padding=3),
                )

        self.rnn1 = torch.nn.GRU(input_size=self.hidden_size * 4, 
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
        
        #self.feat_fc1 = torch.nn.Linear(num_feats, hidden_size*2)
        #self.feat_fc2 = torch.nn.Linear(num_feats, hidden_size*2)
        
        self.activation_fn = torch.relu

        
    def forward(self, x):
        x0 = self.block0(x)
        x1 = self.block1(x)
        x2 = self.block2(x)
        x3 = self.block3(x)
        
        x = torch.cat([x0, x1, x2, x3], dim=1).permute(0, 2, 1)
        
        
        outputs, hidden = self.rnn1(x)
        outputs, hidden = self.rnn2(outputs)
        outputs = self.activation_fn(outputs)
        
        outputs = self.last_fc(outputs)
        return outputs

# Splits data

def train(args):
    log = Logger()
    log.open(f'./logs/log_fold_{args.fold}.txt', mode='w')
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

    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Feat imp... \n")

    #train['signal_kalman'] = pd.read_csv(args.data_dir + '/train_clean_kalman.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})['signal']
    #test['signal_kalman'] = pd.read_csv(args.data_dir + '/test_clean_kalman.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})['signal']

    train, test = denoising(train, test, wavelet='db1', threshold=2)
    #train, test = denoising(train, test, wavelet='db1', threshold=4)
    #train, test = denoising(train, test, wavelet='db1', threshold=8)

    train, test = denoising(train, test, wavelet='db2', threshold=2)
    #train, test = denoising(train, test, wavelet='db2', threshold=4)
    #train, test = denoising(train, test, wavelet='db2', threshold=8)
    
    train = run_feat_engineering(train, batch_size=4000)
    test = run_feat_engineering(test, batch_size=4000)

    STD = 0.01
    old_data = train['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(train))
    train['signal_noised'] = new_data

    old_data = test['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(test))
    test['signal_noised'] = new_data

    use_cols = [c for c in train.columns if c not in ['time', 'open_channels', 'group']]

    for col in use_cols:
        train_input_mean = train[col].mean()
        train_input_sigma = train[col].std()
        train[col] = (train[col]-train_input_mean)/train_input_sigma
        test[col] = (test[col]-train_input_mean)/train_input_sigma

    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Feat imp Done... {use_cols} \n")


    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Folding \n")

    train_vals = train[use_cols].values
    test_vals = test[use_cols].values
    train_y = train['open_channels'].values
    
    len_train = len(train)
    train_seqs = []
    train_ys = []
    for start in np.arange(0, len(train), args.gap):
        if start > len_train - 4000:
            break
        train_seqs.append(train_vals[start:start+4000, :])
        train_ys.append(train_y[start:start+4000])
        
    train = np.stack(train_seqs)
    train_tr = np.stack(train_ys)
    
    
    len_test = len(test)
    test_seqs = []
    for start in np.arange(0, len(test), 4000):
        if start > len_test - 4000:
            break
        test_seqs.append(test_vals[start:start+4000, :])
        
    test = np.stack(test_seqs)
        
    """
    train_feats = pd.DataFrame()
    for idx in range(len(train)):
        s = train[idx][:, 0]

        mfcc = librosa.feature.mfcc(s + 0)
        mfcc_mean = mfcc.mean(axis=1)
        percentile_roll50_std_20 = np.percentile(pd.Series(s).rolling(50).std().dropna().values, 20)

        train_feats.loc[idx, 'num_peaks_1'] = feature_calculators.number_peaks(s, 1)
        train_feats.loc[idx, 'num_peaks_2'] = feature_calculators.number_peaks(s, 2)
        train_feats.loc[idx, 'num_peaks_3'] = feature_calculators.number_peaks(s, 3)
        train_feats.loc[idx, 'var_percentile_roll50_std_20'] = percentile_roll50_std_20
        train_feats.loc[idx, 'var_mfcc_mean18'] = mfcc_mean[18]
        train_feats.loc[idx, 'var_mfcc_mean4'] = mfcc_mean[4]
        
        
    test_feats = pd.DataFrame()
    for idx in range(len(test)):
        s = test[idx][:, 0]

        mfcc = librosa.feature.mfcc(s + 0)
        mfcc_mean = mfcc.mean(axis=1)
        percentile_roll50_std_20 = np.percentile(pd.Series(s).rolling(50).std().dropna().values, 20)

        test_feats.loc[idx, 'num_peaks_1'] = feature_calculators.number_peaks(s, 1)
        test_feats.loc[idx, 'num_peaks_2'] = feature_calculators.number_peaks(s, 2)
        test_feats.loc[idx, 'num_peaks_3'] = feature_calculators.number_peaks(s, 3)
        test_feats.loc[idx, 'var_percentile_roll50_std_20'] = percentile_roll50_std_20
        test_feats.loc[idx, 'var_mfcc_mean18'] = mfcc_mean[18]
        test_feats.loc[idx, 'var_mfcc_mean4'] = mfcc_mean[4]
    
    
    for col in train_feats.columns:
        scaler = StandardScaler()
        train_feats[col] = scaler.fit_transform(train_feats[col].values.reshape(-1, 1))
        test_feats[col] = scaler.transform(test_feats[col].values.reshape(-1, 1))
    """
    

    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} train.shape {train.shape} train_tr.shape {train_tr.shape} test.shape {test.shape} \n")

    FOLDs = KFold(n_splits=5, shuffle=True, random_state=42)

    for n_fold, (trn_idx, vld_idx) in enumerate(FOLDs.split(train)):
        if n_fold == args.fold:
            break

    train_x, train_y = train[trn_idx], train_tr[trn_idx]
    valid_x, valid_y = train[vld_idx], train_tr[vld_idx]
    
    #train_feat_x, valid_feat_x = train_feats.values[trn_idx], train_feats.values[vld_idx]

    train_x = train_x.transpose((0, 2, 1))
    valid_x = valid_x.transpose((0, 2, 1))

    trn_dataset = IonDataset(train_x, train_y, mode='train')
    trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    vld_dataset = IonDataset(valid_x, valid_y, mode='test')
    vld_loader = DataLoader(vld_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Make model, optimizer, criterion

    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Load models ... \n")
    
    model = GRUNet(input_size = train_x.shape[1], hidden_size=128, 
                   output_size=11, num_layers=2, bidirectional=True,  drop_prob=0.2).to(device)

    gradient_accumulation_steps = 1
    t_total = len(trn_loader) // gradient_accumulation_steps * args.epochs

    optimizer = AdamW(model.parameters(), lr=args.initial_lr, eps=1e-6)
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=t_total
    )

    loss_fn = L.FocalLoss()

    best_score = -1
    best_model_name = "_"

    # Training
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Training ... \n")

    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss = []
        model.train()
        #for inputs, targets in tqdm(trn_loader):
        for (inputs, targets) in trn_loader:
            inputs = inputs.to(device)
            #feats = feats.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs.view(-1, outputs.shape[-1]) , targets.view(-1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            scheduler.step()

        vld_loss = []
        vld_true = []
        vld_pred = []
        model.eval()
        with torch.no_grad():
            for (inputs, targets) in vld_loader:
                inputs = inputs.to(device)
                #feats = feats.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                loss = loss_fn(outputs.view(-1, outputs.shape[-1]) , targets.view(-1))
                vld_loss.append(loss.item())

                vld_true.append(targets.cpu().view(-1).numpy())
                vld_pred.append(outputs.cpu().argmax(dim=2).view(-1).numpy())

        vld_true = np.concatenate(vld_true)
        vld_pred = np.concatenate(vld_pred)

        vld_loss = np.mean(vld_loss)
        train_loss = np.mean(train_loss)

        final_score = f1_score(vld_true, vld_pred, average='macro')

        


        if final_score > best_score:

            best_score = final_score
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            #log.write(f"{date_time} Load models ... \n")
            log.write(f'{date_time} epoch: {epoch}; train_loss: {train_loss:.5f}; vld_loss: {vld_loss:.5f}; score: {best_score:.5f} in {time.time()-start_time:.2f}s +\n')


            best_model_name = os.path.join(args.model_dir, args.version) + f"/{args.version}_model_fold_{args.fold}_ep_{epoch}_score_{best_score:.5f}.pt"

            state_dict = model.cpu().state_dict()
            model = model.cuda()
            torch.save(state_dict, best_model_name)
            shutil.copy(best_model_name, os.path.join(args.model_dir, args.version) + f"/Best_{args.version}_fold_{args.fold}.pt")
        else:
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            #log.write(f"{date_time} Load models ... \n")
            log.write(f'{date_time} epoch: {epoch}; train_loss: {train_loss:.5f}; vld_loss: {vld_loss:.5f}; score: {final_score:.5f} in {time.time()-start_time:.2f}s \n')


    # Get subs    
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Get subs ... \n")

    model.load_state_dict(torch.load(os.path.join(args.model_dir, args.version) + f"/Best_{args.version}_fold_{args.fold}.pt"))

    test_dataset = IonTestDataset(test.transpose((0, 2, 1)), test_feats.values)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    test_pred = []
    test_prob = []
    model.eval()
    with torch.no_grad():
        for inputs in test_loader:
        #for inputs in tqdm(test_loader):
            inputs = inputs.to(device)
            #feats = feats.to(device)

            outputs = model(inputs)

            test_pred.append(outputs.cpu().argmax(dim=2).view(-1).numpy())
            test_prob.append(torch.softmax(outputs, dim=2).cpu().view(-1, 11).numpy())

    test_pred = np.concatenate(test_pred)
    test_prob = np.concatenate(test_prob)

    np.save(os.path.join(args.oof_dir, args.version) + f'/{args.version}_test_oofs_fold_{args.fold}_score_{best_score:.4f}.npy', test_prob)
    
    data = pd.read_csv(os.path.join(args.data_dir, 'test_clean.csv'))

    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} Writing submission ... \n")
    submission = pd.DataFrame()
    submission['time'] = data['time']
    submission['open_channels'] = test_pred
    submission.to_csv(args.sub_dir + f'/{args.version}_sub_fold_{args.fold}_score_{best_score:.4f}.csv', index=False, float_format='%.4f')

    # Get subs
    date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    log.write(f"{date_time} ##################### Done ... ##################### \n\n\n\n\n\n")

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--fold', type=int, default=0, help='fold')
    parser.add_argument('--epochs', type=int, default=128, help='total epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--initial_lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--gpu_number', type=str, default="0", help='gpu cuda number')
    parser.add_argument('--num_workers', type=int, default=0, help='num_workers')
    parser.add_argument('--version', type=str, default="debug",help='experiment name')
    parser.add_argument('--debug', type=str, default="False",help='experiment name')
    parser.add_argument('--gap', type=int, default=4000, help='gap')
    
    arguments = parser.parse_args()

    class CFG:
        batch_size = 16
        initial_lr = 1e-3

    args = CFG
    args.data_dir = '/home/lyh/liverpool/'
    args.model_dir = '/home/lyh/liverpool/models/'
    args.oof_dir = '/home/lyh/liverpool/oofs/'
    args.sub_dir = '/home/youhanlee/project/kaggle-liverpool/subs/'
    args.num_classes = 11
    args.seed = 42
    args.fold = arguments.fold
    args.epochs = arguments.epochs
    args.batch_size = arguments.batch_size
    args.initial_lr = arguments.initial_lr
    args.gpu_number = arguments.gpu_number
    args.num_workers = arguments.num_workers
    args.version = arguments.version
    args.debug = arguments.debug
    args.gap = arguments.gap
    
    seed_everything(args.seed)
  
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
        

    

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= f'{args.gpu_number}'  # specify which GPU(s) to be used

    if not os.path.exists(os.path.join(args.model_dir, args.version)):
        os.mkdir(os.path.join(args.model_dir, args.version))

    if not os.path.exists(os.path.join(args.oof_dir, args.version)):
        os.mkdir(os.path.join(args.oof_dir, args.version))
        
    train(args)

if __name__ == '__main__':
    main()
