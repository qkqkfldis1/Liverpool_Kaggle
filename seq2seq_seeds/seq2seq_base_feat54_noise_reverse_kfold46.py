#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from pytorch_toolbelt import losses as L


import gc

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

import time
from datetime import datetime

from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from sklearn.model_selection import KFold
import random
import gc
from utils import *

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_rows', 500)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"  


log = Logger()
log.open(f'./log.seq2seq_base_feat54_noise_reverse_kfold46.txt', mode='w')
date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
log.write(f"{date_time} ##################### Start ... ##################### \n")


# configurations and main hyperparammeters
EPOCHS = 224
NNBATCHSIZE = 16
GROUP_BATCH_SIZE = 4000
SEED = 321
LR = 0.001
SPLITS = 5


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



# In[4]:


# read data
def read_data():
    train = pd.read_csv('/home/lyh/liverpool/train_clean.csv', dtype={'time': np.float32, 'signal': np.float32, 'open_channels':np.int32})
    test  = pd.read_csv('/home/lyh/liverpool/test_clean.csv', dtype={'time': np.float32, 'signal': np.float32})
    sub  = pd.read_csv('/home/lyh/liverpool/sample_submission.csv', dtype={'time': np.float32})
    
    reverse_train = train[::-1].reset_index(drop=True)
    
    train = pd.concat([train, reverse_train], axis=0).reset_index(drop=True)
    
    Y_train_proba = np.load("/home/lyh/liverpool/Y_train_proba.npy")
    Y_test_proba = np.load("/home/lyh/liverpool/Y_test_proba.npy")
    
    Y_train_proba = np.concatenate([Y_train_proba, Y_train_proba[::-1]])
    #Y_test_proba = np.concatenate([Y_test_proba, Y_test_proba[::-1]])
    
    for i in range(11):
        train[f"proba_{i}"] = Y_train_proba[:, i]
        test[f"proba_{i}"] = Y_test_proba[:, i]
        
    STD = 0.01
    old_data = train['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(train))
    train['signal_noised'] = new_data

    old_data = test['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(test))
    test['signal_noised'] = new_data
    
        
    STD = 0.001
    old_data = train['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(train))
    train['signal_noised'] = new_data

    old_data = test['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(test))
    test['signal_noised'] = new_data
    
        
    STD = 0.1
    old_data = train['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(train))
    train['signal_noised'] = new_data

    old_data = test['signal']
    new_data = old_data + np.random.normal(0, STD, size=len(test))
    test['signal_noised'] = new_data
        

    return train, test, sub

# create batches of 4000 observations
def batching(df, batch_size):
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

def diff_diff_features(df, shifts):
    for shift in shifts:    
        df[f'signal_diff_shift_{shift}_diff'] = df.groupby('group')['signal_diff_shift_' + str(shift)].diff().fillna(0)
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
        feature_mean = pd.concat([train[feature], test[feature]], axis = 0).mean()
        train[feature] = train[feature].fillna(feature_mean)
        test[feature] = test[feature].fillna(feature_mean)
    return train, test, features


# In[2]:


print('Reading Data Started...')
train, test, sample_submission = read_data()
train, test = normalize(train, test)
print('Reading and Normalizing Data Completed')

print('Creating Features')
print('Feature Engineering Started...')
train = run_feat_engineering(train, batch_size = GROUP_BATCH_SIZE)
test = run_feat_engineering(test, batch_size = GROUP_BATCH_SIZE)
train, test, features = feature_selection(train, test)
print('Feature Engineering Completed...')


print(f'Training Wavenet model with {SPLITS} folds of GroupKFold Started...')


# In[3]:


seed_everything(SEED)
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
# pivot target columns to transform the net to a multiclass classification estructure (you can also leave it in 1 vector with sparsecategoricalcrossentropy loss function)
tr = pd.concat([train[['open_channels']], train[['group']]], axis=1)


# In[4]:


#tr.columns = ['target_'+str(i) for i in range(11)] + ['group']
#target_cols = ['target_'+str(i) for i in range(11)]
trainval_y = np.array(list(tr.groupby('group').apply(lambda x: x['open_channels'].values))).astype(np.float32)
test_y = np.zeros((test.shape[0], trainval_y.shape[1]))
train = np.array(list(train.groupby('group').apply(lambda x: x[features].values)))
test = np.array(list(test.groupby('group').apply(lambda x: x[features].values)))


# In[5]:


trainval = train.transpose((0,2,1))
test = test.transpose((0,2,1))


# In[6]:


trainval = torch.Tensor(trainval)
test = torch.Tensor(test)

#trainval_y = torch.Tensor(trainval_y)
#test_y = torch.Tensor(test_y)


# In[ ]:





# In[ ]:


class EarlyStopping:
    def __init__(self, patience=5, delta=0, checkpoint_path='checkpoint.pt', is_maximize=True):
        self.patience, self.delta, self.checkpoint_path = patience, delta, checkpoint_path
        self.counter, self.best_score = 0, None
        self.is_maximize = is_maximize

    def load_best_weights(self, model):
        model.load_state_dict(torch.load(self.checkpoint_path))

    def __call__(self, score, model):
        if self.best_score is None or         (score > self.best_score + self.delta if self.is_maximize else score < self.best_score - self.delta):
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_score, self.counter = score, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

    
class Seq2SeqRnn(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, output_size, num_layers=1, bidirectional=False, dropout=.3,
            hidden_layers = [100, 200]):
        
        super().__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers=num_layers
        self.bidirectional=bidirectional
        self.output_size=output_size
        
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, 
                           bidirectional=bidirectional, batch_first=True,dropout=0.3)
         # Input Layer
        if hidden_layers and len(hidden_layers):
            first_layer  = nn.Linear(hidden_size*2 if bidirectional else hidden_size, hidden_layers[0])

            # Hidden Layers
            self.hidden_layers = nn.ModuleList(
                [first_layer]+[nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers) - 1)]
            )
            for layer in self.hidden_layers: nn.init.kaiming_normal_(layer.weight.data)   

            self.intermediate_layer = nn.Linear(hidden_layers[-1], self.input_size)
            # output layers
            self.output_layer = nn.Linear(hidden_layers[-1], output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 
           
        else:
            self.hidden_layers = []
            self.intermediate_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_siz, self.input_size)
            self.output_layer = nn.Linear(hidden_size*2 if bidirectional else hidden_size, output_size)
            nn.init.kaiming_normal_(self.output_layer.weight.data) 

        self.activation_fn = torch.relu
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        x = x.permute(0,2,1)

        outputs, hidden = self.rnn(x)        

        x = self.dropout(self.activation_fn(outputs))
        for hidden_layer in self.hidden_layers:
            x = self.activation_fn(hidden_layer(x))
            x = self.dropout(x)
            
        x = self.output_layer(x)

        return x


class IonDataset(Dataset):
    """Car dataset."""

    def __init__(self, data, labels, training=True, transform=None, flip=0.5, noise_level=0, class_split=0.0):
        self.data = data
        self.labels = labels
        self.transform = transform
        self.training = training
        self.flip = flip
        self.noise_level = noise_level
        self.class_split = class_split

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data = self.data[idx]
        labels = self.labels[idx]
        if np.random.rand() < self.class_split:
            data, labels = class_split(data, labels)
        if  np.random.rand() < self.noise_level:
            data = data * torch.FloatTensor(10000).uniform_(1-self.noise_level, 1+self.noise_level)
        if np.random.rand() < self.flip:
            data = torch.flip(data, dims=[1])
            labels = np.flip(labels, axis=0).copy().astype(int)

        return [data, labels.astype(int)]

    
    
    
test_preds_all = np.zeros((2000000, 11))

FOLDs = KFold(n_splits=5, random_state=46, shuffle=True)

if not os.path.exists("./models"):
            os.makedirs("./models")
#for index, (train_index, val_index, val_orig_idx) in enumerate(new_splits[0:], start=0):
for index, (train_index, val_index) in enumerate(FOLDs.split(trainval)):
    print("Fold : {}".format(index))
    
    batchsize = 16
    train_dataset = IonDataset(trainval[train_index.tolist()],  trainval_y[train_index.tolist()], flip=False, noise_level=0.0, class_split=0.0)
    train_dataloader = DataLoader(train_dataset, batchsize, shuffle=True)

    valid_dataset = IonDataset(trainval[val_index.tolist()],  trainval_y[val_index.tolist()], flip=False)
    valid_dataloader = DataLoader(valid_dataset, batchsize, shuffle=False)

    test_dataset = IonDataset(test,  test_y, flip=False, noise_level=0.0, class_split=0.0)
    test_dataloader = DataLoader(test_dataset, batchsize, shuffle=False)
    train_preds_iter = np.zeros((5000000, 11))
    test_preds_iter = np.zeros((2000000, 11))
    
    it = 0
    for it in range(1):
        device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        model=Seq2SeqRnn(input_size=trainval.shape[1], seq_len=4000, hidden_size=128, output_size=11, num_layers=2, hidden_layers=[128,128,128],
                         bidirectional=True).to(device)
    
        no_of_epochs = 200
        early_stopping = EarlyStopping(patience=200, 
                                       is_maximize=True, 
                                       checkpoint_path="./models/gru_clean_checkpoint_fold_{}_iter_{}_feat54_noise_reverse_kfold46.pt".format(index, it))
        criterion = L.FocalLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        schedular = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3, max_lr=0.001, epochs=no_of_epochs,
                                                steps_per_epoch=len(train_dataloader))
        avg_train_losses, avg_valid_losses = [], [] 
    
    
        for epoch in range(no_of_epochs):
            start_time = time.time()
            
            date_time = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
            log.write(f"{date_time} Epoch: {epoch} ... \n")
            log.write(f"{date_time} learning_rate: {schedular.get_lr()[0]:.9f} ... \n")
    
            train_losses, valid_losses = [], []
    
            model.train() # prep model for training
            train_preds, train_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
    
            for x, y in train_dataloader:
                x = x.to(device)
                y = y.to(device)
    
                optimizer.zero_grad()
                predictions = model(x[:, :trainval.shape[1], :])
    
                predictions_ = predictions.view(-1, predictions.shape[-1]) 
                y_ = y.view(-1)
    
                loss = criterion(predictions_, y_)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                schedular.step()
                # record training lossa
                train_losses.append(loss.item())
    
                train_true = torch.cat([train_true, y_], 0)
                train_preds = torch.cat([train_preds, predictions_], 0)

            model.eval() # prep model for evaluation
            val_preds, val_true = torch.Tensor([]).to(device), torch.LongTensor([]).to(device)
            with torch.no_grad():
                for x, y in valid_dataloader:
                    x = x.to(device)
                    y = y.to(device)
    
                    predictions = model(x[:,:trainval.shape[1],:])
                    predictions_ = predictions.view(-1, predictions.shape[-1]) 
                    y_ = y.view(-1)
    
                    loss = criterion(predictions_, y_)
                    valid_losses.append(loss.item())
        
                    val_true = torch.cat([val_true, y_], 0)
                    val_preds = torch.cat([val_preds, predictions_], 0)

            # calculate average loss over an epoch
            train_loss = np.average(train_losses)
            valid_loss = np.average(valid_losses)
            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)
            
            log.write(f"{date_time} train_loss: {train_loss:.6f}, valid_loss: {valid_loss:.6f} \n")

            train_score = f1_score(train_true.cpu().detach().numpy(), train_preds.cpu().detach().numpy().argmax(1), labels=list(range(11)), average='macro')
    
            val_score = f1_score(val_true.cpu().detach().numpy(), val_preds.cpu().detach().numpy().argmax(1), labels=list(range(11)), average='macro')
            
            log.write(f"{date_time} train_f1: {train_score:.6f}, valid_f1: {val_score:.6f} \n")
            #print( "train_f1: {:0.6f}, valid_f1: {:0.6f}".format(train_score, val_score))
    
            if early_stopping(val_score, model):
                log.write(f"{date_time} Early Stopping ... \n")
                log.write(f"{date_time} Best Val Score: {early_stopping.best_score:.6f} \n")
                #print("Early Stopping...")
                #print("Best Val Score: {:0.6f}".format(early_stopping.best_score))
                break
    
            #print("--- %s seconds ---" % (time.time() - start_time))
            log.write(f"{date_time} --- %s seconds ---: {time.time() - start_time} \n")
            log.write(f"{date_time} Best Val Score: {early_stopping.best_score:.6f} \n")
            
        model.load_state_dict(torch.load("./models/gru_clean_checkpoint_fold_{}_iter_{}_feat54_noise_reverse_kfold46.pt".format(index, it)))
        
        
        with torch.no_grad():
            pred_list = []
            for x, y in test_dataloader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x[:,:trainval.shape[1],:])
                predictions_ = predictions.view(-1, predictions.shape[-1]) 

                pred_list.append(F.softmax(predictions_, dim=1).cpu().numpy())
            test_preds = np.vstack(pred_list)
       
        test_preds_iter += test_preds
        test_preds_all += test_preds
        if not os.path.exists("./predictions/test"):
            os.makedirs("./predictions/test")
        np.save('./predictions/test/seq2seq_fold_{}_iter_{}_feat54_noise_reverse_kfold46.npy'.format(index, it), arr=test_preds_iter)
        np.save('./predictions/test/seq2seq_fold_{}_feat54_noise_reverse_kfold46.npy'.format(index), arr=test_preds_all)

test_preds_all = test_preds_all/np.sum(test_preds_all, axis=1)[:, None]
test_pred_frame = pd.DataFrame({'time': sample_submission['time'].astype(str),
                                'open_channels': np.argmax(test_preds_all, axis=1)})
test_pred_frame.to_csv("./seq2seq_base_feat54_noise_reverse_kfold46.csv", index=False)

