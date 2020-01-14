import os
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from math import floor, ceil
from scipy.stats import spearmanr
from tensorflow.keras.callbacks import Callback


def color_map(val):
    if type(val) == float:
        if val <= 0:
            color = 'red'
        elif val <= 0.1:
            color = 'purple'
        elif val <= 0.3:
            color = 'orange'
        elif val >= 0.8:
            color = 'green'
        else:
            color = 'black'
    else:
        color = 'black'
    return 'color: %s' % color


def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    

def plot_metrics(history, metric_list=['loss', 'dice_coef']):
    fig, axes = plt.subplots(len(metric_list), 1, sharex='col', figsize=(24, len(metric_list)*5))
    axes = axes.flatten()
    
    for index, metric in enumerate(metric_list):
        axes[index].plot(history[metric], label='Train %s' % metric)
        axes[index].plot(history['val_%s' % metric], label='Validation %s' % metric)
        axes[index].legend(loc='best')
        axes[index].set_title(metric)

    plt.xlabel('Epochs')
    sns.despine()
    plt.show()
    

def get_metrics(train_true, train_pred, valid_true, valid_pred, target_cols):
    rho_train = np.round([spearmanr(train_true[:, ind], train_pred[:, ind] + np.random.normal(0, 1e-7, train_pred.shape[0])).correlation for ind in range(len(target_cols))], 3)
    rho_val = np.round([spearmanr(valid_true[:, ind], valid_pred[:, ind] + np.random.normal(0, 1e-7, valid_pred.shape[0])).correlation for ind in range(len(target_cols))], 3)

    metrics_df = [('Averaged', np.round(np.mean(rho_train), 3), np.round(np.mean(rho_val), 3)), 
                 ('Question averaged', np.round(np.mean(rho_train[:21]), 3), np.round(np.mean(rho_val[:21]), 3)), 
                 ('Answer averaged', np.round(np.mean(rho_train[21:]), 3), np.round(np.mean(rho_val[21:]), 3))]

    for index, col in enumerate(target_cols):
        metrics_df += [(col, rho_train[index], rho_val[index])]
        
    metrics_df = pd.DataFrame(metrics_df, columns=['Label', 'Train', 'Validation'])
    metrics_df['Var'] = metrics_df['Train'] - metrics_df['Validation']

    return metrics_df


def get_metrics_category(train_df, valid_df, target_cols, pred_cols, cat_col):
    cat_cols = train_df[cat_col].unique()
    cat_metrics = []
    for cat in cat_cols:
        train_true = train_df[train_df[cat_col] == cat][target_cols].values
        train_pred = train_df[train_df[cat_col] == cat][pred_cols].values
        valid_true = valid_df[valid_df[cat_col] == cat][target_cols].values
        valid_pred = valid_df[valid_df[cat_col] == cat][pred_cols].values
        rho_train = [spearmanr(train_true[:, ind], train_pred[:, ind] + np.random.normal(0, 1e-7, train_pred.shape[0])).correlation for ind in range(len(target_cols))]
        rho_val = [spearmanr(valid_true[:, ind], valid_pred[:, ind] + np.random.normal(0, 1e-7, valid_pred.shape[0])).correlation for ind in range(len(target_cols))]
        cat_metrics.append([rho_train, rho_val])

    
    avg_metrics = np.round(np.nanmean(cat_metrics, axis=-1), 3)
    metrics_df = pd.DataFrame({'Label': cat_cols, 'Train': avg_metrics[:, 0], 'Validation': avg_metrics[:, 1]}, dtype=float)
    metrics_df['Var'] = metrics_df['Train'] - metrics_df['Validation']
        
    return metrics_df


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def load_embeddings(path):
    with open(path,'rb') as f:
        emb_arr = pickle.load(f)
    return emb_arr


def build_matrix(word_index, path, max_features):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []
    
    for word, i in word_index.items():
        if i <= max_features:
            try:
                embedding_matrix[i] = embedding_index[word]
            except KeyError:
                try:
                    embedding_matrix[i] = embedding_index[word.lower()]
                except KeyError:
                    try:
                        embedding_matrix[i] = embedding_index[word.title()]
                    except KeyError:
                        unknown_words.append(word)
    return embedding_matrix, unknown_words


class SpearmanRhoCallback(Callback):
    def __init__(self, training_data, validation_data, model_path, monitor='val_loss', mode='min', 
                 patience=3, checkpoint=True, snapshot=False, snapshot_shift=0):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
        self.model_path = model_path
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.checkpoint = checkpoint
        self.snapshot = snapshot
        self.snapshot_shift = snapshot_shift
        self.patience_cnt = 0
        if self.mode == 'max':
            self.best_val = -float("inf")
        elif self.mode == 'min':
            self.best_val = float("inf")

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict(self.x)
        rho_train = np.mean([spearmanr(self.y[:, ind], y_pred_train[:, ind] + np.random.normal(0, 1e-7, y_pred_train.shape[0])).correlation for ind in range(y_pred_train.shape[1])])
        
        y_pred_val = self.model.predict(self.x_val)
        rho_val = np.mean([spearmanr(self.y_val[:, ind], y_pred_val[:, ind] + np.random.normal(0, 1e-7, y_pred_val.shape[0])).correlation for ind in range(y_pred_val.shape[1])])
        
        print('\rspearman-rho: %.4f val_spearman-rho: %.4f' % (rho_train, rho_val))
        logs['spearman'] = rho_train
        logs['val_spearman'] = rho_val
        
        # Early stopping monitor and checkpoint
        if (self.mode == 'max' and logs[self.monitor] >= self.best_val) or (self.mode == 'min' and logs[self.monitor] <= self.best_val):
            self.best_val = logs[self.monitor]
            if (self.snapshot) and (epoch >= self.snapshot_shift):
                complete_model_path = '%s_%d.h5' % (self.model_path, epoch)
                self.model.save_weights(complete_model_path)
                print('Saved snapshot model weights at "%s"' % complete_model_path)
            elif self.checkpoint:
                complete_model_path = '%s.h5' % self.model_path
                self.model.save_weights(complete_model_path)
                print('Saved model weights at "%s"' % complete_model_path)
        else:
            self.patience_cnt += 1
        if self.patience_cnt >= self.patience:
            print('Epoch %05d: early stopping' % epoch)
            self.model.stop_training = True
        
        return rho_val
    
    
### BERT auxiliar function
def _get_masks(tokens, max_seq_length):
    """Mask for padding"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    return [1]*len(tokens) + [0] * (max_seq_length - len(tokens))

def _get_segments(tokens, max_seq_length, ignore_first_sep=True):
    """Segments: 0 for the first sequence, 1 for the second"""
    if len(tokens)>max_seq_length:
        raise IndexError("Token length more than max seq length!")
    segments = []
    current_segment_id = 0
    for token in tokens:
        segments.append(current_segment_id)
        if token == "[SEP]":
            if ignore_first_sep:
                ignore_first_sep = False 
            else:
                current_segment_id = 1
    return segments + [0] * (max_seq_length - len(tokens))

def _get_ids(tokens, tokenizer, max_seq_length):
    """Token ids from Tokenizer vocab"""
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_ids = token_ids + [0] * (max_seq_length-len(token_ids))
    return input_ids

def _trim_input(title, question, answer, tokenizer, max_sequence_length, 
                t_max_len=30, q_max_len=239, a_max_len=239):

    t = tokenizer.tokenize(title)
    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    t_len = len(t)
    q_len = len(q)
    a_len = len(a)

    if (t_len+q_len+a_len+4) > max_sequence_length:
        
        if t_max_len > t_len:
            t_new_len = t_len
            a_max_len = a_max_len + floor((t_max_len - t_len)/2)
            q_max_len = q_max_len + ceil((t_max_len - t_len)/2)
        else:
            t_new_len = t_max_len
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
            
            
        if t_new_len+a_new_len+q_new_len+4 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (t_new_len+a_new_len+q_new_len+4)))
        
        t = t[:t_new_len]
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return t, q, a

def _convert_to_bert_inputs(title, question, answer, tokenizer, max_sequence_length, ignore_first_sep=True):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + title + ["[SEP]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length, ignore_first_sep=True)

    return [input_ids, input_masks, input_segments]

def compute_input_arays(df, columns, tokenizer, max_sequence_length, ignore_first_sep=True):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        t, q, a = instance.question_title, instance.question_body, instance.answer

        t, q, a = _trim_input(t, q, a, tokenizer, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs(t, q, a, tokenizer, max_sequence_length, ignore_first_sep=True)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]

### Using only two features as inputs
def _trim_input_2(question, answer, tokenizer, max_sequence_length, 
                q_max_len=254, a_max_len=255):

    q = tokenizer.tokenize(question)
    a = tokenizer.tokenize(answer)
    
    q_len = len(q)
    a_len = len(a)

    if (q_len+a_len+4) > max_sequence_length:
      
        if a_max_len > a_len:
            a_new_len = a_len 
            q_new_len = q_max_len + (a_max_len - a_len)
        elif q_max_len > q_len:
            a_new_len = a_max_len + (q_max_len - q_len)
            q_new_len = q_len
        else:
            a_new_len = a_max_len
            q_new_len = q_max_len
                        
        if a_new_len+q_new_len+3 != max_sequence_length:
            raise ValueError("New sequence length should be %d, but is %d" 
                             % (max_sequence_length, (a_new_len+q_new_len+4)))
        
        q = q[:q_new_len]
        a = a[:a_new_len]
    
    return q, a

def _convert_to_bert_inputs_2(question, answer, tokenizer, max_sequence_length, ignore_first_sep=True):
    """Converts tokenized input to ids, masks and segments for BERT"""
    
    stoken = ["[CLS]"] + question + ["[SEP]"] + answer + ["[SEP]"]

    input_ids = _get_ids(stoken, tokenizer, max_sequence_length)
    input_masks = _get_masks(stoken, max_sequence_length)
    input_segments = _get_segments(stoken, max_sequence_length, ignore_first_sep=True)

    return [input_ids, input_masks, input_segments]

def compute_input_arays_2(df, columns, tokenizer, max_sequence_length, ignore_first_sep=True):
    input_ids, input_masks, input_segments = [], [], []
    for _, instance in df[columns].iterrows():
        q, a = instance[columns[0]], instance[columns[1]]

        q, a = _trim_input_2(q, a, tokenizer, max_sequence_length)

        ids, masks, segments = _convert_to_bert_inputs_2(q, a, tokenizer, max_sequence_length, ignore_first_sep=True)
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)
        
    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)]