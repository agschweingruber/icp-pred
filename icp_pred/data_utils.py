import math
import random
from typing import Optional

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning
import numba


def get_seq_list(minutes, norm_targets, target_name, features=None, verbose=True):
    # read df
    df_path = f"data/{minutes}_/yeo_N/normalization_None/median/uni_clip_0.9999/multi_clip_N/df.pkl"
    if verbose:
        print("Reading df from: ", df_path)
    df = pd.read_pickle(df_path)
    if norm_targets:
        df[target_name] = df[target_name] / df[target_name].std()
    if features is not None:
        df = df[features + ["Pat_ID", target_name, "DB_UKE", "DB_eICU", "DB_MIMIC"]]
    # turn into seq list
    seq_list = [df[df["Pat_ID"] == pat_id].drop(columns=["Pat_ID"]) for pat_id in sorted(df["Pat_ID"].unique())]
    return seq_list


def do_fold(dev_data, test_data, dbs, random_starts, min_len, train_noise_std, batch_size, fill_type, flat_block_size, k_fold=0, num_splits=1):
    # do k fold
    if k_fold > 1:
        train_data_list, val_data_list, train_idcs, val_idcs = make_fold(dev_data, k=k_fold)
    else:
        # do train/val split
        train_data_list, val_data_list = [], []
        for i in range(num_splits):
            train_data, val_data, train_idcs, val_idcs = make_split(dev_data, test_size=0.2)
            train_data_list.append(train_data)
            val_data_list.append(val_data)
    # create data modules
    data_modules = [SeqDataModule(train_data, val_data, test_data, 
                                  dbs, 
                                  random_starts=random_starts, 
                                  min_len=min_len, 
                                  train_noise_std=train_noise_std, 
                                  batch_size=batch_size, 
                                  fill_type=fill_type, 
                                  flat_block_size=flat_block_size) 
                    for train_data, val_data in zip(train_data_list, val_data_list)]
    return data_modules




def make_fold(seq_list, k=5, seed=None):
    seq_list = np.array(seq_list, dtype=object)
    labels = create_seq_labels(seq_list)
    folder = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    splits = list(folder.split(seq_list, labels))
    train_idcs = [split[0] for split in splits]
    val_idcs = [split[1] for split in splits]
    train_data = [seq_list[idcs] for idcs in train_idcs]
    val_data = [seq_list[idcs] for idcs in val_idcs]
    
    print("Train data: ", len(train_data))
    print("Val data: ", len(val_data))
    print("Mean len train: ", np.mean([len(seq) for seq in train_data]))
    print("Mean len val: ", np.mean([len(seq) for seq in val_data]))
    return train_data, val_data, train_idcs, val_idcs


def create_seq_labels(seq_list, target_name="ICP_Vital"):
    median_len = np.median([len(pat) for pat in seq_list])
    median_target = np.median([seq[target_name][~seq[target_name].isna()].mean() for seq in seq_list])
    #print("Mean len: ", mean_len)
    #print("Mean target: ", mean_target)
    labels =  [(len(seq) < median_len).astype(int).astype(str) +
               ((seq[target_name][~seq[target_name].isna()].mean() < median_target).astype(int).astype(str))
               for seq in seq_list]
    return labels

def make_split(seq_list, test_size=0.2, seed=1):
    indices = np.arange(len(seq_list))
    labels = create_seq_labels(seq_list)
    train_data, val_data, train_idcs, val_idcs = train_test_split(seq_list, indices, test_size=test_size,
                                                                  stratify=labels, shuffle=True,
                                                                  random_state=seed)
    return train_data, val_data, train_idcs, val_idcs


def create_dl(ds, bs=32, shuffle=False):
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True, collate_fn=seq_pad_collate, persistent_workers=0)            
    return dl


def to_torch(seq_list):
    return [torch.from_numpy(seq.to_numpy()).clone().float() for seq in seq_list]


def seq_pad_collate(batch):
    inputs = [b[0] for b in batch]
    targets = [b[1].unsqueeze(-1) for b in batch]
    lens = torch.tensor([len(seq) for seq in inputs])
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    #packed_inputs = pack_padded_sequence(padded_inputs, lens, batch_first=True, enforce_sorted=False)

    padded_targets = pad_sequence(targets, batch_first=True, padding_value=math.nan)
    return [padded_inputs, padded_targets, lens]


def make_flat_inputs(inputs, flat_block_size, fill_type, median, max_len, targets=None):
    # clip max len
    if max_len > 0:
        inputs = [seq[:max_len] for seq in inputs]

    if flat_block_size > 0:
        # make blocks
        num_feats = inputs[0].shape[-1]
        all_blocks = []
        for pat_idx, pat in enumerate(inputs):
            pat = pat.numpy()
           
            for time_idx in range(len(pat)):
                # skip block for training if targets are not given
                if targets is not None and np.isnan(targets[pat_idx][time_idx]):
                    continue
                
                end_idx = time_idx + 1
                start_idx = max(end_idx - flat_block_size, 0)
                block = pat[start_idx: end_idx]
                # pad at start if block is too short
                size_diff = flat_block_size - block.shape[0]
                if size_diff > 0:
                    if fill_type == "none":
                        pad_prefix = np.zeros((size_diff, num_feats)) * np.nan
                    else:
                        pad_prefix = np.full((size_diff, num_feats), median)
                    block = np.concatenate([pad_prefix, block], axis=0)
                # make flat
                block = block.reshape(-1).copy()

                all_blocks.append(block)
        inputs = np.stack(all_blocks)
    else:
        # make flat
        inputs = np.concatenate(inputs, axis=0)
        if targets is not None:
            targets = np.concatenate(targets, axis=0)
            inputs = inputs[~np.isnan(targets)]
    return inputs


@numba.jit()
def ema_fill(pat: np.ndarray, ema_val: float, mean: np.ndarray):
    # init ema
    ema = np.ones_like(pat[0]) * mean
    # run ema
    ema_steps = np.ones_like(pat)
    for i, pat_step in enumerate(pat):
        pat_step[np.isnan(pat_step)] = 0
        ema = ema_val * ema + (1 - ema_val) * pat_step
        ema_steps[i] = ema.copy()
    return ema_steps


@numba.jit()
def ema_fill_mask(pat: np.ndarray, ema_val: float, mean: np.ndarray):
    # init ema
    ema = np.ones_like(pat[0]) * mean
    # run ema
    ema_steps = np.ones_like(pat)
    for i, pat_step in enumerate(pat):
        mask = np.isnan(pat_step)
        ema[~mask] = ema_val * ema[~mask] + (1 - ema_val) * pat_step[~mask]
        ema_steps[i] = ema.copy()
    return ema_steps



class Preprocessor():
    def __init__(self, fill_type, agg_meds, input_nan_quantile,
                 input_clip_quantile) -> None:
        self.fill_type = fill_type
        self.agg_meds = agg_meds
        self.med_mapping = None
        self.input_nan_quantile = input_nan_quantile
        self.input_clip_quantile = input_clip_quantile
        if self.agg_meds:
            import json
            with open("data/measure_norm_mapping_dict.json", "r") as f:
                self.med_mapping = json.load(f)
    
    def fit(self, df: pd.DataFrame, input_cols) -> None:
        X, y = df[input_cols], df["target"]
        
        self.feature_names = input_cols
        if self.med_mapping is not None:
            # fit
            self.all_med_names, self.med_stds = self.fit_med_agg(X)
            # apply
            X = self.apply_med_agg(X)
            # redo feature_names
            self.feature_names = X.columns
                        
        # clamp extremas according to pre-calculated vals and store them
        # calc quantiles according to train data
        self.upper_quants_nan = (X.quantile(self.input_nan_quantile, axis=0)).to_numpy()
        self.upper_quants_clip = (X.quantile(self.input_clip_quantile, axis=0)).to_numpy()
        self.lower_quants_nan = (X.quantile(1 - self.input_nan_quantile, axis=0)).to_numpy()
        self.lower_quants_clip = (X.quantile(1 - self.input_clip_quantile, axis=0)).to_numpy()
        # apply clipping to train data
        X_nan_clipped = self.apply_quants((X.to_numpy()))
                
        # calc mean, median, std
        self.mean, self.median, self.std = np.nanmean(X_nan_clipped, axis=0), np.nanmedian(X_nan_clipped, axis=0), np.nanstd(X_nan_clipped, axis=0)
        #self.mean, self.median, self.std = (torch.tensor(x).float() for x in (self.mean, self.median, self.std))
        self.mean_train_target = y.mean()
        self.mean_train_target_pat = df.groupby("Pat_ID").apply(lambda pat: pat["target"].mean()).mean()
        
        # calc std target
        self.std_train_target = y.std()
        
        # create tokenizer for gpt if needed
        create_tokenier = False
        if create_tokenier:
            # todo: in preprocess all, we want to apply the procedure below to fit the kmeans tokenizer on train data
            # TODO: then we want to apply kmeans tokenizer to all input steps
            # TODO: finally we need to adjust the GPT model creation - we now just need to reinitialize the dictionary at the start
            # TODO: ! make sure that this dictionary is set to trainable in apply_train_mode, as the "adapters" setting will set it to requires_grad=False

            filled_df = df.copy().fillna(df.median())
            cluster_df = filled_df.copy().drop(columns=["split"])
            cluster_df = cluster_df.drop(columns=["Pat_ID"])
            # create 4 time steps for each Pat_ID that averages the steps between 0-25%, 25-50, 50-75, 75-100%
            def summarize_pat_generalized(pat, num_partitions=4):
                out = []
                for i in range(num_partitions):
                    out.append(pat.iloc[int(len(pat) * i / num_partitions):int(len(pat) * (i + 1) / num_partitions)].mean(axis=0))
                return pd.DataFrame([o.to_numpy() for o in out], columns=pat.columns)#.drop(columns=["Pat_ID"]).columns)

            cluster_df = cluster_df.groupby("Pat_ID").apply(lambda x: summarize_pat_generalized(x, 4) if len(x) >= 4 else None)# x.sample(4, random_state=cfg["seed"]) if len(x) > 4 else None)
            cluster_df = cluster_df.reset_index(drop=True).drop(columns=["Pat_ID"])

            # import kmeans from sklearn
            from sklearn.cluster import KMeans
            # create kmeans model
            kmeans = KMeans(n_clusters=100, random_state=0).fit(cluster_df)
            # get cluster centroids
            centroids = kmeans.cluster_centers_
            centroid_df = pd.DataFrame(centroids, columns=cluster_df.columns)
            # predict labels for cluster df
            labels = kmeans.predict(cluster_df)
    
    def transform(self, pat):
        pat = self.apply_med_agg(pat)
        pat = self.apply_quants(pat)
        pat = self.fill_pat(pat)
        pat = (pat - self.mean) / self.std
        return torch.from_numpy(pat.to_numpy())
    
    def fit_med_agg(self, X):
        # map single meds to their med group. First divide each by their individual std, then sum up
        all_med_names = []
        for group_name, names in self.med_mapping.items():
            # add _Med as suffix to all med names
            names = [name + "_Med" for name in names]
            names = [n for n in names if n in self.feature_names]

            self.med_mapping[group_name] = names
            if len(names) > 0:
                all_med_names.extend(names)
        # clean empty groups
        for group_name in list(self.med_mapping.keys()):
            if len(self.med_mapping[group_name]) == 0:
                del self.med_mapping[group_name]
        # get std
        med_stds = X[all_med_names].std(axis=0)
        return all_med_names, med_stds
    
    def apply_med_agg(self, pat):
        if self.med_mapping is not None:
            # TODO: for external test datasets: if  self.all_med_names are not all in pat, then recalculate them and the med_stds 
            
            meds = pat[self.all_med_names]
            # divide by std
            meds = meds / self.med_stds
            #pat.loc[:, self.all_med_names] = pat[self.all_med_names].copy() / self.med_stds
            # create new columns with mean of meds
            new_col_dict = {}
            for med_name, med_names in self.med_mapping.items():
                new_col_dict[med_name] = meds[med_names].mean(axis=1)
            new_col_df = pd.DataFrame(new_col_dict)
            # merge new columns with pat
            pat = pd.concat([pat, new_col_df], axis=1)
                            
            # delete old columns
            pat = pat.drop(columns=self.all_med_names)
        return pat
    
    def apply_quants(self, pat):
        pat[(pat > self.upper_quants_nan) | (pat < self.lower_quants_nan)] = np.nan
        pat = pat.clip(self.lower_quants_clip, self.upper_quants_clip)
        return pat
    
    def transform_target(self, target):
        return (target - self.mean_train_target) / self.std_train_target
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)
    
    def fill_pat(self, pat: pd.DataFrame):
        if self.fill_type is None or self.fill_type.lower() == "none":
            return pat
        
        #pat = pat.numpy().astype(np.float32)
        median = self.median#.numpy().astype(np.float32)
        
        nan_mask = np.isnan(pat)
        if self.fill_type == "pat_median":
            count = (~nan_mask).cumsum(axis=0) + 1
            # calc cumsum without nans
            median_filled = pat.copy()
            median_filled[nan_mask] = np.expand_dims(median, 0).repeat(pat.shape[0], axis=0)[nan_mask]
            #median.repeat(pat.shape[0], 1)[nan_mask]
            cumsum = median_filled.cumsum(axis=0)
            # calc mean until step:
            mean_until_step = cumsum / count
            # fill mean until step
            pat[nan_mask] = mean_until_step[nan_mask]            
        elif self.fill_type == "pat_ema":        
            ema_val = 0.9
            pat = ema_fill(pat, ema_val, median)
        elif self.fill_type == "pat_ema_mask":
            ema_val = 0.3
            pat = ema_fill_mask(pat, ema_val, median)
        elif self.fill_type == "ffill":
            pat = pat.ffill()

        # always fill remaining NaNs with the median
        # first extend median to same shape as pat
        
        median_series = pd.Series(median, index=pat.columns)
        #print("Median: ", median_series)
        pat = pat.fillna(median_series)
       
        #median_filled = np.expand_dims(median, 0).repeat(pat.shape[0], axis=0)  
      
        #nan_mask = torch.isnan(pat)
        #pat[nan_mask] = median.repeat(pat.shape[0], 1)[nan_mask]
        
        assert pat.isna().sum().sum() == 0, "NaNs still in tensor after filling!"
        return pat
    

class SeqDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, 
                 df,
                 batch_size: int = 32, 
                 random_starts=False, 
                 min_len=10, 
                 train_noise_std=0.0, 
                 fill_type="pat_ema", 
                 flat_block_size=0, 
                 target_name = "ICP_Vital", 
                 target_nan_quantile=0,
                 target_clip_quantile=0,
                 input_nan_quantile=0,
                 input_clip_quantile=0,
                 block_size=0,
                 max_len=0,
                 subsample_frac=1.0,
                 randomly_mask_aug=0.0,
                 agg_meds=False,
                 norm_targets=True,
                 add_diagnosis_features=False,
                ):
        super().__init__()
        
        # dataset choice
        self.target_name = target_name
        self.norm_targets = norm_targets
        # hyperparams
        self.random_starts = random_starts
        self.train_noise_std = train_noise_std
        self.min_len = min_len
        self.batch_size = batch_size
        self.fill_type = fill_type
        self.flat_block_size = flat_block_size
        self.target_nan_quantile = target_nan_quantile
        self.target_clip_quantile = target_clip_quantile
        self.block_size = block_size
        self.max_len = max_len
        
        # construct targets and drop respective columns
        diagnose_cols = [col for col in df.columns if "diagnose" in col.lower()]
        
        #if target_name in ["VS", "CI"]:
        #    label_df = pd.read_csv("data/sah_labels.csv").drop_duplicates("Patienten ID")
        # SAH needs to be done in preprocesing notebook because it happens at one specific time.    
            
        if target_name in diagnose_cols:
            self.regression = False
            df["target"] = df[target_name]
            df = df.drop(columns=diagnose_cols)
        elif target_name == "ICP_Vital":
            self.regression = True
            # remove CPP as it is dependent on ICP and we want to predict ICP
            icp_cols = [col for col in df.columns if "ICP" in col]
            cpp_cols = [col for col in df.columns if "CPP" in col]
            drop_cols = icp_cols + cpp_cols
            if not add_diagnosis_features:
                drop_cols += diagnose_cols
            df["target"] = df[target_name]
            df = df.drop(columns=drop_cols)
        elif target_name.startswith("long_icp_hypertension"):
            self.regression = False
            hyper_thresh = 22.0 # value above which ICP val is considered critical
            hours_thresh = 2 # number of hours to be above to be considered a long phase
            hours_forecasted = int(target_name.split("_")[-1])  # e.g. "long_icp_hypertension_2"

            def add_hypertension_target(seq, hyper_thresh, hours_thresh, hours_forecasted):
                seq = seq.sort_values("rel_time", ascending=True)
                targets = []
                tension_count = 0
                
                # choose columns to define hypertension
                if "ICP_Vital_max" in seq:
                    icp_vals = seq["ICP_Vital_max"]
                #elif "ICP_Vital_std" in seq:
                #    icp_vals = seq["ICP_Vital"] + 1.96 * np.nan_to_num(seq["ICP_Vital_std"].to_numpy())
                else:
                    icp_vals = seq["ICP_Vital"]
                
                # define hypertension
                for icp_val in icp_vals.iloc[hours_forecasted:]:
                    if icp_val > hyper_thresh:
                        tension_count += 1
                    else:
                        tension_count = 0
                    target = float(tension_count > hours_thresh)
                    targets.append(target)
                targets.extend([np.nan] * min(hours_forecasted, len(seq)))
                return pd.Series(targets)

            targets = df.groupby("Pat_ID").apply(lambda pat: 
                                                 add_hypertension_target(pat, hyper_thresh, hours_thresh, hours_forecasted))
            # assign to column
            df["target"] = list(targets)
            # drop columns
            if add_diagnosis_features:
                drop_cls = []
            else:
                drop_cols = diagnose_cols 
            df = df.drop(columns=drop_cols)
        elif target_name in df.columns:
            if target_name == "Geschlecht":
                self.regression = False
            else:
                self.regression = True
            df["target"] = df[target_name]
            drop_cols = [target_name]
            if not add_diagnosis_features:
                drop_cols += diagnose_cols
            df = df.drop(columns=[target_name])
        
        # apply splits
        self.df = df.copy()
        train_data = df[df["split"] == "train"].drop(columns=["split"])
        val_data = df[df["split"] == "val"].drop(columns=["split"])
        test_data = df[df["split"] == "test"].drop(columns=["split"])
        
        
        non_input_cols = ["Pat_ID", "target", "window_id"]
        input_cols = [col for col in train_data.columns if col not in non_input_cols]
        
        # create preprocessor and fit on train data
        self.preprocessor = Preprocessor(fill_type, agg_meds, input_nan_quantile=input_nan_quantile,
                 input_clip_quantile=input_clip_quantile)
        
        self.preprocessor.fit(train_data, input_cols)

        # create datasets
        self.train_ds = SequenceDataset(train_data, train=True, regression=self.regression,
                                        random_starts=self.random_starts,
                                        block_size=self.block_size,
                                        min_len=self.min_len, 
                                        max_len=self.max_len,
                                        train_noise_std=self.train_noise_std,
                                        flat_block_size=self.flat_block_size,
                                        target_nan_quantile=self.target_nan_quantile,
                                        target_clip_quantile=self.target_clip_quantile,
                                        subsample_frac=subsample_frac,
                                        randomly_mask_aug=randomly_mask_aug,
                                        preprocessor=self.preprocessor) if train_data is not None else None
        self.val_ds = SequenceDataset(val_data, train=False, regression=self.regression, random_starts=False, block_size=0, 
                         train_noise_std=0.0, flat_block_size=self.flat_block_size,
                                     max_len=self.max_len,) if val_data is not None else None
        if test_data is not None and len(test_data) > 0:
            self.test_ds = SequenceDataset(test_data, train=False, regression=self.regression, random_starts=False, block_size=0, 
                         train_noise_std=0.0, flat_block_size=self.flat_block_size,
                                      max_len=self.max_len,)
        else: 
            self.test_ds = None
        self.datasets = [ds for ds in [self.train_ds, self.val_ds, self.test_ds] if ds is not None]
        
        self.feature_names = self.train_ds.feature_names
        self.setup_completed = False
                 
    def get_preprocessor(self):
        return self.preprocessor
        
    def setup(self, stage: Optional[str] = None): 
        # this method should only be called shortly before training
        if not self.setup_completed:
            # (yeo not implemented as on UMAP it just looks worse than not using it)
            # potentially get yeo-john lambdas from train dataset
            # potentially apply lambdas to all datasets
            for ds in self.datasets:
                # apply preprocessor
                ds.inputs = [self.preprocessor.transform(pat) for pat in ds.raw_inputs]
                if self.norm_targets and self.regression:
                    ds.targets = [self.preprocessor.transform_target(t) for t in ds.raw_targets]
                else:
                    ds.targets = ds.raw_targets
                ds.make_flat_arrays(self.preprocessor.fill_type, self.preprocessor.median, flat_block_size=self.flat_block_size)
                ds.feature_names = self.preprocessor.feature_names
                self.feature_names = ds.feature_names
            # done
            self.setup_completed = True
             
    def make_flat_arrays(self):
        for ds in self.datasets:
            ds.make_flat_arrays(self.preprocessor.fill_type, self.preprocessor.median, flat_block_size=self.flat_block_size)

    def train_dataloader(self):
        return create_dl(self.train_ds, bs=self.batch_size, shuffle=True) if self.train_ds is not None else None

    def val_dataloader(self):
        eval_bs = max(self.batch_size // 8, 4)
        return create_dl(self.val_ds, bs=eval_bs) if self.val_ds is not None else None

    def test_dataloader(self):
        eval_bs = max(self.batch_size // 8, 4)
        return create_dl(self.test_ds, bs=eval_bs) if self.test_ds is not None else None
    
    def set_block_size(self, block_size, flat_block_size):
        flat_size_changed = self.flat_block_size != flat_block_size
        self.flat_block_size = flat_block_size
        self.block_size = block_size
        for ds in self.datasets:
            ds.block_size = block_size
            ds.flat_block_size = flat_block_size
        if flat_size_changed:
            self.make_flat_arrays()

        
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, data, train, regression, random_starts=1, block_size=0, min_len=10, train_noise_std=0.1, 
                 flat_block_size=0, target_nan_quantile=0, target_clip_quantile=0, max_len=0, 
                 subsample_frac=1.0, randomly_mask_aug=0, preprocessor=None):
        self.random_starts = random_starts
        self.min_len = min_len
        self.max_len = max_len
        self.block_size = block_size
        self.train_noise_std = train_noise_std
        self.train = train
        self.flat_block_size = flat_block_size
        self.target_nan_quantile = target_nan_quantile
        self.subsample_frac = subsample_frac
        self.randomly_mask_aug = randomly_mask_aug
        self.preprocessor = preprocessor
        self.regression = regression
        
        if train and regression:
            if target_nan_quantile > 0:
                data = data.copy()
                self.upper_target_nan_quantile = data["target"].quantile(target_nan_quantile)
                self.lower_target_nan_quantile = data["target"].quantile(1 - target_nan_quantile)
                data.loc[data["target"] > self.upper_target_nan_quantile, "target"] = np.nan
                data.loc[data["target"] < self.lower_target_nan_quantile, "target"] = np.nan
                
            if target_clip_quantile > 0:
                data = data.copy()
                self.upper_target_clip_quantile = data["target"].quantile(target_clip_quantile)
                self.lower_target_clip_quantile = data["target"].quantile(1 - target_clip_quantile)
                data.loc[data["target"] > self.upper_target_clip_quantile, "target"] = self.upper_target_clip_quantile
                data.loc[data["target"] < self.lower_target_clip_quantile, "target"] = self.lower_target_clip_quantile
        
        # copy each sequence to not modify it outside of here
        # group by patient_id and window_id and make list out of it
        grouped = data.groupby(["Pat_ID", "window_id"])
        list_of_windows = []
        for _, group in grouped:
            list_of_windows.append(group.copy().drop(columns=["Pat_ID", "window_id"]))
        data = list_of_windows
        # subsample part of patients
        if subsample_frac < 1.0:
            data = [data[i] for i in np.random.choice(range(len(data)), int(len(data) * subsample_frac), replace=False)]
        self.raw_inputs = ([p.drop(columns=["target"]) for p in data])
        self.raw_targets = to_torch([p["target"] for p in data])
        
        self.feature_names = list(data[0].drop(columns=["target"]).columns)
        
        lens = [len(pat) for pat in self.raw_inputs]
        max_len = 99999999 if self.max_len == 0 else self.max_len
        self.ids = np.concatenate([([i] * lens[i])[:max_len] for i in range(len(lens))])
        self.steps = np.concatenate([np.arange(lens[i])[:max_len] for i in range(len(lens))])
        
        self.all_preprocessed = False

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = self.inputs[index]
        y = self.targets[index]
        seq_len = len(x)
        
        if self.train:
            if self.block_size:
                total_len = self.block_size
                start_idx = random.randint(0, seq_len - total_len) if seq_len > total_len else 0
                end_idx = start_idx + total_len
                x = x[start_idx: end_idx]
                y = y[start_idx: end_idx]
            else:
                start_idx = 0
                
        if self.max_len > 0:
            x = x[: self.max_len]
            y = y[: self.max_len]
        
        # augment data
        if self.train_noise_std:
            x = torch.normal(x, self.train_noise_std)
            
        if self.randomly_mask_aug:
            # set parts randomly to NaN
            mask = torch.rand(x.shape, device=x.device, dtype=x.dtype) < self.randomly_mask_aug
            x[mask] = torch.nan
            # fill 
            x = pd.DataFrame(x.numpy(), columns=self.feature_names)
            x = self.preprocessor.fill_pat(x)
            x = torch.from_numpy(x.to_numpy())
            
        return x.float(), y.float()
        
    def make_flat_arrays(self, fill_type, median, flat_block_size=None):
        if flat_block_size is None:
            flat_block_size = self.flat_block_size
        max_len = 0 if self.train else self.max_len
        self.flat_inputs = make_flat_inputs(self.inputs, self.flat_block_size, fill_type, median, max_len)
        self.flat_targets = np.concatenate([data[:max_len] if max_len > 0 else data for data in self.targets], axis=0)
