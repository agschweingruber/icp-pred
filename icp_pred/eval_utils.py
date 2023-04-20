import copy

import pandas as pd
import numpy as np
import torch
import sklearn
from tqdm import tqdm

from icp_pred.saliency import get_attr_single


def get_dl(data_module, dl_type, bs=32):
    if dl_type == "test":
        ds = data_module.test_ds
    elif dl_type == "train":
        ds = data_module.train_ds
        ds.train = False
    else:
        ds = data_module.val_ds
        
    from icp_pred.data_utils import create_dl
    dl = create_dl(ds, bs=bs)    
    return dl


def get_mean_train_target(data_module):
    return np.concatenate([pat[~torch.isnan(pat)].numpy() for pat in data_module.train_ds.targets]).mean()


@torch.inference_mode()
def make_pred_df(model, dm, dl_type="test", dl=None):
    # Eval model
    # prep
    all_targets = []
    all_preds = []
    all_times = []
    all_ids = []
    all_losses = []
    count = 0
    model.to("cuda")
    model.eval()
    # get dataloader
    max_batch_size = 4

    if dl is None:
        dl = get_dl(dm, dl_type, bs=max_batch_size)
    else:
        # normalize inputs if the dl was not used in datamodule
        if not hasattr(dl.dataset, "was_normed") or dl.dataset.was_normed == False:
            dl.dataset.inputs = [dm.preprocess(pat) for pat in dl.dataset.raw_inputs]
            dl.dataset.was_normed = True

        
    # make preds
    for inputs_raw, targets_raw, lens_raw in dl:
        bs = inputs_raw.shape[0]
        
        targets = copy.deepcopy(targets_raw)  # copy targets to allow closing the thread that loads the data
        inputs = copy.deepcopy(inputs_raw)  # copy inputs to allow closing the thread that loads the data
        lens = copy.deepcopy(lens_raw)  # copy lens to allow closing the thread that loads the data
        del inputs_raw, targets_raw, lens_raw
        # to gpu
        inputs = inputs.to("cuda")
        targets = targets.to("cuda")
        # pred
        preds = model(inputs)
        # loss
        loss = model.loss_func(preds, targets)
        # other details
        times = torch.stack([torch.arange(inputs.shape[1]) for _ in range(bs)]).unsqueeze(-1)
        ids = torch.stack([torch.ones(inputs.shape[1]) * (count + i) for i in range(bs)]).unsqueeze(-1)
        count += bs

        inputs = inputs.cpu().numpy()
        targets = torch.cat([t[:l] for t, l in zip(targets, lens)]).flatten().cpu()
        times = torch.cat([t[:l] for t, l in zip(times, lens)]).flatten().cpu()
        preds = torch.cat([t[:l] for t, l in zip(preds, lens)]).flatten().cpu()
        ids = torch.cat([t[:l] for t, l in zip(ids, lens)]).flatten().cpu()

        all_targets.append(targets)
        all_preds.append(preds)
        all_times.append(times)
        all_ids.append(ids)
        all_losses.append(loss)
    model.to("cpu")
    if dl_type == "train":
        dl.dataset.train = True

    all_losses = torch.stack(all_losses).cpu().flatten().numpy()
    all_targets = torch.cat(all_targets).cpu().flatten().numpy()
    all_preds = torch.cat(all_preds).cpu().flatten().numpy()
    all_ids = torch.cat(all_ids).cpu().flatten().numpy()
    all_times = torch.cat(all_times).cpu().flatten().numpy()
    all_errors = (all_targets - all_preds) ** 2
    

    df = pd.DataFrame({"targets": all_targets, "preds": all_preds, "ids": all_ids, "step": all_times, "error": all_errors})
    return df


def make_pred_df_xgb(model, data_module, regression, dl_type="test", dl=None):
    # get inputs+targets
    if dl is None:
        dl = get_dl(data_module, dl_type)
    else:
        dl.dataset.preprocess_all(data_module.fill_type, 
                                  median=data_module.medians, 
                                  mean=data_module.means, 
                                  std=data_module.stds,
                                  upper_quants=data_module.upper_quants,
                                  lower_quants=data_module.lower_quants)
    inputs = dl.dataset.flat_inputs
    targets = dl.dataset.flat_targets
    # get patient ids and steps
    ids = dl.dataset.ids
    steps = dl.dataset.steps
    # predict
    #print(regression, inputs.shape, targets.shape)
    if regression:
        preds = model.predict(inputs)
    else:
        preds = model.predict_proba(inputs)[:, 1]
    errors = (preds - targets) ** 2
    df = pd.DataFrame({"targets": targets, "preds": preds, "ids": ids, "step": steps, "error": errors})
    return df


def mape(targets, preds):
    target_sum = np.sum(targets)
    error_sum = np.sum(np.abs(targets - preds))
    if target_sum == 0 and error_sum == 0:
        return 0
    if target_sum == 0:
        return 1
    else:
        return error_sum / target_sum
    #epsilon = np.finfo(np.float64).eps
    #return np.mean(np.abs((preds - targets)) / np.max(np.abs(targets) + epsilon))
    
def hypertension_acc(targets, preds):
    thresh = 22
    hyper_targets = targets > thresh
    hyper_preds = preds > thresh
    hyper_acc = (hyper_targets == hyper_preds).astype(float).mean()
    return hyper_acc


from sklearn.metrics import roc_auc_score
def hypertension_auc(targets, preds):
    thresh = 22
    targets = targets > thresh
    #preds = preds > thresh
    #targets = targets.numpy()
    #preds = preds.flatten.numpy()
    auc = roc_auc_score(targets, preds)
    return auc


def hypertension_prec_rec_spec(targets, preds):
    thresh = 22
    hyper_targets = targets > thresh
    hyper_preds = preds > thresh
    CP = hyper_targets == 1
    CN = hyper_targets == 0
    TP = (hyper_targets[CP] == hyper_preds[CP]).sum()
    TN = (hyper_targets[CN] == hyper_preds[CN]).sum()
    FP = (hyper_targets[CN] != hyper_preds[CN]).sum()
    FN = (hyper_targets[CP] != hyper_preds[CP]).sum()
    sens = TP / (TP + FN) if (TP + FN) != 0 else np.nan
    spec = TN / (TN + FP) if (TN + FP) != 0 else np.nan
    prec = TP / (TP + FP) if (TP + FP) != 0 else np.nan
    rec = sens
    return prec, rec, spec


from sklearn.metrics import recall_score
def hypertension_specificity(targets, preds):
    thresh = 22
    targets = targets > thresh
    #targets = targets.numpy()
    #preds = preds.flatten.numpy()
    spec = recall_score(targets, preds, pos_label=0)
    return spec




def calc_metric(model_df):
    # calcs RMSE per patient and averages them
    return model_df.groupby("ids").apply(lambda pat: 
                                         np.sqrt(
                                             ((pat["targets"] - pat["preds"]) ** 2).mean()
                                         )
                                        ).mean()



def clip_and_denorm(dm, pat_targets, pat_preds, 
                    normalize_targets, clip_targets):
    train_mean = dm.preprocessor.mean_train_target
    train_std =  dm.preprocessor.std_train_target

    if hasattr(dm.train_ds, "upper_target_nan_quantile"):
        upper_target_nan_quantile = dm.train_ds.upper_target_nan_quantile
        lower_target_nan_quantile = dm.train_ds.lower_target_nan_quantile
        upper_target_clip_quantile = dm.train_ds.upper_target_clip_quantile
        lower_target_clip_quantile = dm.train_ds.lower_target_clip_quantile
                
    # de-standardize targets if they were standardized in training
    if normalize_targets:
        pat_targets = (pat_targets * train_std) + train_mean
        pat_preds = (pat_preds * train_std) + train_mean

    # clip targets if they were clipped in training
    if clip_targets and hasattr(dm.train_ds, "upper_target_nan_quantile"):
        # set to NaN if way too big
        pat_targets[(pat_targets > upper_target_nan_quantile) | (pat_targets < lower_target_nan_quantile)] = np.nan
        # clip if too big
        pat_targets = pat_targets.clip(lower_target_clip_quantile, upper_target_clip_quantile)
    return pat_preds, pat_targets


@torch.amp.autocast("cuda")
def make_eval_preds(models, dms, block_size, restrict_to_block_size = 0,
                    clip_targets=1, normalize_targets = 1,
                    split = "val", dl_model=True,
                    return_saliency=False,
                    deleted_feature_idcs=None,
                    deletion_prob=1.0,
                    fill_type="median",
                    verbose=True,
                    sal_kwargs=None,
                   ):
    if sal_kwargs is None:
        sal_kwargs = {}
    all_preds = []
    all_targets = []
    all_raw_inputs = []
    all_inputs = []
    all_saliencies = []
    for model, dm in zip(models, dms):
        ds = dm.val_ds # train_ds, val_ds, test_ds
        if split == "val":
            ds = dm.val_ds
        elif split == "test":
            ds = dm.test_ds
        elif split == "train":
            ds = dm.train_ds

        if dl_model:    
            model.cuda()
            for i in tqdm(range(len(ds)), disable=verbose == False):            
                pat_raw_inputs = ds.raw_inputs[i].astype(float)
                pat_inputs = ds.inputs[i].float().cuda()
                pat_targets = ds.targets[i]
                #print(pat_inputs.shape, pat_targets.shape)
                
                # if we are perturbing the input to measure robustness to features, do so here
                if deleted_feature_idcs is not None:
                    # pat_inputs shape = [seq_len, num_feats]
                    # either median or NaN
                    num_feats_deleted = len(deleted_feature_idcs)
                    if fill_type == "none":
                        fill_values = torch.zeros(num_feats_deleted) * torch.nan
                    else:
                        median = torch.tensor(dm.preprocessor.median[deleted_feature_idcs])
                        mean = torch.tensor(dm.preprocessor.mean[deleted_feature_idcs])
                        std = torch.tensor(dm.preprocessor.std[deleted_feature_idcs])
                        
                        fill_values = (median - mean) / std
                    # bring to right shape
                    fill_values = fill_values.reshape(1, num_feats_deleted).float()
                    fill_values = fill_values.repeat(pat_inputs.shape[0], 1)
                    # make deletion mask
                    deletion_mask = torch.rand_like(fill_values) <= deletion_prob
                    # replace fill_values with original values where we do not delete
                    # (do it here because pat_inpust[:, deleted_feature_idcs][deletion_mask] creates a copy of pat_inputs, so we cannot reassign)
                    fill_values[deletion_mask] = pat_inputs[:, deleted_feature_idcs][deletion_mask].cpu().float()
                    # insert
                    pat_inputs[:, deleted_feature_idcs] = fill_values.float().to(pat_inputs.device)
                
                # make prediction
                with torch.no_grad():
                    if restrict_to_block_size and len(pat_inputs) > block_size:
                        pat_preds = []
                        for j in range(0, len(pat_inputs) - block_size + 1):
                            preds =  model(pat_inputs[j: j + block_size].unsqueeze(0))
                            pat_preds.append(preds)

                        # take all preds of first batch, then only the final one
                        pat_preds = [p.squeeze() for p in pat_preds]
                        if block_size == 1:
                            pat_preds = torch.stack(pat_preds)
                        else:
                            single_preds = torch.stack([p[-1] for p in pat_preds[1:]])
                            real_preds = torch.cat([pat_preds[0], single_preds], dim=0)
                            pat_preds = real_preds
                    else:
                        pat_preds = model(pat_inputs.unsqueeze(0))
                # to cpu
                pat_preds = pat_preds.squeeze().float().cpu()
                # denorm and clip targets
                pat_preds, pat_targets = clip_and_denorm(dm, pat_targets, pat_preds, 
                                                         normalize_targets, clip_targets)
                # append to list
                if len(pat_targets) > 0 and np.isnan(pat_targets.numpy()).mean() != 1.0:
                    all_preds.append(pat_preds.numpy())
                    all_targets.append(pat_targets.squeeze().numpy())
                    all_raw_inputs.append(pat_raw_inputs)
                    all_inputs.append(pat_inputs.cpu())
                    if return_saliency:
                        model.train()
                        sal = get_attr_single(model, pat_id=None, max_len=128, pat_data=pat_inputs, **sal_kwargs)
                        all_saliencies.append(sal)
                        model.eval()
            model.cpu()
            
        else:
            inputs = ds.flat_inputs
            targets = ds.flat_targets
            # bring into right shape and avoid NaN targets
            mask = ~np.isnan(targets)
            inputs = inputs[mask]
            targets = targets[mask]
            # make preds
            preds = model.predict(inputs)
            # denorm and clip targets
            preds, targets = clip_and_denorm(dm, targets, preds, 
                                             normalize_targets, clip_targets)
            # append to list
            all_preds.append(preds)
            all_targets.append(targets)
            all_raw_inputs.append(inputs)
            
            
    out_list = [all_targets, all_preds, all_raw_inputs]
            
    if return_saliency:
        out_list.append(all_inputs)
        out_list.append(all_saliencies)
    return out_list


def calc_metrics(targets, preds):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    #targets = [t if hasattr(t, "__len__") else [t.tolist()] for t in targets]
    targets = [t if len(t.shape) > 0 else [t.tolist()] for t in targets]
    flat_targets = np.concatenate(targets)
    not_nan_mask = ~np.isnan(flat_targets)
    flat_targets = flat_targets[not_nan_mask]
    
    if isinstance(preds, float):
        flat_preds = np.ones_like(flat_targets) * preds
    else:
        preds = [t if len(t.shape) > 0 else [t.tolist()] for t in preds]
        flat_preds = np.concatenate(preds)[not_nan_mask]

    preds = flat_preds
    targets = flat_targets
    
    # clinically relevant range
    clin_min, clin_max = 10, 30
    clin_mask = (targets >= clin_min) & (targets <= clin_max)
    clin_targets = targets[clin_mask]
    clin_preds = preds[clin_mask]
    
    metrics = {"mse": mean_squared_error(targets, preds),
                "mae": mean_absolute_error(targets, preds),
                "rmse": np.sqrt(mean_squared_error(targets, preds)),
                "min_abs_error": np.abs(preds - targets).min(),
                "max_abs_error": np.abs(preds - targets).max(),
                "hyp_acc": hypertension_acc(targets, preds),
                "hyp_auc": hypertension_auc(targets, preds),
                "r2": r2_score(targets, preds),
                "rmse_clin": np.sqrt(mean_squared_error(clin_targets, clin_preds)),
                "mae_clin": mean_absolute_error(clin_targets, clin_preds),
                "hyp_acc_clin": hypertension_acc(clin_targets, clin_preds),
                "hyp_auc_clin": hypertension_auc(clin_targets, clin_preds),
              }
    return metrics, flat_targets, flat_preds



def r2_score(targets, preds, baseline_target=None):
    if baseline_target is None:
        baseline_target = np.mean(targets)
    return 1 - (np.sum((targets - preds) ** 2) / np.sum((targets - baseline_target) ** 2))


def print_all_metrics(df):
    mean_train_target = df["mean_train_target"].iloc[0]
        
    print("Performance over splits: ")
    mean_df = df.groupby("model_id").mean()[["targets", "preds", "error"]]
    print(mean_df)
    df_nona = df[~df["targets"].isna()]
    targets = df_nona["targets"]
    preds = df_nona["preds"]
    mean_pred = preds.mean()
    std_pred = preds.std()
    mean_target = targets.mean()
    std_target = targets.std()
    print("Mean train target: ", mean_train_target)
    print("Mean/Std preds: ", mean_pred, std_pred)
    print("Mean/Std targets: ", mean_target, std_target)
    print("Max error: ", np.max(df_nona["error"]))
    print("Accuracy for hypertension baseline: ", hypertension_acc(targets, np.zeros((len(targets,)))))
    print()
        
    test_target_mse = sklearn.metrics.mean_squared_error(targets, np.ones(len(targets)) * mean_target)
    pred_mse = mean_df["error"].mean()

    print("Model metrics:")
    print("RMSE: ", np.sqrt(pred_mse))
    print("MSE: ", pred_mse)
    print("MAE: ", df_nona.groupby("model_id").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], pat["preds"])).mean())
    print("MAPE: ", df_nona.groupby("model_id").apply(lambda pat: mape(pat["targets"], pat["preds"])).mean())
    #print("all R2", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)))
    print("R2 custom: ", 1 - pred_mse / test_target_mse)
    print("R2 macro: ", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)).mean())
    print("R2: ", df_nona.groupby("model_id").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)).mean())
    print("R2 old: ", sklearn.metrics.r2_score(targets, preds))
    print("Accuracy for hypertension: ", hypertension_acc(targets, preds).mean())#.groupby("model_id").apply(lambda pat: hypertension_acc(pat["targets"], pat["preds"])).mean())
    print("Precision for hypertension: ", df_nona.groupby("model_id").apply(lambda pat: hypertension_prec_rec_spec(pat["targets"], pat["preds"])[0]).mean())
    print("Recall for hypertension: ", df_nona.groupby("model_id").apply(lambda pat: hypertension_prec_rec_spec(pat["targets"], pat["preds"])[1]).mean())
    print()

    train_target_mse = df.groupby("model_id").apply(lambda pat: (pat["targets"] - mean_train_target).pow(2).mean()).mean()
    print("Mean train baseline metrics:")
    print("Mean train target:", mean_train_target)
    print("RMSE: ", np.sqrt(train_target_mse))
    print("MSE: ", train_target_mse)
    #print("Loss: ", sklearn.metrics.mean_squared_error(targets, baseline_preds))
    print("MAE: ", df_nona.groupby("model_id").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("MAPE: ", df_nona.groupby("model_id").apply(lambda pat: mape(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("R2 custom: ", 1 - train_target_mse / mean_target)
    print("R2 score: ", df_nona.groupby("model_id").apply(lambda pat: r2_score(pat["targets"], np.ones(len(pat)) * mean_train_target, baseline_target=mean_target)).mean())
    print()
    print()
    print("Scores for test target mean")
    print("Mean test target: ", mean_target)
    print("MSE: ", test_target_mse)
    print("MAE: ", df.groupby("model_id").apply(lambda pat: np.abs(pat["targets"] - mean_target).mean()).mean())
    print("R2 score: ", 0)


def print_all_metrics_pat(df):
    mean_train_target = df["mean_train_target"].iloc[0]
        
    print("Performance over splits: ")
    mean_df = df.groupby("model_id").apply(lambda model_df: model_df.groupby("ids").mean().mean())[["targets", "preds", "error"]]
    print(mean_df)
    df_nona = df[~df["targets"].isna()]
    targets = df_nona["targets"]
    preds = df_nona["preds"]
    mean_pred = mean_df["preds"].mean()
    std_pred = mean_df["preds"].std()
    mean_target = mean_df["targets"].mean()
    std_target = mean_df["targets"].std()
    print("Mean train target: ", mean_train_target)
    print("Mean/Std preds: ", mean_pred, std_pred)
    print("Mean/Std targets: ", mean_target, std_target)
    print("Max error: ", np.max(df.groupby("ids").mean()["error"]))
    print("Accuracy for hypertension baseline: ", hypertension_acc(df_nona["targets"], np.ones(len(pat))))
    print()
    
    test_target_mse = df.groupby("ids").apply(lambda pat: (pat["targets"] - mean_target).pow(2).mean()).mean()
    pred_mse = df.groupby("ids").apply(lambda pat: pat["error"].mean()).mean()


    print("Model metrics:")
    print("RMSE: ", np.sqrt(pred_mse))
    print("MSE: ", pred_mse)
    print("MAE: ", df_nona.groupby("ids").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], pat["preds"])).mean())
    print("MAPE: ", df_nona.groupby("ids").apply(lambda pat: mape(pat["targets"], pat["preds"])).mean())
    #print("all R2", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)))
    print("R2 custom: ", 1 - pred_mse / test_target_mse)
    print("R2: ", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], pat["preds"], baseline_target=mean_target)).mean())
    print("Accuracy for hypertension: ", df_nona.groupby("ids").apply(lambda pat: hypertension_acc(pat["targets"], pat["preds"])).mean())
    print("Precision for hypertension: ", df_nona.groupby("ids").apply(lambda pat: hypertension_prec_rec_spec(pat["targets"], pat["preds"])[0]).mean())
    print("Recall for hypertension: ", df_nona.groupby("ids").apply(lambda pat: hypertension_prec_rec_spec(pat["targets"], pat["preds"])[1]).mean())
    print()

    train_target_mse = df.groupby("ids").apply(lambda pat: (pat["targets"] - mean_train_target).pow(2).mean()).mean()
    print("Mean train baseline metrics:")
    print("Mean train target:", mean_train_target)
    print("RMSE: ", np.sqrt(train_target_mse))
    print("MSE: ", train_target_mse)
    #print("Loss: ", sklearn.metrics.mean_squared_error(targets, baseline_preds))
    print("MAE: ", df_nona.groupby("ids").apply(lambda pat: sklearn.metrics.mean_absolute_error(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("MAPE: ", df_nona.groupby("ids").apply(lambda pat: mape(pat["targets"], np.ones(len(pat)) * mean_train_target)).mean())
    print("R2 custom: ", 1 - train_target_mse / mean_target)
    print("R2 score: ", df_nona.groupby("ids").apply(lambda pat: r2_score(pat["targets"], np.ones(len(pat)) * mean_train_target, baseline_target=mean_target)).mean())
    print()
    print()
    print("Scores for test target mean")
    print("Mean test target: ", mean_target)
    print("MSE: ", test_target_mse)
    print("MAE: ", df.groupby("ids").apply(lambda pat: np.abs(pat["targets"] - mean_target).mean()).mean())
    print("R2 score: ", 0)



def get_all_dfs(models, data_modules, model_type, regression, dl_type="test", dl=None, norm_targets=True):
    classical_models = ["linear", "xgb", "rf"]
    if model_type in classical_models:
        dfs = [make_pred_df_xgb(model, data_module, regression, dl_type=dl_type, dl=dl) for model, data_module in zip(models, data_modules)]
    else:
        dfs = [make_pred_df(model, data_module, dl_type=dl_type, dl=dl) for model, data_module in zip(models, data_modules)]
    for i in range(len(dfs)):
        dfs[i]["model_id"] = i
        df = dfs[i]
        dm = data_modules[i]
       
        df["mean_train_target"] = dm.preprocessor.mean_train_target
        df["std_train_target"] = dm.preprocessor.std_train_target
        #print(df["mean_train_target"].mean())
    df = pd.concat(dfs)
    
    if norm_targets and regression:
        df["preds"] = (df["std_train_target"] * df["preds"]) + df["mean_train_target"]
        df["targets"] = (df["std_train_target"] * df["targets"]) + df["mean_train_target"]
    return df
