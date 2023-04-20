import os
import datetime
import copy

import sklearn
import numpy as np
from tqdm.auto import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import optuna

from icp_pred.train_utils import train_model
from icp_pred.data_utils import SeqDataModule
from icp_pred.eval_utils import get_all_dfs, make_eval_preds, calc_metrics


def obj(df, cfg, opt_object, num_seeds=1, split="val", dms=None):
    # put in op_df
    cfg = copy.deepcopy(cfg)
    if isinstance(opt_object, pd.DataFrame):
        opt_dict = opt_object.iloc[0].to_dict()
    else:
        opt_dict = opt_object
    cfg = merge_params_in_cfg(cfg, opt_dict)
    # calculate metrics for number of seeds
    metrics = []
    for seed in range(num_seeds):
        cfg["seed"] = seed
        metrics.append(train_and_eval_model(df, cfg, split=split, log=False, dms=dms))
    metric = np.mean(metrics)
    return metric


def scale_hyperparameters(opt_df):
    # scale batch size and reduction_factor exponentially
    if "reduction_factor" in opt_df:
        opt_df["reduction_factor"] = int(2 ** opt_df["reduction_factor"])
    if "bs" in opt_df:
        opt_df["bs"] = 2 ** opt_df["bs"]
    # enable nan_embed layer if we do not fill nans
    if opt_df["fill_type"] == "none":
        opt_df["use_nan_embed"] = True
    # set min len to max len
    if "max_len" in opt_df:
        opt_df["min_len"] = opt_df["max_len"]
    # convert to int
    if "flat_block_size" in opt_df:
        opt_df["flat_block_size"] =int(opt_df["flat_block_size"])
    if "n_estimators" in opt_df:
        opt_df["n_estimators"] = int(opt_df["n_estimators"])
    if "bs" in opt_df:
        opt_df["bs"] = int(opt_df["bs"])
    if "batch_size" in opt_df:
        opt_df["batch_size"] = int(opt_df["batch_size"])

    return opt_df


def train_and_eval_model(df, cfg, split, log=True, dms=None):
    #setup dm
    if dms is None:
        from icp_pred.train_utils import make_train_val_fold
        dms = make_train_val_fold(df, cfg, cfg["inner_fold"])
    # train model on datamodule
    models, _ = train_model(cfg["model_type"], dms, cfg, verbose=False, log=log)
    
    metric = eval_model_new(models, dms, cfg, split)
    #metric = eval_model(dms[0].regression, models, dms, cfg["model_type"], split, cfg["norm_targets"])
    return metric


def eval_model_new(models, dms, cfg, split):
    from icp_pred.eval_utils import make_eval_preds, calc_metrics
    
    # calc metrics
    all_metrics = []
    for model, dm in zip(models, dms):
        all_targets, all_preds, all_raw_inputs = make_eval_preds([model], [dm], cfg["block_size"],
                                                                 restrict_to_block_size = cfg["block_size"] <= 16,
                                                                 clip_targets=True, 
                                                                 normalize_targets=cfg["norm_targets"],
                                                                 split = split,
                                                                 dl_model=cfg["model_type"] in ["rnn", "transformer"],
                                                                )
        used_targets = []
        used_preds = []
        for i, t in enumerate(all_targets):
            if len(t.shape) > 0:
                used_targets.append(t)
                used_preds.append(all_preds[i])
        
        metrics, flat_targets, flat_preds = calc_metrics(used_targets, used_preds)
        #metrics, flat_targets, flat_preds = calc_metrics(all_targets, all_preds)
        all_metrics.append(metrics)
    metrics_df = pd.DataFrame(all_metrics)
    metric = metrics_df.mean()["mse"]
    return metric


def setup_dm(df, cfg):
    import pytorch_lightning as pl
    pl.utilities.seed.seed_everything(seed=cfg["seed"], workers=False)
    # create datamodule with dataloaders
    print(cfg)
    dm = SeqDataModule(df, cfg["db_name"],
                        target_name=cfg["target_name"],
                        random_starts=cfg["random_starts"], 
                        min_len=cfg["min_len"], 
                        max_len=cfg["max_len"],
                        train_noise_std=cfg["train_noise_std"], 
                        batch_size=cfg["bs"], 
                        fill_type=cfg["fill_type"], 
                        flat_block_size=cfg["flat_block_size"],
                        target_nan_quantile=cfg["target_nan_quantile"],
                        target_clip_quantile=cfg["target_clip_quantile"],
                        block_size=cfg["block_size"],
                        subsample_frac=cfg["subsample_frac"],
                        randomly_mask_aug=cfg["randomly_mask_aug"],
                        )
    dm.setup()
    return dm


def setup_dm_and_train(df, cfg, log=True):
    #setup dm
    from icp_pred.train_utils import create_dm
    dm = create_dm(df, cfg)
    # train model on datamodule
    models, trainers = train_model(cfg["model_type"], [dm], cfg, verbose=False, log=log)
    return dm, models, trainers

    

def eval_model(regression, models, data_modules, model_type, split, norm_targets):
    # make preds on val set
    pred_df = get_all_dfs(models, data_modules, model_type,
                          regression, 
                          dl_type=split, dl=None,
                          norm_targets=norm_targets)
    
    # calc target metrics
    pred_df = pred_df.dropna(subset=["targets"])
    pred_targets = pred_df["targets"]
    preds = pred_df["preds"]
    if regression:
        try:
            score = sklearn.metrics.mean_squared_error(pred_targets, preds)
            #score = sklearn.metrics.r2_score(pred_targets, preds)
        except ValueError:
            score = 200
    else:
        try:
            score = sklearn.metrics.roc_auc_score(pred_targets, preds)
        except ValueError:
            score = 0        
    return np.array([score])


def merge_params_in_cfg(cfg, params):
    cfg = copy.deepcopy(cfg)
    # put the best hyperparameters in the config
    for p in params:
        cfg[p] = params[p]
    cfg = scale_hyperparameters(cfg)
    return cfg


def train_and_test(df, cfg, num_seeds=5, return_weights=False):
    # train the model with the best hyperparameters and test it on test split
    val_scores = []
    #test_scores = []
    weights = []
    all_models = []
    dms = []
    for seed in tqdm(range(cfg["seed"], cfg["seed"] + num_seeds), desc="Training models with best parameters", disable=num_seeds==1):
        cfg["seed"] = seed
        dm, models, _ = setup_dm_and_train(df, cfg, log=False)
        #val_score = eval_model(dm.regression, models, [dm], cfg["model_type"], "val", cfg["norm_targets"])
        val_score = eval_model_new(models, [dm], cfg, split="val")
        
        if hasattr(models[0], "state_dict") and return_weights:
            model_weights = models[0].state_dict()  # get weights of the model
            model_weights = {k: v.cpu() for k, v in model_weights.items()}  # move to cpu
            weights.append(model_weights)
        val_scores.append(val_score)
        #test_scores.append(test_score)
        
        all_models.append(models[0])
        dms.append(dm)

    if return_weights:
        return val_scores, weights, all_models, dms
    else:
        return val_scores


def get_best_params(study, num_trials=5):
    trials = study.trials
    best_trials = sorted(trials, key=lambda trial: trial.value, reverse=True)[:num_trials]
    top_params = [trial.params for trial in best_trials]
    top_vals = [trial.value for trial in best_trials]
    return top_params, top_vals


def train_multiple(param_list, df, cfg):
    # train models for the best params and get the model weights
    top_param_weights = []
    top_param_val_scores = []
    for params in tqdm(param_list, desc="Training models with best parameters", disable=len(param_list)==1):
        cfg = merge_params_in_cfg(cfg, params)
        val_scores, weights, _, _ = train_and_test(df, cfg, num_seeds=1, 
                                             return_weights=True)
        top_param_val_scores.append(val_scores[0])
        top_param_weights.append(weights[0])
    return top_param_val_scores, top_param_weights


def make_optuna_foldername(cfg):
    # create folder name according to the database name, minutes, model type and date
    folder_name = f"tunings/{cfg['db_name']}_{cfg['minutes']}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{cfg['model_type']}_{cfg['opt_steps']}_inner{cfg['inner_folds']}"
    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def save_study_results(study, folder_name, model_type):
    # create a dataframe with the results
    tune_result_df = pd.DataFrame(study.trials_dataframe())
    tune_result_df.to_csv(f"{folder_name}/results.csv")
    
    print("Best study params: ", study.best_params)

    def save_plot(plot, name):
        plot.write_image(os.path.join(folder_name, name + ".png"), width=1024 * 1.5, height=800 * 1.5)

    plot = optuna.visualization.plot_slice(study)
    save_plot(plot, "slice")
    plot = optuna.visualization.plot_param_importances(study)
    save_plot(plot, "param_importances")
    plot = optuna.visualization.plot_optimization_history(study)
    save_plot(plot, "optimization_history")
    
    #if model_type in ["rnn", "gpt", "mlp"]:
    #    params = ["lr", "weight_decay", "grad_clip_val"]
        #if "bs" in study.best_params:
        #    params.append("bs")
        #params = [p for p in study.best_params]
        
    #    plot = optuna.visualization.plot_contour(study, params=params)
    #elif model_type == "xgb":
    #    plot = optuna.visualization.plot_contour(study, params=["lr", "n_estimators", "max_depth", "subsample",
    #                                                            "colsample_bytree", "gamma", "min_child_weight",
    #                                                            "flat_block_size"])
    #elif model_type == "linear":
    #    plot = optuna.visualization.plot_contour(study, params=["C", "max_iter", "flat_block_size"])
    #save_plot(plot, "contour")

def suggest_deep_learning(trial: optuna.trial.Trial):
    # suggest hyperparameters for deep learning models
    rec = {'lr': trial.suggest_loguniform("lr", 1e-5, 5e-3),
            #'min_len': trial.suggest_int("min_len", 2, 128),
            #'train_noise_std': trial.suggest_float("train_noise_std", 0.001, 0.2),
            #'train_noise_std': trial.suggest_int("train_noise_std", 0, 2),
            'weight_decay': trial.suggest_discrete_uniform("weight_decay", 0.0, 0.4, 0.02),
            'grad_clip_val': trial.suggest_discrete_uniform("grad_clip_val", 0, 1.5, 0.1), 
            
            #'fill_type': trial.suggest_categorical("fill_type", ["median", "none"]),
            #'max_epochs': trial.suggest_int("max_epochs", 5, 100),
            }
    return rec


def suggest_tree(trial):
    rec = {'lr': trial.suggest_loguniform("lr", 0.0005, 0.5),
            'n_estimators': trial.suggest_discrete_uniform("n_estimators", 10, 300, 10),
            'max_depth': trial.suggest_int("max_depth", 2, 10),
            'subsample': trial.suggest_discrete_uniform("subsample", 0.5, 1.0, 0.05),
            'colsample_bytree': trial.suggest_discrete_uniform("colsample_bytree", 0.3, 1.0, 0.05),
            'gamma': trial.suggest_discrete_uniform("gamma", 0.0, 3.0, 0.1),
            'min_child_weight': trial.suggest_discrete_uniform("min_child_weight", 0.0, 3.0, 0.1),
            #'fill_type': trial.suggest_categorical("fill_type", ["median", "none"]),
    }
    return rec


def suggest_linear(trial):
    # suggest hyperparameters for sklearn logistic regression
    #rec = {'C': trial.suggest_loguniform("C", 0.00005, 10.0),
    #       'max_iter': trial.suggest_int("max_iter", 10, 500),
    #       'l1_ratio': trial.suggest_discrete_uniform("l1_ratio", 0.0, 1.0, 0.05),}
    
    # suggest parameters for regression
    rec = {'alpha': trial.suggest_loguniform("alpha", 0.00005, 100.0),
           'l1_ratio': trial.suggest_discrete_uniform("l1_ratio", 0.0, 1.0, 0.05),}
    return rec


def objective_optuna(trial: optuna.Trial, df, cfg, dms=None):
    cfg = copy.deepcopy(cfg)

    # CAREFUL! all setting of rec must be done via tria.suggest, otherwise best params are not set.

    # Invoke suggest methods of a Trial object to generate hyperparameters.
    if cfg["model_type"] in ["rnn", "gpt", "mlp", "transformer"]:
        rec = suggest_deep_learning(trial)
        if cfg["model_type"] == "gpt":
            if cfg["gpt_name"] == "gpt2":
                rec["bs"] =  trial.suggest_int("bs", 2, 6)
            elif cfg["gpt_name"] == "gpt2-medium":
                rec["bs"] =  trial.suggest_int("bs", 2, 4)
            elif cfg["gpt_name"] == "gpt2-large":
                rec["bs"] =  trial.suggest_int("bs", 2, 3)
            elif cfg["gpt_name"] == "gpt2-xl":
                rec["bs"] =  trial.suggest_int("bs", 0, 0)
            elif cfg["gpt_name"] == "distilgpt2":
                rec["bs"] = trial.suggest_int("bs", 2, 7)
            
            #rec["reduction_factor"] = trial.suggest_discrete_uniform("reduction_factor", 1, 6, 1)
        #elif cfg["model_type"] == "transformer":
        #    rec["bs"] = trial.suggest_int("bs", 2, 6)
        #else:
        #rec["bs"] = trial.suggest_int("bs", 2, 8)
        rec["bs"] = trial.suggest_int("bs", 2, 4)
            
        if cfg["tune_masking"]:
            rec["randomly_mask_aug"] = trial.suggest_discrete_uniform("randomly_mask_aug", 0.0, 0.2, 0.02)
            
        # dropout if transformer or multilayer rnn
        if cfg["model_type"] == "transformer" or (cfg["model_type"] == "rnn" and  cfg["rnn_layers"] > 1):
            rec["dropout"] = trial.suggest_discrete_uniform("dropout", 0.0, 0.2, 0.01)
        
    else:
        if cfg["model_type"] in ("xgb", "rf"):
            rec = suggest_tree(trial)
        elif cfg["model_type"] == "linear":
            rec = suggest_linear(trial)

        if cfg["flat_block_size_range"] > 1:
            rec["flat_block_size"] =  trial.suggest_discrete_uniform("flat_block_size", 0, cfg["flat_block_size_range"], 1)

    score = obj(df, cfg, rec, num_seeds=cfg["num_seeds"], dms=dms)
    return score  

