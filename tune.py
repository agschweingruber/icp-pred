import os

import hydra
import numpy as np    
import pandas as pd



@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    import logging
    logging.getLogger("lightning").setLevel(logging.ERROR)
    
    # transformers to run offline
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    # change to original working directory
    os.chdir(hydra.utils.get_original_cwd())
    cfg = dict(cfg)      
                    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg["gpu"])
    # disable wandb printing
    os.environ['WANDB_SILENT'] = "true"
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import logging
    import transformers
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

    import torch
    torch.multiprocessing.set_sharing_strategy('file_system')
    # load df
    path = f'data/DB_{cfg["db_name"]}_{cfg["minutes"]}_final_df.parquet'
    df = pd.read_parquet(path)
    print(df.shape)
    # drop columns that are completely NaN
    df = df.dropna(axis=1, how="all")
    print(df.shape)
    # drop columns that are completely zero
    mean_zeros = (df == 0).mean()
    df = df[list(mean_zeros[mean_zeros < 1.0].index)]
    print(df.shape)

    # OPTUNA
    print("Setup df...")    
    dev_df = df[df["split"] != "test"].copy()
    test_df = df[df["split"] == "test"].copy()
    # setup dms
    prepro_name = f'{cfg["seed"]}{cfg["inner_folds"]}{cfg["train_noise_std"]}{cfg["fill_type"]}{cfg["target_nan_quantile"]}{cfg["target_clip_quantile"]}{cfg["input_nan_quantile"]}{cfg["input_clip_quantile"]}{cfg["subsample_frac"]}{cfg["randomly_mask_aug"]}{cfg["agg_meds"]}{cfg["norm_targets"]}{cfg["target_name"]}'
    prepro_name = prepro_name.replace(".", "_")
    cache_path = path.replace(".parquet", f"_{prepro_name}.pickle")
    print(cache_path)
    if not os.path.exists(cache_path):
        from icp_pred.train_utils import make_train_val_fold
        dms = make_train_val_fold(dev_df, cfg, cfg["inner_folds"], test_df=test_df, seed=cfg["seed"])
        try:
            torch.save(dms, cache_path)
        except OverflowError:
            print("Could not save file because it is too large")
    else:
        try:
            dms = torch.load(cache_path)
        except RuntimeError:
            from icp_pred.train_utils import make_train_val_fold
            dms = make_train_val_fold(dev_df, cfg, cfg["inner_folds"], test_df=test_df, seed=cfg["seed"])
            torch.save(dms, cache_path)
        except FileNotFoundError:
            from icp_pred.train_utils import make_train_val_fold
            dms = make_train_val_fold(dev_df, cfg, cfg["inner_folds"], test_df=test_df, seed=cfg["seed"])
            torch.save(dms, cache_path)
        
    # test dataloader
    dm = dms[0]

    inputs, targets, lens = next(iter(dm.train_dataloader()))
    #print(dm.feature_names)
    print(inputs.shape, inputs.min(), inputs.max())
    print(targets.shape)
    print(targets[~torch.isnan(targets)].mean())
    print(lens)
    print(lens.float().mean(), lens.max())
    # NaNs in inputs
    nan_inputs = torch.isnan(inputs).any(dim=1)
    print("Nans total and percentage: ", nan_inputs.sum(), nan_inputs.sum().float() / nan_inputs.numel())
    print(cfg)
    assert cfg["fill_type"] == "none" or nan_inputs.sum() == 0

    # tune!
    print("Start tuning...")
    import optuna
    from icp_pred.tune_utils import objective_optuna
    
    study = optuna.create_study(direction="minimize")   #direction="maximize")  # Create a new study.
    study.optimize(lambda study: objective_optuna(study, df, cfg, dms=dms), 
                   n_trials=cfg["opt_steps"], gc_after_trial=True,
                   show_progress_bar=True,
                   )
    import gc
    torch.cuda.empty_cache()
    gc.collect()

    from icp_pred.tune_utils import make_optuna_foldername, save_study_results

    # create a folder to save the results
    folder_name = make_optuna_foldername(cfg)
    print("Saving in folder name: ", folder_name)
    save_study_results(study, folder_name, cfg["model_type"])
    
    
    # store best params and scores in a dataframe
    best_param_df = pd.DataFrame(study.best_params, index=[0])
    best_param_df.to_csv(f"{folder_name}/best_params.csv")

    # save cfg
    import json
    with open(f"{folder_name}/cfg.json", "w+") as f:
        json.dump(cfg, f)

    # retrain best on full dev set
    dev_df["split"] = "train"
    test_df["split"] = "val"
    dev_test_df = pd.concat([dev_df, test_df])
    from icp_pred.tune_utils import merge_params_in_cfg
    best_params = study.best_params
    best_cfg = merge_params_in_cfg(cfg, best_params)
    # print best cfg, cuda used memory and free memory
    print("Best cfg: ", best_cfg)
    print("Cuda used memory: ", torch.cuda.memory_allocated())
    print("Cuda cached memory: ", torch.cuda.memory_cached())
    from icp_pred.tune_utils import train_and_test
    dev_training_test_scores, weights, dev_models, dev_dms = train_and_test(dev_test_df, best_cfg, num_seeds=3, return_weights=True)
    # save scores 
    cross_val_score = study.best_value
    test_score = np.mean(dev_training_test_scores)
    # save as df
    scores_df = pd.DataFrame({"cross_val_score_tune": cross_val_score, "test_score": test_score,
                              "test_score_std": np.std(dev_training_test_scores)}, index=[0])
    scores_df.to_csv(f"{folder_name}/scores.csv")
    
    
    # get param count
    if cfg["model_type"] in ["rnn", "transformer"]:
        model = dev_models[0]
        count = 0
        trainable_param_count = 0
        for n, p in model.named_parameters():
            #print(n, p.shape, p.numel())
            count += p.numel()
            if p.requires_grad:
                trainable_param_count += p.numel()
        print(count, trainable_param_count)
        param_df = pd.DataFrame({"Num params": [count],
                                 "Trainable params": [trainable_param_count] 
                                },
                                index=[0],
                               )
        param_df.to_csv(f"{folder_name}/num_params.csv")
    
    ####### calculate scores properly!
    from icp_pred.eval_utils import make_eval_preds, calc_metrics
    all_metrics = []
    for dev_model, dev_dm in zip(dev_models, dev_dms):
        all_targets, all_preds, all_raw_inputs = make_eval_preds([dev_model], [dev_dm],
                                                                 best_cfg["block_size"],
                                                                 restrict_to_block_size=best_cfg["block_size"] <= 16,
                                                                 clip_targets=True, 
                                                                 normalize_targets=best_cfg["norm_targets"],
                                                                 split = "val",
                                                                 dl_model=cfg["model_type"] in ["rnn", "transformer"],
                                                                )
        used_targets = []
        used_preds = []
        for i, t in enumerate(all_targets):
            if len(t.shape) > 0:
                used_targets.append(t)
                used_preds.append(all_preds[i])
                
        
        metrics, flat_targets, flat_preds = calc_metrics(used_targets, used_preds)
        all_metrics.append(metrics)
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(f"{folder_name}/metrics_clipped.csv")
    metrics_df.mean().to_csv(f"{folder_name}/metrics_mean_clipped.csv")
    metrics_df.std().to_csv(f"{folder_name}/metrics_std_clipped.csv")

    ### without target clipping
    all_metrics = []
    for dev_model, dev_dm in zip(dev_models, dev_dms):
        all_targets, all_preds, all_raw_inputs = make_eval_preds([dev_model], [dev_dm], best_cfg["block_size"],
                                                                 restrict_to_block_size = best_cfg["block_size"] <= 16,
                                                                 clip_targets=False, normalize_targets=best_cfg["norm_targets"],
                                                                 split = "val",
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
    metrics_df.to_csv(f"{folder_name}/metrics.csv")
    metrics_df.mean().to_csv(f"{folder_name}/metrics_mean.csv")
    metrics_df.std().to_csv(f"{folder_name}/metrics_std.csv")
    
    
    # done here, after this is just commented out experimental code
    
    """
    # REMOVE FOR NOW: 
    # retrain on cross-val to get std
    from icp_pred.train_utils import train_model
    is_regression = dms[0].regression
    # train model on cross-val with best params
    models, _ = train_model(cfg["model_type"], dms, best_cfg, verbose=False, log=False)
    from icp_pred.eval_utils import get_all_dfs
    def calc_score(preds, targets):
        # drop nan values
        preds = preds[~np.isnan(targets)]
        targets = targets[~np.isnan(targets)]
        # mse error
        from sklearn.metrics import mean_squared_error
        return mean_squared_error(targets, preds)
    print(len(models), len(dms))
    pred_df_val = get_all_dfs(models, dms, cfg["model_type"], is_regression, dl_type="val", dl=None, norm_targets=cfg["norm_targets"])
    scores = pred_df_val.groupby("model_id").apply(lambda x: calc_score(x["preds"], x["targets"]))
    mean_cross_val_score_retrain = np.mean(scores)
    std_cross_val_score_retrain = np.std(scores)
    pred_df_test = get_all_dfs(models, dms, cfg["model_type"], is_regression, dl_type="test", dl=None,
                               norm_targets=cfg["norm_targets"])
    scores = pred_df_test.groupby("model_id").apply(lambda x: calc_score(x["preds"], x["targets"]))
    mean_test_score_retrain = np.mean(scores)
    std_test_score_retrain = np.std(scores)
    scores_df_retrain = pd.DataFrame({"cross_val_score_retrain": mean_cross_val_score_retrain, "cross_val_score_retrain_std": std_cross_val_score_retrain,
                                      "test_score_retrain": mean_test_score_retrain, "test_score_retrain_std": std_test_score_retrain},
                                     index=[0])
    scores_df_retrain.to_csv(f"{folder_name}/scores_retrain.csv")
    """
    
    # eval on external dataset test sets!
    if 1 == 0 and cfg["agg_meds"]:
        # DOES NOT WORK: we remove features database-dependent on too much missingness etc - would need to align datasets first...
        all_dataset_names = ["MIMIC", "eICU", "UKE"]
        externals = [d for d in all_dataset_names if d != cfg["db_name"]]
        external_eval_dict = {"db_name": [],
                           "mean_score": [],
                           "std_score": [],
                           }
        for external_db_name in externals:
            print("Evaluating on ", external_db_name)
            # load external test split
            path = f'data/DB_{external_db_name}_{cfg["minutes"]}_final_df.parquet'
            ext_df = pd.read_parquet(path)
            ext_test = ext_df[ext_df["split"] == "test"]

            ext_test["split"] = "val"
            dev_ext_test_df = pd.concat([dev_df, ext_test]).copy()
            from icp_pred.train_utils import create_dm
            dm = create_dm(dev_ext_test_df, best_cfg)
            # TODO: call setup - but take care that meds are aggregated correctly! - so somehow fit med aggregation on ext_df train data.
            test_scores = []
            for model in dev_models:
                test_score = eval_model(dm.regression, models, [dm], best_cfg["model_type"], "val", best_cfg["norm_targets"])
                test_scores.append(test_score)
            mean_test_score = np.mean(test_scores)
            std_test_score = np.std(test_scores)
            external_eval_dict["db_name"].append(external_db_name)
            external_eval_dict["mean_score"].append(mean_test_score)
            external_eval_dict["std_score"].append(std_test_score)
        external_eval_df = pd.DataFrame(external_eval_dict, index=list(range(len(external_eval_dict["db_name"]))))
        external_eval_df.to_csv(f"{folder_name}/scores_external.csv")

    return test_score


if __name__ == "__main__":
    main()
