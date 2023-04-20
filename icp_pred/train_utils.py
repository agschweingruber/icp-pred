import logging

import numpy as np
import pandas as pd
from icp_pred.model import LitRNN, LitMLP, LitCLIP, LitGPT, LitTransformer
import pytorch_lightning as pl
#from pytorch_lightning.callbacks.early_stopping import EarlyStopping
#from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import wandb
import torch

from icp_pred.data_utils import do_fold, SeqDataModule


def make_train_val_fold(df, cfg, folds, test_df=None, seed=None):
    dms = []
    if folds > 0:
        pat_list = [pat[1] for pat in df.groupby("Pat_ID")]
        
        from icp_pred.data_utils import make_fold
        train_data_list, val_data_list, _, _ = make_fold(pat_list, k=folds, seed=seed)
        
        for train_data, val_data in zip(train_data_list, val_data_list):
            train_df = pd.concat(train_data)
            train_df["split"] = "train"
            val_df = pd.concat(val_data)
            val_df["split"] = "val"
            df_list = [train_df, val_df]
            if test_df is not None:
                test_df["split"] = "test"
                df_list.append(test_df)
            df = pd.concat(df_list)
        
            dm = create_dm(df, cfg)
            dm.setup()
            dms.append(dm)
    else:
        if test_df is not None:
            test_df["split"] = "test"
            df_list = [df, test_df]
            df = pd.concat(df_list)
        dm = create_dm(df, cfg)
        dm.setup()
        dms.append(dm)
    return dms


def create_dm(df, cfg):
    cols = df.columns
    diagnose = [c for c in cols if "Diagnose" in c]
    basics = ["split", "window_id", "Pat_ID", "ICP_Vital"]
    if cfg["only_med_and_demo_features"]:
        demographics = ["Geschlecht", "Alter", "Größe", "Gewicht", "rel_time"]
        meds = [col for col in cols if "_Med" in col]
        keep_cols = basics + diagnose + demographics + meds
        df = df[keep_cols]    
        
    final_gen_feats = ['mittl_Vital',
                 'syst_Vital',
                 'rel_time',
                 'HF_Vital',
                 'PCO2_BGA',
                 'FiO2_Vital',
                 'aPTT_Labor',
                 'Lac_BGA',
                 'Thrombocyten_Labor',
                 'CK_Labor',
                 'AF_Vital']
    
    gen_feats_with_diag = ['syst_Vital',
                         'diast_Vital',
                         'PCO2_BGA',
                         'rel_time',
                         'Pupille li_Vital',
                         'PEEP_Vital',
                         'RASS_Vital',
                         'Pupille re_Vital',
                         'GCS_auge_Vital',
                         'Diagnose_ICH',
                         'CK_Labor',
                         'GCS_motor_Vital',
                         'HCO3_BGA',
                         'Diagnose_SAH',
                         'Pmean_Vital',
                         'sO2_BGA',
                         'FiO2_Vital',
                         'Diagnose_TBI',
                         'Thrombocyten_Labor',
                         'Albumin_Labor',
                         'Geschlecht',
                         'HF_Vital_std',
                         'Kreatinin_Labor',
                         'SpO2_Vital',
                         'EVB_Labor',
                         'Alter',
                         'Ca_BGA',
                         'Lac_BGA']

    if cfg["use_general_features"]:
        # kick out PEEP and GCS values because they are too different among datasets
        used_gen_feats = final_gen_feats
        used_gen_feats = [f for f in used_gen_feats if "PEEP" not in f and "GCS_" not in f and "Pupille" not in f and "Pmean" not in f]
        keep_cols = basics + diagnose + used_gen_feats
        df = df[keep_cols]
    elif cfg["use_general_features_diag"]:
        # kick out PEEP and GCS values because they are too different among datasets
        used_gen_feats = gen_feats_with_diag
        used_gen_feats = [f for f in used_gen_feats if "PEEP" not in f and "GCS_" not in f and "Pupille" not in f and "Pmean" not in f]
        keep_cols = basics + used_gen_feats
        df = df[keep_cols]
    elif cfg["use_general_features_sparse"]:
        used_gen_feats = ["syst_Vital", "RASS_Vital", "diast_Vital", "PCO2_BGA", "Lac_BGA", "HF_Vital_std"]
        keep_cols = basics + used_gen_feats
        df = df[keep_cols]
        
        
    #print(df.shape)
    dm = SeqDataModule(df,
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
                        input_nan_quantile=cfg["input_nan_quantile"],
                        input_clip_quantile=cfg["input_clip_quantile"],
                        block_size=cfg["block_size"],
                        subsample_frac=cfg["subsample_frac"],
                        randomly_mask_aug=cfg["randomly_mask_aug"],
                        agg_meds=cfg["agg_meds"],
                        norm_targets=cfg["norm_targets"],
                        add_diagnosis_features=cfg["add_diagnosis_features"],
                        )
    dm.setup()
    return dm

def retrain(dev_data, test_data, best_params, cfg, verbose=True):
    # create splits
    data_modules = do_fold(dev_data, test_data, 
                           cfg["dbs"], 
                           cfg["random_starts"], 
                           cfg["min_len"], 
                           cfg["train_noise_std"],
                           cfg["bs"],
                           cfg["fill_type"], 
                           cfg["flat_block_size"],
                           k_fold=0, 
                           num_splits=10,
                          )
    # load best params
    for key in best_params:
        cfg[key] = best_params[key]
    # retrain
    models, trainers = train_model(cfg["model_type"], data_modules, cfg, verbose=True)
    if verbose:
        # print metrics
        df = get_all_dfs(models, trainers, cfg["model_type"], dl_type="test")
        print_all_metrics(df)
        loss = df.groupby("model_id").apply(lambda model_df: model_df["error"]).mean()
        print()
        print("Loss: ", loss)
        print("Std of loss: ", df.groupby("model_id").apply(lambda model_df: model_df["error"].std()))
    return models, trainers




# define model
def create_model(model_type, data_module, cfg):
    import copy
    cfg = copy.deepcopy(cfg)
    
    # assign size
    if cfg["model_size"] is not None and model_type in ["rnn", "transformer"]:
        transformer_assignments = {"xt":    {"hidden_size": 32,  "num_transformer_blocks": 3},
                                   "tiny":  {"hidden_size": 64,  "num_transformer_blocks": 3},
                                   "small": {"hidden_size": 64,  "num_transformer_blocks": 5},
                                   "base":  {"hidden_size": 128, "num_transformer_blocks": 5},
                                   "large": {"hidden_size": 256, "num_transformer_blocks": 5},
                                   "xl":    {"hidden_size": 512, "num_transformer_blocks": 5},
                                  }
        rnn_assignments = {"xt":    {"hidden_size": 64,  "rnn_layers": 1},
                           "tiny":  {"hidden_size": 128,  "rnn_layers": 1},
                           "small": {"hidden_size": 128,  "rnn_layers": 2},
                           "base":  {"hidden_size": 256,  "rnn_layers": 2},
                           "large": {"hidden_size": 512,  "rnn_layers": 2},
                           "xl":    {"hidden_size": 1024, "rnn_layers": 2},
                           "xxl":    {"hidden_size": 2048, "rnn_layers": 2},
                          }
        if model_type == "transformer":
            assignments = transformer_assignments
        else:
            assignments = rnn_assignments
        # put in cfg
        size_assignment = assignments[cfg["model_size"]]
        for key in size_assignment:
            cfg[key] = size_assignment[key]

    
    general_keys = ["weight_decay", "max_epochs", "use_macro_loss",
                    "use_pos_weight", "use_nan_embed", "lr", "use_huber",
                    "use_static", "freeze_nan_embed",
                    "norm_nan_embed","nan_embed_size",
                    "use_nan_embed_transformer",
                    "nan_embed_transformer_n_layers", 
                    "nan_embed_transformer_n_heads",
                    "dropout", "low_mem_mode"]
    # get pos frac for weighting positive targets more
    targets = torch.cat(data_module.train_ds.targets)
    pos_frac = targets[~torch.isnan(targets)].mean()
    len_inputs = len(data_module.train_ds.inputs)
    steps_per_epoch = len_inputs // cfg["bs"] + len_inputs % cfg["bs"]
    feature_names = data_module.feature_names
    regression = data_module.regression
        
    general_args = (feature_names, regression, steps_per_epoch, pos_frac)
    general_kwargs = {key: cfg[key] for key in general_keys}
    if model_type == "rnn":
        model = LitRNN(*general_args, 
                       hidden_size=cfg["hidden_size"], 
                       rnn_layers=cfg["rnn_layers"], 
                       rnn_type=cfg["rnn_type"],
                       use_lens=cfg["use_lens"],
                       use_in_mapping=cfg["use_in_mapping"], 
                       use_out_mapping=cfg["use_out_mapping"],
                       **general_kwargs
                        )
    elif model_type  == "mlp":
        model = LitMLP(*general_args, 
                       hidden_size=cfg["hidden_size"], 
                       **general_kwargs)
    elif model_type == "clip":
        model = LitCLIP(*general_args,
                        clip_name=cfg["clip_name"],
                        **general_kwargs)
    elif model_type == "gpt":
        model = LitGPT(*general_args,
                       gpt_name=cfg["gpt_name"],
                       mode=cfg["mode"],
                       pretrained=cfg["pretrained"],
                       reduction_factor=cfg["reduction_factor"],
                       **general_kwargs)
    elif model_type == "transformer":
        
        model = LitTransformer(
                    *general_args,
                     num_transformer_blocks=cfg["num_transformer_blocks"],
                     hidden_size=cfg["hidden_size"], 
                     n_heads=cfg["n_heads"],
                     **general_kwargs
                )
    else:
        raise ValueError("Unknown model_type: " + str(model_type))
        
    #model = torch.jit.script(model)
    #model = model.to_torchscript()
    return model


def create_trainer(cfg, verbose=True, log=True):
    # default logger used by trainer
    #logger = None
    #logger = pl.loggers.mlflow.MLFlowLogger(
    #    experiment_name='default', 
    #)
    import logging
    logging.getLogger("lightning").setLevel(logging.ERROR)
    
    if log:
        wandb_logger = pl.loggers.WandbLogger(name=None, save_dir="wandb", offline=False, id=None,
                                              anonymous=None, version=None, project=cfg["target_name"],
                                              log_model=False, experiment=None, prefix='')
        hyperparam_dict = {**cfg}
        wandb_logger.log_hyperparams(hyperparam_dict)
    else:
        wandb_logger = None

    # log gradients and model topology
    #wandb_logger.watch(lit_model)

    
    # early stopping and model checkpoint
    #es = EarlyStopping(monitor='val_loss', patience=3, verbose=False, mode="min", check_on_train_epoch_end=False)
    #mc = ModelCheckpoint(monitor='val_loss', verbose=False, save_last=False, save_top_k=1,
    #                     save_weights_only=False, mode='min', period=None)
    callbacks = []
    # verbosity
    verbose_kwargs = {"enable_progress_bar": True if verbose else False,
                      "enable_model_summary": True if verbose else False,
                     }
    if not verbose:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        #pl.utilities.distributed.log.setLevel(logging.ERROR)
        
    #row_log_interval=k
    #log_save_interval=k
        
    # create trainer
    trainer = pl.Trainer(
                        enable_checkpointing=False,
                        callbacks=callbacks,
                        precision=16,
                        #precision="bf16",  # not supported on Titan V / Volta architectures, only from Ampere
                        max_steps=cfg["max_steps"],
                        gradient_clip_val=cfg["grad_clip_val"],
                        max_epochs=cfg["max_epochs"],
                        track_grad_norm=-1, # -1 to disable
                        logger=wandb_logger,
                        accelerator='gpu',
                        devices=1,
                        #val_check_interval=100,#cfg["val_check_interval"],
                        auto_lr_find=cfg["auto_lr_find"],
                        **verbose_kwargs,
                        #overfit_batches=1,
                        #limit_val_batches=0.0
    ) 
    return trainer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_nn(model_type, data_module, cfg, verbose=True, log=True):
    # init model and trainer
    model = create_model(model_type, data_module, cfg)
    # print number of trainable parameters if verbose
    if verbose:
        print("Number of trainable parameters: ", count_parameters(model))
    trainer = create_trainer(cfg, verbose=verbose, log=log)
    # track gradients etc
    if verbose and log:
        trainer.logger.watch(model)
        
    if cfg["auto_lr_find"]:
        #trainer.tune(model, data_module.train_dataloader())
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, data_module.train_dataloader(), min_lr=1e-6, max_lr=1e-2)

        # Results can be found in
        print(lr_finder.results)

        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        #model.hparams.lr = new_lr
        model.lr = new_lr

        
    # do actual training
    trainer.fit(model, data_module)
    if log:
        wandb.finish(quiet=True)
    #trainer.logger.unwatch(model)
    # load best checkpoint based on early stopping - not anymore. Disabled because it just slows down training by 5x
    #callbacks = trainer.callbacks
    #mc = [x for x in trainer.callbacks if isinstance(x, ModelCheckpoint)][0]
    #best_path = mc.best_model_path
    #model = model.load_from_checkpoint(best_path, data_module=model.data_module)
    # delete all other checkpoints
    #path_to_folder = os.path.join(*best_path.split("/")[:-1])
    #checkpoints_without_best = [cp for cp in os.listdir(path_to_folder) if cp not in best_path]
    #for cp in checkpoints_without_best:
    #    path_to_delete = os.path.join(path_to_folder, cp)
    #    os.unlink(path_to_delete)
    # plot loss curves
    """
    if verbose:
        try:
            #print("Best model path: ", mc.best_model_path)
            # show loss plot
            logger = trainer.logger

            metrics_path = f"./mlruns/1/{logger.run_id}/metrics"
            #with open(os.path.join(metrics_path, "val_loss_step"), "r")
            train_loss_step = pd.read_csv(os.path.join(metrics_path, "train_loss"), delimiter=" ", header=None, names=["time", "val", "step"])
            #pd.read_csv(os.path.join(metrics_path, "train_loss_step"), delimiter=" ", header=None, names=["time", "val", "step"])
            val_loss_epoch = pd.read_csv(os.path.join(metrics_path, "val_loss_epoch"), delimiter=" ", header=None, names=["time", "val", "step"])
            sns.lineplot(x=val_loss_epoch["step"], y=val_loss_epoch["val"], label="val")
            sns.lineplot(x=train_loss_step["step"], y=train_loss_step["val"], label="train")
            #plt.ylim(val_loss_epoch["val"].min() * 0.9, val_loss_epoch["val"].max() * 1.1)
            plt.show()
        except Exception:
            print("WARNING:", Exception, "happened")
            pass
    """
    return model, trainer


def train_classical(model_type, data_module, cfg, verbose=True):
    # prep data to have one time step as input
    inputs = data_module.train_dataloader().dataset.flat_inputs
    targets = data_module.train_dataloader().dataset.flat_targets
    # mask out NaN targets
    mask = ~np.isnan(targets)
    inputs = inputs[mask]
    targets = targets[mask]
    if verbose:
        print("Input, target shape: ", inputs.shape, targets.shape)
    # train
    if model_type == "xgb":
        from xgboost import XGBRegressor, XGBClassifier
        if data_module.regression:
            XGBClass = XGBRegressor
        else:
            XGBClass = XGBClassifier
        clf = XGBClass(n_estimators=cfg["n_estimators"],
                           max_depth=cfg["max_depth"],
                           min_child_weight=cfg["min_child_weight"],
                           gamma=cfg["gamma"],
                           subsample=cfg["subsample"],
                           colsample_bytree=cfg["colsample_bytree"],
                           tree_method=cfg["tree_method"],
                           eval_metric="logloss",
                           seed=cfg["seed"],
                           n_jobs=1,
                          )
    #tree_method, set it to hist or gpu_hist
    elif model_type == "rf":
        from xgboost import XGBRFRegressor, XGBRFClassifier
        if data_module.regression:
            XGBClass = XGBRFRegressor
        else:
            XGBClass = XGBRFClassifier
        clf = XGBClass(n_estimators=cfg["n_estimators"],
                             max_depth=cfg["max_depth"],
                             min_child_weight=cfg["min_child_weight"],
                             gamma=cfg["gamma"],
                             subsample=cfg["subsample"],
                             colsample_bytree=cfg["colsample_bytree"],
                             tree_method=cfg["tree_method"],)
    elif model_type == "linear":
        from sklearn.linear_model import LogisticRegression
        if data_module.regression:
            from cuml import ElasticNet
            clf = ElasticNet(alpha=cfg["alpha"], 
                              l1_ratio=cfg["l1_ratio"])
        else:        
            clf = LogisticRegression(penalty="elasticnet", 
                                     solver="saga",
                                     C=cfg["C"], 
                                     l1_ratio=cfg["l1_ratio"],
                                     max_iter=cfg["max_iter"])
    clf.fit(inputs, targets)
    return clf



def apply_cfg_to_dm(dm, cfg):
    dm.batch_size = cfg["bs"]
    dm.randomly_mask_aug = cfg["randomly_mask_aug"]
    dm.block_size = cfg["block_size"]
    dm.train_noise_std = cfg["train_noise_std"]
    # set block size in dm. Re-initialize in case the block size changed
    dm.set_block_size(cfg["block_size"], cfg["flat_block_size"])
    if cfg["fill_type"] == "none":
        cfg["use_nan_embed"] = True
            

def train_model(model_type, data_modules, cfg, verbose=True, log=True):
    classical_models = ["linear", "xgb", "rf"]
    models = []
    trainers = []
    for data_module in data_modules:
        # set cfg in dms (make sure that batch size etc of dms matches cfg)
        apply_cfg_to_dm(data_module, cfg)
    
        if model_type in classical_models:
            model = train_classical(model_type, data_module, cfg, verbose=verbose)
        else:
            model, trainer = train_nn(model_type, data_module, cfg, 
                                      verbose=verbose, log=log)
            trainers.append(trainer)
        # store model
        models.append(model)
        
    return models, trainers
