import copy
from typing import Optional, Tuple
import math

import numpy as np
import pytorch_lightning as pl
import sklearn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LitSeqModel(pl.LightningModule):
    def __init__(self, feature_names, regression, steps_per_epoch, pos_frac,
                 max_epochs=5, weight_decay=0.1, use_macro_loss=True, 
                 use_pos_weight=True, use_nan_embed=False, lr=0.001, use_huber=False,
                 use_static=True, freeze_nan_embed=False, norm_nan_embed=False, 
                 nan_embed_size=512, use_nan_embed_transformer=False, 
                 nan_embed_transformer_n_layers=3, 
                 nan_embed_transformer_n_heads=8, dropout=0.2, low_mem_mode=False):
        super().__init__()        
        self.regression = regression
        self.use_macro_loss = use_macro_loss
        self.use_nan_embed = use_nan_embed
        self.lr = lr
        self.use_huber = use_huber
        self.use_static = use_static
        self.freeze_nan_embed = freeze_nan_embed
        self.nan_embed_size = nan_embed_size
        self.norm_nan_embed = norm_nan_embed
        self.dropout = dropout
        
        # get feature names
        feature_names = feature_names
        
        if use_static:
            # save idcs of static idcs to have separate streams in model
            static_names = ['Geschlecht', 'Alter', 'Größe', 'Gewicht']
            static_idcs = [i for i, name in enumerate(feature_names)
                        if name in static_names
                        or name.startswith("Diagnose")
                        or name.startswith("DB_")]
            non_static_idcs = [i for i in range(len(feature_names)) if i not in static_idcs]
            self.register_buffer("static_idcs", torch.tensor(static_idcs))
            self.register_buffer("recurrent_idcs", torch.tensor(non_static_idcs))
            self.num_recurrent_inputs = len(self.recurrent_idcs)
            self.num_static_inputs = len(self.static_idcs)
        else:
            self.num_recurrent_inputs = len(feature_names)
            self.num_static_inputs = 0
            self.static_idcs = None
            self.recurrent_idcs = list(range(len(feature_names)))
        self.num_inputs = len(feature_names)
        
        # define loss 
        self.pos_weight = (1 - pos_frac) / pos_frac if use_pos_weight else None
        self.loss_func = SequentialLoss(self.regression, self.use_macro_loss, self.pos_weight, self.use_huber)
        
        if self.use_nan_embed:
            from icp_pred.nan_emb import NanEmbed, NanEmbedTransformer
            emb_size = self.nan_embed_size
            if use_nan_embed_transformer:
                self.embed = NanEmbedTransformer(self.num_inputs, emb_size, low_mem_mode,
                n_layers=nan_embed_transformer_n_layers, n_heads=nan_embed_transformer_n_heads, dropout=self.dropout)
            else:
                self.embed = NanEmbed(self.num_inputs, emb_size, low_mem_mode)

            if self.freeze_nan_embed:
                for p in self.embed.parameters():
                    p.requires_grad = False

            if self.norm_nan_embed:
                self.embed = torch.nn.Sequential(self.embed, torch.nn.LayerNorm(emb_size))
            
            self.num_recurrent_inputs = emb_size
            self.num_static_inputs = 0
            self.static_idcs = None
            self.recurrent_idcs = torch.tensor(list(range(emb_size)))
        
        
        self._average_model = None
        
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        self.weight_decay = weight_decay
        
    def forward(self, x, *args, **kwargs):
        if self.use_nan_embed:
            x = self.embed(x)
            #x_emb = self.embed(x.reshape(-1, x.shape[-1]))#[:, 0])
            #x = x_emb.reshape(*x.shape[:-1], x_emb.shape[-1])
        preds = self.make_preds(x, *args, **kwargs)

        return preds
    
    def predict(self, x, *args, **kwargs):
        preds =  self.forward(x, *args, **kwargs)
        if not self.regression:
            preds = torch.sigmoid(preds)
        return preds
    # custom methods

    # PT-lightning methods
    def training_step(self, batch, batch_idx):
        hiddens = None
        inputs, targets, lens = batch
        loss, preds = self.calc_loss(inputs, targets, hiddens, lens=lens)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        lr = self.scheduler.get_last_lr()[0]
        self.log('lr', lr, on_epoch=False, on_step=True, prog_bar=False, logger=True)
        return loss
        #return {"loss": loss}#, "hiddens": self.hiddens}
    
    def validation_step(self, batch, batch_idx):
        inputs, targets, lens = batch
        loss, preds = self.calc_loss(inputs, targets, lens=lens)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return copy.deepcopy(preds.detach().cpu()), copy.deepcopy(targets.detach().cpu())
    
    def validation_epoch_end(self, val_step_output_list):
        # calculate f1_score, accuracy etc
        preds = torch.cat([step_out[0].flatten() for step_out in val_step_output_list]).cpu().squeeze().float().numpy()
        targets = torch.cat([step_out[1].flatten() for step_out in val_step_output_list]).cpu().squeeze().float().numpy()
        # remove NaNs
        mask = ~np.isnan(targets)
        preds = preds[mask]
        targets = targets[mask]
        
        if self.regression:
            try:
                r2 = sklearn.metrics.r2_score(targets, preds)
                mse = sklearn.metrics.mean_squared_error(targets, preds)
                rmse = np.sqrt(mse)
                mae = sklearn.metrics.mean_absolute_error(targets, preds)
            except ValueError:
                print("ValueError: NaNs or Infs in targets or predictions")
                print(targets.shape, preds.shape)
                print("Inf target preds:", np.isinf(targets).sum(), np.isinf(preds).sum())
                print("Nan target preds: ", np.isnan(targets).sum(), np.isnan(preds).sum())
                print(targets, preds)
                # to stop training, raise a KeyboardInterrupt
                raise KeyboardInterrupt
            self.log("val_r2", r2, on_epoch=True, prog_bar=True)
            self.log("val_mse", mse, on_epoch=True, prog_bar=True)
            self.log("val_rmse", rmse, on_epoch=True, prog_bar=True)
            self.log("val_mae", mae, on_epoch=True, prog_bar=True)
            
        else:
            #print(targets.min(), targets.max())
            
            targets = targets.astype(int)
            #print(preds.min(), preds.max())
            #print(targets.min(), targets.max())
            # average precision
            ap = sklearn.metrics.average_precision_score(targets, preds, average="macro", pos_label=1)
            self.log('val_ap', ap, on_epoch=True, prog_bar=False)
            # auc - per class and macro-averaged
            #print("Shapes: ", targets.shape, preds.shape)
            auc_micro = sklearn.metrics.roc_auc_score(targets, preds, average="micro")
            #auc_macro = sklearn.metrics.roc_auc_score(targets, preds, average="macro")
            #self.log('val_auc_macro', auc_macro, on_epoch=True, prog_bar=True)
            self.log('val_auc_micro', auc_micro, on_epoch=True, prog_bar=True)

            # metrics based on binary predictions
            binary_preds = (preds > 0.5).astype(int)
            self.log("val_acc_micro", sklearn.metrics.accuracy_score(targets.reshape(-1), binary_preds.reshape(-1)), on_epoch=True)
            #macro_acc = (targets == binary_preds).astype(float).mean(axis=0).mean()
            #self.log("val_acc_macro", macro_acc, on_epoch=True)
            #self.log("val_f1_macro", sklearn.metrics.f1_score(targets, binary_preds, average="macro"), on_epoch=True, logger=True)
            self.log("val_f1_micro", sklearn.metrics.f1_score(targets, binary_preds, average="micro"), on_epoch=True, logger=True)

            # log diagnostics of output distribution
            preds_for_pos = preds[targets == 1]
            preds_for_neg = preds[targets == 0]
            pos_mean = preds_for_pos.mean()
            neg_mean = preds_for_neg.mean()
            self.log("debug/pos_preds_mean", pos_mean, on_epoch=True)
            self.log("debug/pos_preds_std", preds_for_pos.std(), on_epoch=True)
            self.log("debug/neg_preds_mean", neg_mean, on_epoch=True)
            self.log("debug/neg_preds_std", preds_for_neg.std(), on_epoch=True)
            self.log("debug/preds_mean", preds.mean(), on_epoch=True)
            self.log("debug/preds_std", preds.std(), on_epoch=True)
            self.log("debug/preds_mean_diff", pos_mean - neg_mean, on_epoch=True)
    

    def configure_optimizers_old(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.1)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.98)
        return [optimizer]#, [scheduler]
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                              lr=self.lr,#5e-5,
                              betas=(0.9, 0.98),
                              eps=1e-6,
                              weight_decay=self.weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, self.lr, steps_per_epoch=self.steps_per_epoch, epochs=self.max_epochs, pct_start=0.05)
        scheduler = {"scheduler": self.scheduler, 
                     "interval": "step" }  # necessary to make PL call the scheduler.step after every batch
        return [optimizer], [scheduler]
    

    def calc_loss(self, inputs, targets, hiddens=None, lens=None):
        # pred
        if hiddens is None:
            preds = self(inputs, lens=lens)
        else:
            preds = self(inputs, hiddens=hiddens, lens=lens)
        # calc loss
        loss = self.loss_func(preds, targets)
        return loss, preds
    
    
    ### The two following functions fix the  'UserWarning: Detected call of `lr_scheduler.step()` before `optimizer.step()`'
    # will be patched in later versions of PyTorch-lightning
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, **kwargs):
        self.should_skip_lr_scheduler_step = False
        scaler = getattr(self.trainer.strategy.precision_plugin, "scaler", None)
        if scaler:
            scale_before_step = scaler.get_scale()
        optimizer.step(closure=optimizer_closure)
        if scaler:
            scale_after_step = scaler.get_scale()
            self.should_skip_lr_scheduler_step = scale_before_step > scale_after_step

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        if self.should_skip_lr_scheduler_step:
            return
        scheduler.step()

    
class LitCLIP(LitSeqModel):
    def __init__(self, *args, 
                 clip_name="ViT-B/16",
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.clip_name = clip_name
        
        import clip
        clip_model, transform = clip.load(clip_name, device="cpu", jit=False)
        
        # get relevant clip layers
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        # freeze attention layers
        for n, p in self.transformer.named_parameters():
            p.requires_grad = "mlp" in n or "ln" in n
        
        # define own mapping layers
        self.input_mapping = torch.nn.Linear(self.num_recurrent_inputs + self.num_static_inputs, self.token_embedding.weight.shape[1])
        self.out_mapping = torch.nn.Linear(self.transformer.width, 1)
        
    def make_preds(self, x, lens=None):
        # x = [BS, seq_len, feats]
        
        #x = self.token_embedding(text)#.type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = self.input_mapping(x)
        
        x = x + self.positional_embedding#.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) #.type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence) if we only want a single embedding for the sentence
        #x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        #x = x[torch.arange(x.shape[0]), lens] @ self.text_projection
        
        # make prediction for each time-step
        x = self.out_mapping(x)
        return x
        
        
def load_gpt_model(name, use_adapter=False, reduction_factor=2):
    device = torch.device("cpu")
    if name == "neo1.3":
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B", pad_token_id=gpt_tokenizer.eos_token_id)
    elif name == "neo2.7":
        from transformers import GPTNeoForCausalLM, GPT2Tokenizer
        gpt_tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")
        gpt_model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B", pad_token_id=gpt_tokenizer.eos_token_id)
    elif "gpt2" in name:
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        gpt_tokenizer = GPT2Tokenizer.from_pretrained(name)
        if use_adapter:
            from transformers import GPT2AdapterModel
            gpt_model = GPT2AdapterModel.from_pretrained(name, pad_token_id=gpt_tokenizer.eos_token_id)
            # task adapter - only add if not existing
           
            
            import transformers
            config = transformers.PfeifferConfig({
                                                    "reduction_factor": reduction_factor,
                                                },
                                                 reduction_factor=reduction_factor)

            gpt_model.add_adapter("icp", config=config, overwrite_ok=False, set_active=True)            
            
            # Enable adapter training
            gpt_model.train_adapter("icp")
            #gpt_model.set_active_adapters("icp")
        else:
            gpt_model = GPT2LMHeadModel.from_pretrained(name, pad_token_id=gpt_tokenizer.eos_token_id)     
           
    gpt_model = gpt_model.to(device)
    return gpt_model, gpt_tokenizer


def apply_train_mode_gpt(mode, model):
    # freeze certain layers
    if mode == "freeze":
        for p in model.parameters():
            p.requires_grad = False
    elif mode == "train_norm":
        for n, p in model.named_parameters():
            p.requires_grad = "ln_" in n
    elif mode == "full":
        for p in model.parameters():
            p.requires_grad = True
    elif mode == "train_mlp_norm":
        for n, p in model.named_parameters():
            p.requires_grad = "ln_" in n or "mlp" in n
    elif mode == "adapters":
        #return  # the adapters library handles this
        for n, p in model.named_parameters():
            p.requires_grad = "adapter" in n or "ln_" in n

        
class LitGPT(LitSeqModel):
    def __init__(self, *args, 
                 gpt_name="gpt2",
                 mode="train_mlp_norm",
                 pretrained=True,
                 reduction_factor=2,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_adapter = "adapter" in mode
        self.gpt_name = gpt_name
        # load model
        self.model, self.tokenizer = load_gpt_model(gpt_name, use_adapter=self.use_adapter, reduction_factor=reduction_factor)
        # re-init weights if not using pretrained
        if not pretrained:
            self.model.init_weights()  #.apply(self.model._init_weights)            

        # freeze some params
        apply_train_mode_gpt(mode, self.model)
        # get width
        self.width = self.model.transformer.wte.weight.shape[1]
        # create input and output layers
        self.input_mapping = torch.nn.Linear(self.num_recurrent_inputs + self.num_static_inputs, self.width)
        self.out_mapping = torch.nn.Linear(self.width, 1, bias=False)
        self.out_norm = torch.nn.LayerNorm(self.width)
        # replace output layer by our newly initialized one
        #self.model.transformer.wte = self.input_mapping
        #self.model.lm_head = self.out_mapping
        #self.model.add_classification_head("icp", num_labels=1, activation_function="linear")
        
        
    def make_preds(self, x, lens=None):
        # x = [BS, seq_len, feats]
        #print(x.shape)
        x = self.input_mapping(x)
        x = self.model(inputs_embeds=x)   
        x = x["last_hidden_state"]
        x = self.out_norm(x)
        x = self.out_mapping(x)
        #print(x.shape)
        #x = self.model(x)["logits"]
        return x

    
class LitTransformer(LitSeqModel):
    def __init__(self, *args,
                 num_transformer_blocks=3,
                 hidden_size=512, 
                 n_heads=8,
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.out_size = 1
        self.hidden_size = hidden_size
        
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        
        self.pos_encoder = PositionalEncoding(hidden_size, self.dropout)
        encoder_layers = TransformerEncoderLayer(hidden_size, n_heads, 
                                                 hidden_size * 4, self.dropout, 
                                                 batch_first=True,
                                                 norm_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_transformer_blocks)
        self.encoder = torch.nn.Linear(self.num_recurrent_inputs, hidden_size)
        self.decoder = torch.nn.Linear(hidden_size, 1)
        
        
    def make_preds(self, x, lens=None):
        # 1. map to tokens
        x = self.encoder(x) * math.sqrt(self.hidden_size)
        # 2. add positional encoding
        x = self.pos_encoder(x)
        # 3. apply transformer
        src_mask = generate_square_subsequent_mask(x.shape[1]).to(x.device).to(x.dtype)
        x = self.transformer_encoder(x, src_mask)
        # 4. map to output
        x = self.decoder(x)
        return x

    
def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Is used in the causal transformer modelling such that tokens can only attend to past tokens. 
    """
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)


class PositionalEncoding(torch.nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

        
class LitMLP(LitSeqModel):
    def __init__(self, 
                 *args, 
                 hidden_size=256, 
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()    
            
        self.hidden_size = hidden_size
        self.out_size = 1
        
        # define model
        # static part
        if self.static_idcs is not None:
            self.encode_static = torch.nn.Sequential(
                torch.nn.Linear(self.num_static_inputs, self.hidden_size),
                torch.nn.ReLU(True),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                #torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
                torch.nn.ReLU(True),
            )
        # recurrent part
        self.encode_recurrent = torch.nn.Sequential(
            torch.nn.Linear(self.num_recurrent_inputs, self.hidden_size),
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.hidden_size),
            #torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
            torch.nn.ReLU(True),
        )
        # out part
        self.out_mlp = torch.nn.Sequential(
            torch.nn.Linear(self.hidden_size, self.hidden_size), 
            torch.nn.ReLU(True),
            torch.nn.Linear(self.hidden_size, self.out_size))
        
    def make_preds(self, x, lens=None, hiddens=None):
        # encode
        x = self.encode_recurrent(x)
        x = self.out_mlp(x)
        return x
        
    def make_preds_old(self, 
                x: torch.Tensor, 
                lens: Optional[torch.Tensor] = None, 
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        # encode recurrence using mlp
        recurrent_stream = x[..., self.recurrent_idcs]
        recurrent_stream = self.encode_recurrent(recurrent_stream)
        
        # encode static inputs
        if self.static_idcs is not None:
            static_stream = x[..., self.static_idcs]
            static_stream = self.encode_static(static_stream)
        else:
            static_stream = 0
            
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        return x


class LitRNN(LitSeqModel):
    def __init__(self, *args, hidden_size=256, rnn_layers=1,  rnn_type="lstm", use_lens=False,
                 use_in_mapping=False, use_out_mapping=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        
        self.hidden_size = hidden_size
        self.out_size = 1
        self.use_lens = use_lens
        self.use_in_mapping = use_in_mapping
        self.use_out_mapping = use_out_mapping
        
        # define model
        # recurrent part
        rnn_args = [self.hidden_size if use_in_mapping else self.num_recurrent_inputs,
                    self.hidden_size if use_out_mapping else self.out_size,
                   ]
        rnn_kwargs = {"num_layers": rnn_layers,
                     "batch_first": True,
                     "dropout": self.dropout if rnn_layers > 1 else 0}
        if rnn_type == "lstm":
            self.rnn = torch.nn.LSTM(*rnn_args, **rnn_kwargs)
        elif rnn_type == "gru":
            self.rnn = torch.nn.GRU(*rnn_args, **rnn_kwargs)
        #self.layer_norm = torch.nn.LayerNorm(normalized_shape=[self.hidden_size])
        
        # static part
        if self.static_idcs is not None:
            self.encode_static = torch.nn.Sequential(
                torch.nn.Linear(self.num_static_inputs, self.hidden_size),
                torch.nn.ReLU(True),
                torch.nn.Linear(self.hidden_size, self.hidden_size),
                torch.nn.LayerNorm(normalized_shape=[self.hidden_size]),
                torch.nn.ReLU(True),
            )
        # out part
        if use_out_mapping:
            self.out_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.hidden_size, self.hidden_size), 
                torch.nn.ReLU(True),
                torch.nn.Linear(self.hidden_size, self.out_size))
        if use_in_mapping:
            self.in_mlp = torch.nn.Sequential(
                torch.nn.Linear(self.num_recurrent_inputs, self.hidden_size), 
                torch.nn.ReLU(True),
                torch.nn.Linear(self.hidden_size, self.hidden_size))
        self.hiddens = (torch.zeros(0), torch.zeros(0))  # init for TorchScript
        
    def make_preds(self, x, lens=None):
        if self.use_in_mapping:
            x = self.in_mlp(x)
        # apply rnn
        if lens is not None and self.use_lens:
            x = pack_padded_sequence(x, lens.cpu(), batch_first=True, enforce_sorted=False)
        x, self.hiddens = self.rnn(x)
        if lens is not None and self.use_lens:
            x, lens = pad_packed_sequence(x, batch_first=True)
        # apply mlp to transform    
        if self.use_out_mapping:
            x = self.out_mlp(x)
        return x
    
    
    def make_preds_old(self, 
                x: torch.Tensor, 
                lens: Optional[torch.Tensor] = None, 
                hiddens: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        
        # encode recurrence using rnn
        recurrent_stream = x[..., self.recurrent_idcs]
        if lens is not None:
            recurrent_stream = pack_padded_sequence(recurrent_stream, lens.cpu(), batch_first=True, enforce_sorted=False)
        recurrent_stream, self.hiddens = self.rnn(recurrent_stream)
        if lens is not None:
            recurrent_stream, lens = pad_packed_sequence(recurrent_stream, batch_first=True)
        #recurrent_stream = self.layer_norm(recurrent_stream)
        
        # encode static inputs
        if self.static_idcs is not None:
            static_stream = x[..., self.static_idcs]
            static_stream = self.encode_static(static_stream)
        else:
            static_stream = 0
            
        # apply mlp to transform        
        x = self.out_mlp(static_stream + recurrent_stream)
        return x


class SequentialLoss(torch.nn.Module):#jit.ScriptModule):
    def __init__(self, regression, use_macro_loss, pos_weight, use_huber):
        super().__init__()
        self.regression = regression
        self.use_macro_loss = use_macro_loss
        self.pos_weight = pos_weight
        
        if regression:
            if use_huber:
                self.loss_func = torch.nn.SmoothL1Loss(reduction='mean')
            else:
                self.loss_func = torch.nn.MSELoss(reduction='mean')
        else:
            self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    def __call__(self, pred: torch.Tensor, target: torch.Tensor):
        """Calculates the loss and considers missing values in the loss given by mask indicating where targets are NaN"""
        # shape: [BS, LEN, FEATS]
        mask = ~torch.isnan(target)
        
        if self.use_macro_loss:
            # Apply mask:
            num_feats = target.shape[-1]
            pred_per_pat = [pred_p[mask_p].reshape(-1, num_feats) for pred_p, mask_p in zip(pred, mask) if sum(mask_p) > 0]
            target_per_pat = [target_p[mask_p].reshape(-1, num_feats) for target_p, mask_p in zip(target, mask) if sum(mask_p) > 0]
            # calc raw loss per patient
            loss_per_pat = [self.loss_func(p, t) for p, t in zip(pred_per_pat, target_per_pat)]
            loss = torch.stack(loss_per_pat).mean()

            #if torch.isinf(loss) or torch.isnan(loss):
            #    print("Found NAN or inf loss!")
            #    print(loss)
            #    raise ValueError("Found NAN or inf!")
        else:
            #print(pred.shape, target.shape, mask.shape)
            loss = self.loss_func(pred[mask], target[mask])

        return loss
