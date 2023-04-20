import torch
import numpy as np


class ContinuousEmbedding(torch.nn.Module):
    def __init__(self, in_size, out_size, init_std=0.1):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        
        self.weights = torch.nn.Parameter(torch.zeros(1, in_size, out_size).normal_(0, init_std))
        self.bias = torch.nn.Parameter(torch.rand(1, in_size, 1))

    def forward(self, x):
        """
        Assumes input shape [batch size, seq_len, in_size]
        """
        bsz = x.shape[0]
        steps = x.shape[1]
        x = x.reshape(-1, self.in_size).unsqueeze(-1).repeat(1, 1, self.out_size)
        #print(x.shape, self.weights.shape)
        out = x * self.weights + self.bias
        out = out.reshape(bsz, steps, self.in_size, self.out_size)
        return out


class NanEmbed(torch.nn.Module):
    def __init__(self, in_size, out_size, low_mem_mode=False):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.low_mem_mode = low_mem_mode
        # create embedding weights
        self.cont_emb = ContinuousEmbedding(in_size, out_size)
        
    def forward(self, x):
        """
        Assumes input shape [batch size, seq_len, in_size]
        """
        if self.low_mem_mode:
            embs = []
            for seq in x:
                seq_emb = self.apply_cont_emb(seq.unsqueeze(0))
                embs.append(seq_emb)
            emb = torch.cat(embs)
        else:
            emb = self.apply_cont_emb(x)
        return emb
    
    def apply_cont_emb(self, x):
        # create mask to later fill with zeros
        mask = torch.isnan(x)
        x = torch.nan_to_num(x)
        # embed each feature into a larger embedding vector of size out_size
        out = self.cont_emb(x)
        # shape [batch size, seq_len, in_size, out_size]
        # fill embedding with 0 where we had a NaN before
        with torch.no_grad():
            out[mask] = 0
        # average the embedding
        emb = out.mean(dim=2) 
        return emb
    
    
class NanEmbedTransformer(torch.nn.Module):
    def __init__(self, in_size, out_size, low_mem_mode=False,
                n_layers=3, n_heads=4, dropout=0.2):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.low_mem_mode = low_mem_mode
        # create embedding weights
        self.cont_emb = ContinuousEmbedding(in_size, out_size)
        # create transformer that operates on these
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        encoder_layers = TransformerEncoderLayer(out_size, n_heads, out_size, 
                                                 dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        
        scale = out_size ** -0.5
        self.positional_embedding = torch.nn.Parameter(scale * torch.randn(in_size, out_size))
        self.ln_pre = torch.nn.LayerNorm(out_size)
        
    def forward(self, x):
        if self.low_mem_mode:
            embs = []
            for seq in x:
                seq_emb = self.apply_nan_transformer(seq.unsqueeze(0))
                embs.append(seq_emb)
            emb = torch.cat(embs)
        else:
            emb = self.apply_nan_transformer(x)
        return emb
            
    def apply_nan_transformer(self, x):
        # create mask to later fill with zeros
        mask = torch.isnan(x)
        x = torch.nan_to_num(x)
        # embed each feature into a larger embedding vector of size out_size
        out = self.cont_emb(x)
        # shape [batch size, seq_len, in_size, out_size]
        # fill embedding with 0 where we had a NaN before
        with torch.no_grad():
            out[mask] = 0
        # apply transformer to "tokens"
        # transform to shape  [batch size * seq_len, in_size, out_size]
        orig_shape = out.shape
        out = out.reshape(out.shape[0] * out.shape[1], out.shape[2], out.shape[3])
        # first add positional encoding
        out = out + self.positional_embedding.to(out.dtype)
        # transform
        out = self.ln_pre(out)
        #dtype = self.transformer_encoder.layers[0].self_attn.out_proj.weight.dtype
        #print(dtype, out.dtype)
        #out = out.to(dtype)
        mask = torch.zeros(out.shape[1], out.shape[1], dtype=out.dtype, device=out.device)
        #print("Transformer input shapes: ", out.shape, mask.shape)
        out = self.transformer_encoder(out, mask)
        # put back into original sequence shape
        out = out.reshape(*orig_shape)
        # average the embedding over all tokens
        emb = out.mean(dim=2) 
        return emb
    
    
if __name__ == "__main__":
    # create test data
    x = torch.arange(24).reshape(2, 3, 4)
    emb_layer = NanEmbed(4, 8)

    # apply layer 
    emb = emb_layer(x)
    emb.shape, emb
    
    # MANUAL CALC
    weights = emb_layer.cont_emb.weights.squeeze().detach()
    bias = emb_layer.cont_emb.bias.squeeze().detach()
    print(weights, bias)
    print()
    manuel_emb = []
    for batch in x:
        step_emb = []
        for step in batch:
            feat_emb = []
            for idx, feat in enumerate(step):
                #print(feat, bias[idx], weights[idx])
                #print(weights[idx] * feat + bias[idx])
                #print()
                result = weights[idx] * feat + bias[idx]
                feat_emb.append(result)
            step_emb.append(torch.stack(feat_emb))
        manuel_emb.append(torch.stack(step_emb))
    manuel_emb = torch.stack(manuel_emb)
    #print(manuel_emb.shape)
    mean_manuel_emb = manuel_emb.mean(dim=2)
    
    assert torch.allclose(emb, mean_manuel_emb)
    print("Test passed, NaNEmbed embeds correctly")
    
    