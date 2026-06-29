import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from embedding import EmbeddingLayer, PositionalEmbedding
from feed_forward import PointWiseFeedForward
from attention import MultiHeadAttention


class SASRec(nn.Module):
    def __init__(self, item_num, args):
        super(SASRec, self).__init__()
        self.item_num = item_num
        self.dev = args.device
        
        # Item embedding layer
        self.item_emb = nn.Embedding(item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # Attention layers
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = MultiHeadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)
    
    def log2feats(self, log_seqs):
        """Convert input sequences to features"""
        seqs = self.item_emb(log_seqs)
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        
        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim
        
        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(len(self.attention_layers)):
            # Layer normalization
            Q = self.attention_layernorms[i](seqs)
            
            # Multi-head attention
            mha_outputs = self.attention_layers[i](Q, seqs, seqs, attention_mask)
            
            # Residual connection
            seqs = Q + mha_outputs
            
            # Layer normalization
            seqs = self.forward_layernorms[i](seqs)
            
            # Feed forward
            ff_output = self.forward_layers[i](seqs)
            
            # Residual connection
            seqs = seqs + ff_output
        
        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)
        
        return log_feats
    
    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        """Forward pass for training"""
        log_feats = self.log2feats(log_seqs)  # (batch_size, seq_len, hidden_units)
        
        pos_embs = self.item_emb(pos_seqs)
        neg_embs = self.item_emb(neg_seqs)
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits
    
    def predict(self, user_ids, log_seqs, item_indices):
        """Predict scores for candidate items"""
        log_feats = self.log2feats(log_seqs)  # (batch_size, seq_len, hidden_units)
        
        # Use the last position for prediction
        final_feat = log_feats[:, -1, :]  # (batch_size, hidden_units)
        
        item_embs = self.item_emb(item_indices)  # (batch_size, num_candidates, hidden_units)
        
        # Calculate scores
        scores = (final_feat.unsqueeze(1) * item_embs).sum(dim=-1)  # (batch_size, num_candidates)
        
        return scores


if __name__ == "__main__":
    # Test the model
    class Args:
        def __init__(self):
            self.hidden_units = 50
            self.num_blocks = 2
            self.num_heads = 1
            self.dropout_rate = 0.2
            self.maxlen = 100
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = Args()
    model = SASRec(1000, args)
    
    # Test forward pass
    batch_size, seq_len = 32, 50
    user_ids = torch.ones(batch_size, dtype=torch.long)
    log_seqs = torch.randint(1, 1000, (batch_size, seq_len))
    pos_seqs = torch.randint(1, 1000, (batch_size, seq_len))
    neg_seqs = torch.randint(1, 1000, (batch_size, seq_len))
    
    pos_logits, neg_logits = model(user_ids, log_seqs, pos_seqs, neg_seqs)
    print(f"Model output shapes: pos_logits {pos_logits.shape}, neg_logits {neg_logits.shape}")
    print("Model test passed!")