import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_units,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True  # 改为batch_first=True以匹配SASRec的输入格式
        )
        
    def forward(self, query, key, value, attn_mask=None):
        # query, key, value shape: (batch_size, seq_len, hidden_units)
        # attn_mask shape: (batch_size * num_heads, seq_len, seq_len) or (seq_len, seq_len)
        
        attn_output, attn_weights = self.attention(
            query, key, value, 
            attn_mask=attn_mask,
            need_weights=False
        )
        return attn_output


class SASRecAttentionLayer(nn.Module):
    def __init__(self, hidden_units, num_heads, dropout_rate):
        super(SASRecAttentionLayer, self).__init__()
        
        self.multihead_attention = MultiHeadAttention(hidden_units, num_heads, dropout_rate)
        
    def forward(self, seqs, attention_mask):
        # seqs shape: (batch_size, seq_len, hidden_units)
        # attention_mask shape: (seq_len, seq_len)
        
        # Apply multihead attention directly (batch_first=True)
        attn_output = self.multihead_attention(
            query=seqs,
            key=seqs,
            value=seqs,
            attn_mask=attention_mask
        )
        
        return attn_output


if __name__ == "__main__":
    # Test MultiHeadAttention
    batch_size, seq_len, hidden_units = 32, 100, 50
    num_heads = 5
    
    mha = MultiHeadAttention(hidden_units, num_heads, 0.2)
    
    # Create causal mask
    attention_mask = ~torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool))
    
    # Test input (seq_len, batch_size, hidden_units)
    query = torch.randn(seq_len, batch_size, hidden_units)
    key = torch.randn(seq_len, batch_size, hidden_units)
    value = torch.randn(seq_len, batch_size, hidden_units)
    
    output = mha(query, key, value, attention_mask)
    print(f"MultiHeadAttention input shape: {query.shape}")
    print(f"MultiHeadAttention output shape: {output.shape}")
    
    # Test SASRecAttentionLayer
    sasrec_attn = SASRecAttentionLayer(hidden_units, num_heads, 0.2)
    seqs = torch.randn(batch_size, seq_len, hidden_units)
    output = sasrec_attn(seqs, attention_mask)
    print(f"SASRecAttentionLayer input shape: {seqs.shape}")
    print(f"SASRecAttentionLayer output shape: {output.shape}")