import torch
import torch.nn as nn


class EmbeddingLayer(nn.Module):
    def __init__(self, num_emb, embedding_dim, padding_idx=0):
        super(EmbeddingLayer, self).__init__()

        self.embeddings = nn.Embedding(num_emb, embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, batch_cat):
        batch_embedding = self.embeddings(batch_cat)
        return batch_embedding
    

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embedding_dim, padding_idx=0):
        super(PositionalEmbedding, self).__init__()
        
        self.embeddings = nn.Embedding(max_len + 1, embedding_dim, padding_idx=padding_idx)
        nn.init.xavier_uniform_(self.embeddings.weight)

    def forward(self, positions):
        pos_embedding = self.embeddings(positions)
        return pos_embedding


if __name__ == "__main__":
    # Test item embedding
    item_emb = EmbeddingLayer(1000, 12)
    items = torch.ones((2048, ), dtype=torch.long)
    print(f"Item embedding shape: {item_emb(items).shape}")  # torch.Size([2048, 12])
    
    # Test positional embedding
    pos_emb = PositionalEmbedding(200, 12)
    positions = torch.arange(100, dtype=torch.long)
    print(f"Positional embedding shape: {pos_emb(positions).shape}")  # torch.Size([100, 12])