import os
import sys
import time
import random
import numpy as np
import argparse
import torch
import torch.nn as nn

from data_loader import WarpSampler, data_partition
from model import SASRec
from utils import evaluate, evaluate_valid


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def train(
        dataset='ml-1m',
        batch_size=128,
        maxlen=200,
        hidden_units=50,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.2,
        lr=0.001,
        epochs=200,
        l2_emb=0.0,
        seed=42,
        model_type='SASRec'):
    
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seal_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Create output directories
    out_dir1 = "output"
    out_dir2 = "best_model"
    os.makedirs(out_dir1, exist_ok=True)
    os.makedirs(out_dir2, exist_ok=True)
    
    model_path = f"{out_dir1}/cpkt_{model_type}_{dataset}_{str(seed)}"
    best_model_path = f"{out_dir2}/cpkt_{model_type}_{dataset}_{str(seed)}"
    
    # Parse arguments
    class Args:
        def __init__(self):
            self.hidden_units = hidden_units
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.dropout_rate = dropout_rate
            self.maxlen = maxlen
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.l2_emb = l2_emb
            self.batch_size = batch_size
    
    args = Args()
    
    # Load data
    data_file = f"{dataset}.txt"
    dataset = data_partition(data_file)
    [user_train, user_valid, user_test, usernum, itemnum] = dataset
    
    print(f"Dataset: {usernum} users, {itemnum} items")
    
    # Create sampler
    sampler = WarpSampler(user_train, usernum, itemnum, 
                         batch_size=batch_size, maxlen=maxlen)
    
    # Create model
    model = SASRec(itemnum, args)
    model.to(args.device)
    
    # Initialize weights
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass
    
    # Zero out padding embeddings
    model.item_emb.weight.data[0, :] = 0
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    # Create optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98))
    criterion = nn.BCEWithLogitsLoss()
    
    best_ndcg = 0.0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        num_batches = len([u for u in user_train if len(user_train[u]) > 1]) // batch_size + 1
        
        for step in range(num_batches):
            u, seq, pos, neg = sampler.next_batch()
            u, seq, pos, neg = np.array(u), np.array(seq), np.array(pos), np.array(neg)
            
            # Move to device
            u = torch.LongTensor(u).to(args.device)
            seq = torch.LongTensor(seq).to(args.device)
            pos = torch.LongTensor(pos).to(args.device)
            neg = torch.LongTensor(neg).to(args.device)
            
            optimizer.zero_grad()
            
            pos_logits, neg_logits = model(u, seq, pos, neg)
            
            # Create labels
            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)
            
            # Only compute loss on non-padded positions
            indices = (pos != 0)
            
            loss = criterion(pos_logits[indices], pos_labels[indices])
            loss += criterion(neg_logits[indices], neg_labels[indices])
            
            # L2 regularization on item embeddings
            for param in model.item_emb.parameters():
                loss += l2_emb * torch.sum(param ** 2)
            
            loss.backward()
            optimizer.step()
            
            loss_sum += loss.item()
            
            if step % 100 == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")
        
        avg_loss = loss_sum / num_batches
        
        # Evaluate after each epoch
        model.eval()
        test_ndcg, test_hr = evaluate(model, dataset, args)
        
        print(f"Epoch {epoch}: Loss={avg_loss:.4f}, Test NDCG@10={test_ndcg:.4f}, Test HR@10={test_hr:.4f}")
        
        # Save best model
        if test_ndcg > best_ndcg:
            best_ndcg = test_ndcg
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with NDCG@10: {best_ndcg:.4f}")
        
        model.train()
    
    sampler.close()
    print("Training completed!")


def test(
        dataset='ml-1m',
        maxlen=200,
        hidden_units=50,
        num_blocks=2,
        num_heads=1,
        dropout_rate=0.2,
        seed=42,
        model_type='SASRec',
        model_path=""):
    
    if model_path == "":
        model_path = f"best_model/cpkt_{model_type}_{dataset}_{str(seed)}"
    
    # Parse arguments
    class Args:
        def __init__(self):
            self.hidden_units = hidden_units
            self.num_blocks = num_blocks
            self.num_heads = num_heads
            self.dropout_rate = dropout_rate
            self.maxlen = maxlen
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = Args()
    
    # Load data
    data_file = f"{dataset}.txt"
    dataset = data_partition(data_file)
    [_, _, _, usernum, itemnum] = dataset
    
    # Create model
    model = SASRec(itemnum, args)
    model.to(args.device)
    
    # Load trained weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=args.device))
    except:
        print(f"Failed to load model from {model_path}")
        return
    
    model.eval()
    
    # Evaluate
    test_ndcg, test_hr = evaluate(model, dataset, args)
    print(f"Test Results - NDCG@10: {test_ndcg:.4f}, HR@10: {test_hr:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SASRec Trainer")
    parser.add_argument('--mode', type=str, default='train', help='train or test')
    parser.add_argument('--dataset', type=str, default='ml-1m', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--maxlen', type=int, default=200, help='maximum sequence length')
    parser.add_argument('--hidden_units', type=int, default=50, help='hidden units')
    parser.add_argument('--num_blocks', type=int, default=2, help='number of blocks')
    parser.add_argument('--num_heads', type=int, default=1, help='number of heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='dropout rate')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='l2 regularization')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--model', type=str, default='SASRec', help='model type')
    parser.add_argument('--model_path', type=str, default='', help='path to model checkpoint for testing')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(
            dataset=args.dataset,
            batch_size=args.batch_size,
            maxlen=args.maxlen,
            hidden_units=args.hidden_units,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            lr=args.lr,
            epochs=args.epochs,
            l2_emb=args.l2_emb,
            seed=args.seed,
            model_type=args.model
        )
    elif args.mode == 'test':
        test(
            dataset=args.dataset,
            maxlen=args.maxlen,
            hidden_units=args.hidden_units,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            dropout_rate=args.dropout_rate,
            seed=args.seed,
            model_type=args.model,
            model_path=args.model_path
        )
    else:
        print("Invalid mode. Use 'train' or 'test'.")