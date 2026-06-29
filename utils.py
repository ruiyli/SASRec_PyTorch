import numpy as np
import torch
import random
import copy
from collections import defaultdict


def evaluate(model, dataset, args):
    """Evaluate model on test set"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(test[u]) < 1:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        # Use validation item if available, otherwise use last training item
        if len(valid[u]) > 0:
            seq[idx] = valid[u][0]
            idx -= 1
        
        # Fill sequence with training items
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        rated = set(train[u])
        if len(valid[u]) > 0:
            rated.add(valid[u][0])
        rated.add(0)
        
        item_idx = [test[u][0]]
        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        # Get predictions
        with torch.no_grad():
            predictions = -model.predict(
                torch.LongTensor([u]).to(args.device),
                torch.LongTensor([seq]).to(args.device),
                torch.LongTensor([item_idx]).to(args.device)
            )
            predictions = predictions[0].cpu().numpy()
        
        rank = predictions.argsort().argsort()[0]
        
        valid_user += 1
        
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        
        if valid_user % 100 == 0:
            print('.', end="")
            
    return NDCG / valid_user, HT / valid_user


def evaluate_valid(model, dataset, args):
    """Evaluate model on validation set"""
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)
    
    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0
    
    if usernum > 10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1:
            continue
        
        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        
        # Fill sequence with training items
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1:
                break
        
        rated = set(train[u])
        rated.add(0)
        
        item_idx = [valid[u][0]]
        for _ in range(99):
            t = np.random.randint(1, itemnum + 1)
            while t in rated:
                t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        
        # Get predictions
        with torch.no_grad():
            predictions = -model.predict(
                torch.LongTensor([u]).to(args.device),
                torch.LongTensor([seq]).to(args.device),
                torch.LongTensor([item_idx]).to(args.device)
            )
            predictions = predictions[0].cpu().numpy()
        
        rank = predictions.argsort().argsort()[0]
        
        valid_user += 1
        
        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        
        if valid_user % 100 == 0:
            print('.', end="")
            
    return NDCG / valid_user, HT / valid_user


def calc_auc(raw_arr):
    """Calculate AUC for binary classification"""
    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1
    
    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append((fp / neg, tp / pos))
    
    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += (x - prev_x) * (y + prev_y) / 2.
            prev_x = x
            prev_y = y
    
    return auc


if __name__ == "__main__":
    # Test AUC calculation
    test_data = [[0.9, 1], [0.8, 1], [0.7, 0], [0.6, 1], [0.5, 0]]
    auc = calc_auc(test_data)
    print(f"Test AUC: {auc:.4f}")
    print("Utils test passed!")