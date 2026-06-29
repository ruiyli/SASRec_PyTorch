import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def random_neq(l, r, s):
    """Sample negative item that is not in set s"""
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


class SASRecDataset(Dataset):
    """Dataset for SASRec model"""
    def __init__(self, user_train, usernum, itemnum, maxlen):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.maxlen = maxlen
        
        # Create list of users with sufficient sequence length
        self.users = []
        for u in range(1, usernum + 1):
            if len(user_train[u]) > 1:  # Need at least 2 items for training
                self.users.append(u)
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        u = self.users[idx]
        seq = self.user_train[u]
        
        # Sample sequence
        if len(seq) > self.maxlen:
            seq = seq[-self.maxlen:]
        
        # Create input sequence
        seq_len = len(seq)
        input_seq = np.zeros([self.maxlen], dtype=np.int32)
        pos_seq = np.zeros([self.maxlen], dtype=np.int32)
        neg_seq = np.zeros([self.maxlen], dtype=np.int32)
        
        nxt = seq[-1]
        idx = self.maxlen - 1
        
        ts = set(seq)
        for i in reversed(seq[:-1]):
            input_seq[idx] = i
            pos_seq[idx] = nxt
            neg_seq[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        return (u, input_seq, pos_seq, neg_seq)


class WarpSampler:
    """Multi-process sampler for SASRec (simplified single-process version)"""
    def __init__(self, user_train, usernum, itemnum, batch_size=64, maxlen=10):
        self.user_train = user_train
        self.usernum = usernum
        self.itemnum = itemnum
        self.batch_size = batch_size
        self.maxlen = maxlen
        
        # Create list of users with sufficient sequence length
        self.users = []
        for u in range(1, usernum + 1):
            if len(user_train[u]) > 1:
                self.users.append(u)
        
        self.counter = 0
        np.random.shuffle(self.users)
    
    def sample(self, u):
        """Sample training sequence for a user"""
        seq = self.user_train[u]
        
        # Sample sequence
        if len(seq) > self.maxlen:
            seq = seq[-self.maxlen:]
        
        # Create input sequence
        seq_len = len(seq)
        input_seq = np.zeros([self.maxlen], dtype=np.int32)
        pos_seq = np.zeros([self.maxlen], dtype=np.int32)
        neg_seq = np.zeros([self.maxlen], dtype=np.int32)
        
        nxt = seq[-1]
        idx = self.maxlen - 1
        
        ts = set(seq)
        for i in reversed(seq[:-1]):
            input_seq[idx] = i
            pos_seq[idx] = nxt
            neg_seq[idx] = random_neq(1, self.itemnum + 1, ts)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        
        return (u, input_seq, pos_seq, neg_seq)
    
    def next_batch(self):
        """Get next batch of training samples"""
        if self.counter % len(self.users) == 0:
            np.random.shuffle(self.users)
            self.counter = 0
        
        one_batch = []
        for i in range(self.batch_size):
            u = self.users[self.counter % len(self.users)]
            one_batch.append(self.sample(u))
            self.counter += 1
        
        return zip(*one_batch)
    
    def close(self):
        pass


def data_partition(fname):
    """Partition data into train/valid/test sets"""
    usernum = 0
    itemnum = 0
    User = {}
    
    # Read data file
    with open(fname, 'r') as f:
        for line in f:
            u, i = line.rstrip().split(' ')
            u = int(u)
            i = int(i)
            usernum = max(u, usernum)
            itemnum = max(i, itemnum)
            
            if u not in User:
                User[u] = []
            User[u].append(i)
    
    # Split data
    user_train = {}
    user_valid = {}
    user_test = {}
    
    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    
    return [user_train, user_valid, user_test, usernum, itemnum]


if __name__ == "__main__":
    # Test data partition
    # Create test data file
    with open('/tmp/test_data.txt', 'w') as f:
        for u in range(1, 101):
            for i in range(1, 11):
                f.write(f"{u} {i}\n")
    
    dataset = data_partition('/tmp/test_data.txt')
    print(f"Dataset: {dataset[3]} users, {dataset[4]} items")
    
    # Test sampler
    sampler = WarpSampler(dataset[0], dataset[3], dataset[4], batch_size=32, maxlen=50)
    u, seq, pos, neg = sampler.next_batch()
    print(f"Batch shapes: u={len(u)}, seq={len(seq)}, pos={len(pos)}, neg={len(neg)}")
    print("Data iterator test passed!")