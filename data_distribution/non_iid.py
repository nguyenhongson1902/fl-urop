import numpy as np

from torch.utils.data import Subset
import torch

np.random.seed(0) # Don't change the line

def generate_non_iid_data(train_dataset, test_dataset, dataset, n_workers):
    """
        train_dataloader
        test_dataloader
    """
    # get train labels
    y_train = np.array(train_dataset.targets)

    number_of_classes = len(np.unique(y_train))
    n_train = len(train_dataset)
    n_nets = n_workers #total clients
    
    partition_alpha = 0.5
    min_size = 0
    min_required_size = 40 # #min_samples/client
    K = number_of_classes # number of classes
    net_dataidx_map = {}

    while (min_size < min_required_size) or (dataset == 'mnist' and min_size < 100):
        idx_batch = [[] for _ in range(n_nets)]
        # for each class in the dataset
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(partition_alpha, n_nets))
            ## Balance
            proportions = np.array([p*(len(idx_j) < n_train/n_nets) for p,idx_j in zip(proportions,idx_batch)])
            proportions = proportions/proportions.sum()
            proportions = (np.cumsum(proportions)*len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j,idx in zip(idx_batch,np.split(idx_k,proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_nets):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    
    # count how many samples in each client
    total_sample = 0
    for j in range(n_nets):
        print("Client %d: %d samples" % (j, len(net_dataidx_map[j])))
        cnt_class = {}
        for i in net_dataidx_map[j]:
            label = y_train[i]
            if label not in cnt_class:
                cnt_class[label] = 0
            cnt_class[label] += 1
        total_sample += len(net_dataidx_map[j])
        print("Client %d: %s" % (j, str(cnt_class)))
        print("--------"*10)
    print("Total training: %d samples" % total_sample)
    print("Total testing: %d samples" % len(test_dataset))
    # train_loaders = [
    #     torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_size=args.batch_size,
    #         # sampler=SubsetRandomSampler(indices), # For random sampling
    #         sampler=SequentialSampler(indices),
    #     )
    #     for _, indices in net_dataidx_map.items()
    # ]
    subsets = [Subset(train_dataset, indices) for _, indices in net_dataidx_map.items()]
    train_loaders = []
    for subset in subsets:
        train_loaders.append(torch.utils.data.DataLoader(subset, batch_size=64)) 
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    return train_loaders, test_loader, net_dataidx_map