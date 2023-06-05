import torch
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import argparse

from nets.cifar10_cnn import Cifar10CNN

from utils import get_dataset
from federated_learning.fed_avg import average_nn_parameters
from data_distribution.non_iid import generate_non_iid_data
from client import Client


def create_clients(train_data_loaders, test_data_loader, global_model, n_workers):
    """
    Create a set of clients.
    train_data_loaders: 
    Task: Create a list containing a set of clients using Client()
    """
    clients = []
    # START CODING HERE

    return clients


def run_machine_learning(clients, global_model, n_epochs):
    """
    Complete machine learning over a series of clients.
    clients: A set of clients that needs to be trained on
    n_epochs: Number of communication rounds

    Task: 
    For every round, first you need to reinitialize the local models (using reinitialize_after_each_round method)
    Then run training on a subset of clients using function train_subset_of_clients
    Finally, the test accuracy and workers selected at each round will be appended to 
    epoch_test_set_results and worker_selection, respectively
    """

    epoch_test_set_results = [] # test accuracy on the test set at a round
    worker_selection = [] # chosen clients at a round

    for epoch in range(1, n_epochs + 1): # communication rounds
        pass # Remove this line and finish your task here

        # Reinitialize the local model
        # Your code goes here
            
        # Train the local model
        # Your code goes here

        # Append result and workers selected at a round to epoch_test_set_results and worker_selection, respectively
        # Your code goes here
        
    
    return epoch_test_set_results, worker_selection


def train_subset_of_clients(epoch, clients, global_model):
    random_workers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # You can change this
    for client_idx in random_workers:
        clients[client_idx].train(epoch) # machine learning

    print("Averaging client parameters")
    parameters = [clients[client_idx].get_nn_parameters() for client_idx in random_workers] # get params of chosen clients at a round
    new_nn_params = average_nn_parameters(parameters) # take the average
    global_model.load_state_dict(deepcopy(new_nn_params), strict=True) # update global params to the global model

    for client in clients:
        print("Updating parameters on client #{}", str(client.get_client_index()))
        client.update_nn_parameters(new_nn_params) # update global params to each client's params
    
    return clients[0].test(), random_workers # test accuracy

def run_exp():
    """
    Run an experiment
    """
    parser = argparse.ArgumentParser(description="Federated Learning Settings") # parsing arguments passed from terminal
    parser.add_argument("--dataset", type=str, help="Dataset", default="Cifar10")
    parser.add_argument("--n_workers", type=int, help="Number of workers", default=10)
    parser.add_argument("--cuda", type=bool, help="Use cuda", default=True)
    parser.add_argument("--n_epochs", type=int, help="Number of communication rounds", default=10)
    dataset = parser.parse_args().dataset # get argument dataset passed from terminal
    n_workers = parser.parse_args().n_workers # get argument n_workers passed from terminal
    cuda = parser.parse_args().cuda # get argument cuda passed from terminal
    n_epochs = parser.parse_args().n_epochs # get argument n_epochs passed from terminal
    train_dataset, test_dataset = get_dataset(dataset) # get train and test datasets

    train_loaders, test_data_loader, net_dataidx_map = generate_non_iid_data(train_dataset, test_dataset, dataset, n_workers)

    # Initialize a global model
    if cuda:
        global_model = Cifar10CNN().cuda()
        # global_model = ResNet18().cuda()
    else:
        global_model = Cifar10CNN()
        # global_model = ResNet18()
    
    clients = create_clients(train_loaders, test_data_loader, global_model, n_workers)

    results, worker_selection = run_machine_learning(clients, global_model, n_epochs)
    print("results: ", results)
    print("worker_selection: ", worker_selection)
