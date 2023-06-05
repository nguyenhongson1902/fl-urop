import torch
import torchvision.transforms as transforms
import torchvision
from copy import deepcopy

from schedulers.min_lr_step import MinCapableStepLR


class Client:
    def __init__(self, client_idx, train_data_loader, test_data_loader, global_model):
        self.client_idx = client_idx
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.net = deepcopy(global_model) # Initialize a local model
        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.5)
        self.scheduler = MinCapableStepLR(self.optimizer, 50, 0.5, 1e-10)

        self.device = self.initialize_device()

    def reinitialize_after_each_round(self, global_model):
        """
        global_model: The old global model
        """
        self.net = deepcopy(global_model)
        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.01, momentum=0.5)
        self.scheduler = MinCapableStepLR(self.optimizer, 50, 0.5, 1e-10)

    def initialize_device(self):
        """
        Creates appropriate torch device for client operation.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()
    
    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx
    
    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(deepcopy(new_params), strict=True)

    def train(self, epoch):
        """
        epoch: Communication round #
        Task:
        Do forwarding to get the outputs. E.g: outputs = model(inputs)
        Compute loss. E.g: loss_func(outputs, labels)
        Do backwarding using the loss above
        Update weights using optimizer. E.g: optimizer.step()

        """
        self.net.train() # switch to training mode, do this before training

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(self.train_data_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            # Forward
            # Your code goes here
            outputs = None

            # Compute loss
            # Your code goes here
            loss = None

            # Backward
            # Your code goes here

            # Update weights
            # Your code goes here

            # print statistics
            running_loss += loss.item()
            if i % 100 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / 100))

                running_loss = 0.0

        self.scheduler.step()
    
    def test(self):
        """
        Compute test accuracy
        Task:
        Get images and labels (refer back to the train function above)
        Get the outputs
        Compute loss
        Compute predicted class. Hint: Use torch.max(..., axis=1)
        Compute the total number of test dataset samples
        Compute the number of classes correctly predicted
        """
        self.net.eval() # switch to evaluation mode, do this before testing 

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.49421428, 0.48513139, 0.45040909), (0.24665252, 0.24289226, 0.26159238)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=False, transform=transform_test)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

        correct, total = 0, 0
        for i, (images, labels) in enumerate(test_loader): # original test CIFAR-10
            images, labels = None, None
            with torch.no_grad():
                # Your code goes here
                logits = None
                out_loss = None
                _, predicted = None
                total += None
                correct += None
        acc_clean = correct / total
        print('\nTest Accuracy %.2f' % (acc_clean*100))
        print('Test Loss:', out_loss)

        return acc_clean
