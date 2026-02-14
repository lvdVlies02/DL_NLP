import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn

class OptimizerTemplate:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr
        
    def zero_grad(self):
        ## Set gradients of all parameters to zero
        for p in self.params:
            if p.grad is not None:
                p.grad.detach_() # For second-order optimizers important
                p.grad.zero_()
    
    @torch.no_grad()
    def step(self):
        ## Apply update step to all parameters
        for p in self.params:
            if p.grad is None: # We skip parameters without any gradients
                continue
            self.update_param(p)
            
    def update_param(self, p):
        # To be implemented in optimizer-specific classes
        raise NotImplementedError


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs):
    accuracies = []
    iter = 0
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Load images as Variable
            images = images.view(-1, 28*28).requires_grad_()
            
            # Clear gradients w.r.t. parameters
            optimizer.zero_grad()
            
            # Forward pass to get output/logits
            outputs = model(images)
            
            # Calculate Loss: softmax --> cross entropy loss
            loss = criterion(outputs, labels)
            
            # Getting gradients w.r.t. parameters
            loss.backward()
            
            # Updating parameters
            optimizer.step()
            
            iter += 1
            
            if iter % 500 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset

                with torch.no_grad():  # Disable gradient computation for evaluation
                    for images, labels in test_loader:
                        # Load images to a Torch Variable
                        images = images.view(-1, 28*28)
                        
                        # Forward pass only to get logits/output
                        outputs = model(images)
                        
                        # Get predictions from the maximum value
                        _, predicted = torch.max(outputs.data, 1)
                        
                        # Total number of labels
                        total += labels.size(0)
                        
                        # Total correct predictions
                        correct += (predicted == labels).sum()
                
                accuracy = 100 * correct / total
                
                # Print Loss
                print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.item(), accuracy))

                accuracies.append(accuracy.item())

    return accuracies