# Functions for the fast gradient sign method attack
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from IPython.display import clear_output


def apply_BIM(model, mean, std, image, label, alpha, epsilon, num_iterations=10):
    '''
    Applies given number of steps of the Basic Iterative Method (BIM) attack on the input image.
    
    Inputs:
    model          -- Net under attack
    image          -- Image data as tensor of shape (1, 3, 224, 224)
    mean           -- Mean
    std            -- Standard deviation
    label          -- Label from image as numpy array
    alpha          -- Hyperparameter for iterative step
    epsilon        -- Hyperparameter for sign method
    num_iterations -- Number of iterations to perform. Default is 10
    
    Returns:
    image_adver    -- Adversarial image as tensor
    '''

    # Convert label to torch tensor of shape (1)
    label = torch.tensor([label])

    # Check input image and label shapes
    assert(image.shape == torch.Size([1, 3, 224, 224]))
    assert(label.shape == torch.Size([1]))
    
    # Initialize adversarial image as image according to equation (3)
    image_adver = image.clone()    
    
    # Calculate normalized range [0, 1] and convert them to tensors
    zero_normed = [-m/s for m,s in zip(mean, std)]
    zero_normed = torch.tensor(zero_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    max_normed = [(1-m)/s for m,s in zip(mean,std)]
    max_normed = torch.tensor(max_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    # Calculated normalized epsilon and convert it to a tensor
    eps_normed = [epsilon/s for s in std]
    eps_normed = torch.tensor(eps_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    # Calculate the maximum change in pixel value using epsilon to be later used in clip function
    image_plus = image + eps_normed
    image_minus = image - eps_normed
    
    for i in range(num_iterations):
        
        # Make a copy and detach so the computation graph can be constructed
        image_adver = image_adver.clone().detach()
        image_adver.requires_grad=True
        
        # Compute cost with example image_adversarial        
        pred = model(image_adver)        
        loss = F.nll_loss(pred, label)        
        model.zero_grad()        
        loss.backward()        
        grad_x = image_adver.grad.data       
        
        # Check if gradient exists
        assert(image_adver.grad is not None)
               
        # Compute X_prime according to equation (1)
        image_prime = image_adver + alpha * grad_x.detach().sign()
        assert(torch.equal(image_prime, image_adver) == False)
      
        # Equation 1.2
        third_part_1 = torch.max(image_minus, image_prime)
        third_part = torch.max(zero_normed, third_part_1)
              
        # Equation (2)
        image_adver = torch.min(image_plus, third_part)                 
        image_adver = torch.min(max_normed, image_adver)                        

    return image_adver


def compute_all_bim(model, data_loader, predict, mean, std, epsilons):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples using BIM 
    in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/BIM-all_samples.csv

    Inputs:
    model       -- Neural net to attack
    data_loader -- Pytorch data loader object
    predict     -- Predict function
    mean        -- Mean used in data preprocessing
    std         -- Standard deviation used in data preprocessing
    epsilons    -- List of epsilons for FGSM attack

    Returns:
    top1        -- Top 1 accuracy
    top5        -- Top 5 accuracy
    conf        -- Confidence
    '''
    
    # Set parameters
    alpha = 1/255   

    # Initialize lists
    top1 = []
    top5 = []
    conf = []
    
    for epsilon in epsilons:        
        num_iterations = int(np.min([np.ceil(epsilon*255*4), np.ceil(1.25+(epsilon*255))]))
    
        top1_sub = []
        top5_sub = []
        conf_sub = []
                
        for sample in range(1000): 
            clear_output(wait=True)
            print("Running epsilon: {:.2f}".format(epsilon*255))
            print("Sample: {:.0f} of {}".format(sample, 1000))
            print("Number of iterations: {}".format(num_iterations))
            
            # Get data
            image_clean, target_class = data_loader.dataset[sample]
            image_clean.unsqueeze_(0)
            target_class.unsqueeze_(0)

            # Predict clean example
            _, _, gradient = predict(model, image_clean, target_class, return_grad=True)

            # Compute adversarial image and predict for it.    
            image_adv = apply_BIM(model, mean, std, image_clean, target_class, alpha, epsilon, num_iterations=num_iterations)
            class_adv, confidences_adv, _ = predict(model, image_adv, target_class, return_grad=False)
            
            # Compute accuracies:
            if class_adv[0] == target_class.squeeze().numpy():
                top1_sub.append(1)
            else:
                top1_sub.append(0)
                
            if target_class.squeeze().numpy() in class_adv:
                top5_sub.append(1)
            else:
                top5_sub.append(0)
                
            conf_sub.append(confidences_adv[0])
        
        # Get averages
        top1.append(np.mean(top1_sub))
        top5.append(np.mean(top5_sub))
        conf.append(np.mean(conf_sub))
        
    # Save as dataframe
    results = pd.DataFrame()
    results["Epsilon"] = list(np.array(epsilons) * 255)
    results["Top1"] = top1
    results["Top5"] = top5
    results["Confidence"] = conf
    results.to_csv("results/BIM-all_samples.csv")

    return top1, top5, conf