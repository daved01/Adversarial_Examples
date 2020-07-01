# Functions for the fast gradient sign method attack
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output


def apply_fgsm(image, mean, std, epsilon, grad_x):
    '''
    Generates adversarial image from the input image using the Fast Gradient Sign Method (FGSM).
    
    Inputs:
    image       -- Image data as tensor
    mean        -- Means from image standardization
    std         -- Standard deviation from image standardization
    epsilon     -- Hyperparameter
    grad_x      -- Gradient of the cost with respect to x
    
    Returns:
    image_tilde -- Adversarial image as tensor
    '''
    
    ## Calculated normalized epsilon and convert it to a tensor   
    eps_normed = [epsilon/s for s in std]
    eps_normed = torch.tensor(eps_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    ## Compute eta part
    eta = eps_normed * grad_x.sign()

    ## Apply perturbation
    image_tilde = image + eta    
    
    ## Clip image to maintain the range [min, max]
    image_tilde = torch.clamp(image_tilde, image.detach().min(), image.detach().max())
    
    ## Calculate normalized range [0, 1] and convert them to tensors
    zero_normed = [-m/s for m,s in zip(mean, std)]
    zero_normed = torch.tensor(zero_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    max_normed = [(1-m)/s for m,s in zip(mean,std)]
    max_normed = torch.tensor(max_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)
    
    ## Clip image so after denormalization and destandardization, the range is [0, 255]
    image_tilde = torch.max(image_tilde, zero_normed)
    image_tilde = torch.min(image_tilde, max_normed)
    
    return image_tilde


def compute_all_fgsm(model, data_loader, predict, mean, std, epsilons):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/FGSM-all_samples.csv

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
    
    # Initialize lists
    top1 = []
    top5 = []
    conf = []
    
    for epsilon in epsilons:        

        top1_sub = []
        top5_sub = []
        conf_sub = []
      
        for sample in range(1000): 
            clear_output(wait=True)
            print("Running epsilon: {:.2f}".format(epsilon*255))
            print("Sample: {:.0f} of {}".format(sample, 1000))
            
            
            # Get data
            image_clean, target_class = data_loader.dataset[sample]
            image_clean.unsqueeze_(0)
            target_class.unsqueeze_(0)

            # Predict clean example
            _, _, gradient = predict(model, image_clean, target_class, return_grad=True)

            # Compute adversarial image and predict for it.    
            image_adv = apply_fgsm(image_clean, mean, std, epsilon, gradient)
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
    results.to_csv("results/FGSM-all_samples.csv")
    
    return top1, top5, conf