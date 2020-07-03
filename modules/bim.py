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
    model          -- Network under attack
    image          -- Image data as tensor of shape (1, 3, 224, 224)
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    label          -- Label from image as numpy array
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255.
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255.
    num_iterations -- Number of iterations to perform. Default is 10. It is recommended to use the heuristic from the
                      paper "Adversarial Examples in the Pysical World" to determine the number of iterations.
    
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
    
    # Calculate normalized alpha
    alpha_normed = [alpha/s for s in std]
    alpha_normed = torch.tensor(alpha_normed, dtype=torch.float).unsqueeze(-1).unsqueeze(-1)

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
        image_prime = image_adver + alpha_normed * grad_x.detach().sign()
        assert(torch.equal(image_prime, image_adver) == False)
      
        # Equation 1.2
        third_part_1 = torch.max(image_minus, image_prime)
        third_part = torch.max(zero_normed, third_part_1)
              
        # Equation (2)
        image_adver = torch.min(image_plus, third_part)                 
        image_adver = torch.min(max_normed, image_adver)                        

    return image_adver


def compute_all_bim(model, data_loader, predict, mean, std, epsilons, alpha, filename_ext):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples using BIM 
    in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/BIM/BIM-all_samples.csv

    Inputs:
    model       -- Network under attack
    data_loader -- Pytorch data loader object
    predict     -- Predict function from module helper
    mean        -- Mean from data preparation
    std         -- Standard deviation from data preparation
    epsilons    -- List of epsilons for FGSM attack
    alpha       -- Hyperparameter for BIM. Must be provided as a scaled number alpha/255

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
        num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))
    
        top1_sub = []
        top5_sub = []
        conf_sub = []

        counter = 0

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

            counter += 1

            # Save intermediate results
            if counter % 20==0:
                temp = pd.DataFrame()
                temp["top1"] = top1_sub
                temp["top5"] = top5_sub
                temp["conf"] = conf_sub
                temp.to_csv("results/BIM/temp-all_samples-epsilon_" + str(epsilon) + ".csv")
                counter = 0

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
    results.to_csv("results/BIM/BIM-all_samples-part_" + str(filename_ext) + ".csv")

    return top1, top5, conf


def BIM_attack_with_selected_samples(min_confidence, max_confidence, data_loader, predict, model, mean, std, epsilons, alpha):
    '''
    Attacks the model with images from the dataset on which the model achieves clean predictions with
    confidences in the provided interval [min_confidence, max_confidence]. Only if the original
    prediction is correct an adversary is generated.
    
    Returns an average of the top1, top5 and confidence for all these samples.
    
    The number of iterations for the BIM attack is calculated by this function according to the heuristic
    from the authors of "Adversarial Examples in the Physical World".
    
    Inputs:
    min_confidence -- Minimum confidence to consider
    max_confidence -- Maximum confidence to consider
    model          -- Network under attack
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    epsilons       -- Hyperparameter 1 for attack. Provide scaled as epsilon/255 
    alpha          -- Hyperparameter 2 for attack. Provide scaled as alpha/255
    
    Returns:
    result         -- Dataframe with top1, top5 and confidence for prediction
    '''
    
    # Take list results
    result = pd.read_csv("results/Clean-Predictions.csv", index_col=0)

    # Filter correct predictions
    samples = result.loc[result["Accuracy Top 1"] == 1]

    # Filter confidence
    samples = samples.loc[samples["Confidence 1"] > min_confidence]
    samples = samples.loc[samples["Confidence 1"] <= max_confidence]

    # Get samples
    samples = list(samples.index)

    # Predict
    accurcy_top1 = []
    accurcy_top5 = []
    confidence_adversarial = []

    for epsilon in epsilons: 
        
        # Compute number of iterations
        num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))
        
        acc_sub_adver_top1 = []
        acc_sub_adver_top5 = []
        conf_sub_adver = []    
        i = 1

        for sample in samples:
            image_clean, target_class = data_loader.dataset[sample]
            image_clean.unsqueeze_(0)
            target_class.unsqueeze_(0)

            clear_output(wait=True)       
            print("Running for epsilon {:.2f}".format(epsilon*255))
            print("Sample: "+ str(i) + " of " + str(len(samples)))
            print("Accuracy top 1 adversarial: {:.2f}".format(np.mean(acc_sub_adver_top1)))
            print("Accuracy top 5 adversarial: {:.2f}".format(np.mean(acc_sub_adver_top5)))
            print("Confidence adversarial: {:.2f}".format(np.mean(conf_sub_adver)))
            print("Number of iterations: {}".format(num_iterations))

            # Predict with clean image
            predicted_classes, _, gradient = predict(model, image_clean, target_class, return_grad=True)
            
            # Generate adversarial example only if initial prediction was correct
            if predicted_classes[0] == target_class.squeeze().numpy():            
                
                # Predict with adversarial image
                image_adversarial = apply_BIM(model, mean, std, image_clean, target_class, alpha, epsilon, num_iterations=num_iterations)
                predicted_classes, confidences, _ = predict(model, image_adversarial, target_class, return_grad=False)
                

                if predicted_classes[0] == target_class.squeeze().numpy():
                    acc_sub_adver_top1.append(1)
                else:
                    acc_sub_adver_top1.append(0)
                    
                if target_class.squeeze().numpy() in predicted_classes:
                    acc_sub_adver_top5.append(1)
                    
                else:
                    acc_sub_adver_top5.append(0)
                    
                conf_sub_adver.append(confidences[0])

            i += 1

        # Add accuracies and confidences for clean and adversarial example
        accurcy_top1.append(np.mean(acc_sub_adver_top1))
        accurcy_top5.append(np.mean(acc_sub_adver_top5))
        confidence_adversarial.append(np.mean(conf_sub_adver))

    # Save results
    result = pd.DataFrame()
    epsilon = np.array(epsilons) * 255
    result["Epsilon_255"] = epsilon
    result["Accuracy Top 1"] = accurcy_top1
    result["Accuracy Top 5"] = accurcy_top5
    result["Confidence"] = confidence_adversarial
    result.to_csv("results/BIM/BIM-Conf" + str(int(min_confidence*100)) + ".csv") 
    
    return result


def compare_examples_bim(data_loader, mean, std, model, predict, summarize_attack, alpha, epsilon, idx, folder=None):
    '''
    Generates an example using BIM. Prints infos and plots clean and adversarial image side-by-side.
    
    Inputs:
    data_loader      -- Pytorch data loader object
    mean             -- Mean from data preparation
    std              -- Standard deviation from data preparation
    model            -- Network under attack
    predict          -- Predict function from module helper
    summarize_attack -- Function from module helper to describe attack
    alpha            -- Hyperparameter for BIM
    epsilon          -- Hyperparameter for BIM
    idx              -- Index of sample   
    folder           -- If given image will be saved to this folder
    '''
    
    num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))
    print("Number of iterations: " + str(num_iterations))
    
    
    # Get data
    image_clean, target_class = data_loader.dataset[idx]
    image_clean.unsqueeze_(0)
    target_class.unsqueeze_(0)

    # Predict clean example
    labels, confidences, gradient = predict(model, image_clean, target_class, return_grad=True)
    label_clean = labels[0]
    conf_clean = confidences[0]
    
    # Compute adversarial image and predict for it.    
    image_adv = apply_BIM(model, mean, std, image_clean, target_class, alpha, epsilon, num_iterations=num_iterations)
    labels, confidences, _ = predict(model, image_adv, target_class, return_grad=False)
    label_adv = labels[0]
    conf_adv = confidences[0]
    
    
    # Plot
    summarize_attack(image_clean, image_adv, conf_clean, conf_adv, label_clean, label_adv, target_class, idx,
                        folder=folder)