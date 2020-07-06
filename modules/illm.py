# Functions for the fast gradient sign method attack
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from IPython.display import clear_output
import time


def attack_ILLM(mean, std, model, image, class_index, epsilon, alpha, num_iterations=10):
    '''
    Applies given number of steps of the Iterative Least Likely Method (ILLM) attack on the input image.
    
    Inputs:
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack
    image          -- Image data as tensor of shape (1, 3, 224, 224)
    class_index    -- Label from image as numpy array   
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    num_iterations -- Number of iterations to perform. Default is 10. It is recommended to use the heuristic from the
                      paper "Adversarial Examples in the Pysical World" to determine the number of iterations
    
    Returns:
    image_adver    -- Adversarial image as tensor
    '''

    # Convert label to torch tensor of shape (1)
    class_index = torch.tensor([class_index])

    # Check input image and label shapes
    assert(image.shape == torch.Size([1, 3, 224, 224]))
    assert(class_index.shape == torch.Size([1]))
    
    # Initialize adversarial image as image according to equation 3.1
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
        
        # Compute gradient of cost with least likely class     
        pred = model(image_adver)
        least_likeliest_class = torch.argmin(pred)
        least_likeliest_class.unsqueeze_(0)     
        loss = F.nll_loss(pred, least_likeliest_class)        
        model.zero_grad()        
        loss.backward()        
        grad_x = image_adver.grad.data       

        # Check if gradient exists
        assert(image_adver.grad is not None)
               
        # Compute X_prime according to equation 3.2
        image_prime = image_adver - alpha_normed * grad_x.detach().sign()
        assert(torch.equal(image_prime, image_adver) == False)
      
        # Equation 3.3 part 1
        third_part_1 = torch.max(image_minus, image_prime)
        third_part = torch.max(zero_normed, third_part_1)
              
        # Equation 3.3 part 2
        image_adver = torch.min(image_plus, third_part)                 
        image_adver = torch.min(max_normed, image_adver)                        

    return image_adver


def single_attack_stats_ILLM(data_loader, mean, std, model, predict, epsilon, alpha, sample, idx_to_name, num_iterations):
    '''
    Computes ILLM attack and returns info about success.
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    sample         -- Index of sample 
    idx_to_name    -- Function to return the class name from a class index. From module helper
    num_iterations -- Number of iterations to perform the ILLM with
    
    Returns:
    conf_adv       -- Confidence of adversary
    corr           -- Integer to indicate if predicted class is correct (1) or not (0)
    class_name_adv -- Label of adversarial class
    '''


    # Get data
    image_clean, class_index = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    class_index.unsqueeze_(0)

    # Predict clean example
    _, _, gradient = predict(model, image_clean, class_index, return_grad=True)
              
    # Compute adversarial image and predict for it.
    image_adv = attack_ILLM(mean, std, model, image_clean, class_index, epsilon, alpha, num_iterations=num_iterations)    
    predicted_classes, confidences, _ = predict(model, image_adv, class_index, return_grad=False)
    
    
    if predicted_classes[0] == class_index.squeeze().numpy():
        corr_adv = 1
    else: 
        corr_adv = 0
        
    conf_adv = confidences[0] 
    class_name_adv = idx_to_name(predicted_classes[0])
        
    return conf_adv, corr_adv, class_name_adv


def visualize_attack_ILLM(data_loader, mean, std, model, predict, epsilon, alpha, sample, summarize_attack, folder=None):
    '''
    Generates an adversary using ILLM. Prints infos and plots clean, generated perturbance and resulting adversarial image side-by-side.
    
    Inputs:
    data_loader      -- Pytorch data loader object
    mean             -- Mean from data preparation
    std              -- Standard deviation from data preparation
    model            -- Network under attack   
    predict          -- Predict function from module helper   
    epsilon          -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha            -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    sample           -- Index of sample 
    summarize_attack -- Function from module helper to describe attack
    folder           -- If given image will be saved to this folder
    '''
    
    num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))
    print("Number of iterations: " + str(num_iterations))
    
    
    # Get data
    image_clean, target_class = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    target_class.unsqueeze_(0)

    # Predict clean example
    labels, confidences, gradient = predict(model, image_clean, target_class, return_grad=True)
    label_clean = labels[0]
    conf_clean = confidences[0]
    
    # Compute adversarial image and predict for it.
    image_adv = attack_ILLM(mean, std, model, image_clean, target_class, epsilon, alpha, num_iterations=num_iterations)
    labels, confidences, _ = predict(model, image_adv, target_class, return_grad=False)
    label_adv = labels[0]
    conf_adv = confidences[0]

    # Plot
    summarize_attack(image_clean, image_adv, conf_clean, conf_adv, label_clean, label_adv, target_class, sample,
                        folder=folder)


def all_samples_attack_ILLM(data_loader, mean, std, model, predict, epsilons, alpha, filename_ext, restart=None):
    '''
    Computes top 1, top 5 accuracy and confidence for all samples using ILLM 
    in data_loader for each epsilon.
    Does not filter false initial predictions.
    Saves the results as csv file to: ./results/ILLM/ILLM-all_samples.csv

    Inputs:
    data_loader   -- Pytorch data loader object
    mean          -- Mean from data preparation
    std           -- Standard deviation from data preparation
    model         -- Network under attack   
    predict       -- Predict function from module helper   
    epsilons      -- List of hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha         -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    filename_ext  -- Extension to file name
    restart       -- List to use previous partial results. 
                     Format: [<filename>, <[remaining_epsilons]>]

    Returns:
    top1          -- Top 1 accuracy
    top5          -- Top 5 accuracy
    conf          -- Confidence
    ''' 

    # Initialize lists
    top1 = []
    top5 = []
    conf = []
    
    
    if restart != None:
        epsilons = restart[1]
        print("Resetting epsilons. New values: " + str(np.array(epsilons)*255))
        

    for epsilon in epsilons:
        num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))
    
        top1_sub = []
        top5_sub = []
        conf_sub = []
        calculated_samples = 0
        
        if restart != None:
            val = pd.read_csv("results/ILLM/temp-all_samples-epsilon_" + str(restart[0]) + ".csv", index_col=0)
            top1_sub = list(val["top1"])
            top5_sub = list(val["top5"])
            conf_sub = list(val["conf"])
            
            calculated_samples = len(top1_sub)

            print("Continuing at sample {:.0f} for epsilon {:.0f}".format(calculated_samples, epsilon))
            time.sleep(3)

        counter = 0

        
        for sample in range(calculated_samples, 1000): 
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
            image_adv = attack_ILLM(mean, std, model, image_clean, target_class, epsilon, alpha, num_iterations=num_iterations)
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
            calculated_samples = 0

            # Save intermediate results
            if counter % 20==0:
                temp = pd.DataFrame()
                temp["top1"] = top1_sub
                temp["top5"] = top5_sub
                temp["conf"] = conf_sub
                temp.to_csv("results/ILLM/temp-all_samples-epsilon_" + str(epsilon*255) + "-range_" + str(filename_ext) + ".csv")
                counter = 0

        # Get averages
        top1.append(np.mean(top1_sub))
        top5.append(np.mean(top5_sub))
        conf.append(np.mean(conf_sub))
        
        restart = None
        
    # Save as dataframe
    results = pd.DataFrame()
    results["Epsilon"] = list(np.array(epsilons) * 255)
    results["Top1"] = top1
    results["Top5"] = top5
    results["Confidence"] = conf
    results.to_csv("results/ILLM/ILLM-all_samples-range_" + str(filename_ext) + ".csv")

    return top1, top5, conf


def confidence_range_attack_ILLM(data_loader, mean, std, model, predict, epsilons, alpha, min_confidence, max_confidence):
    '''
    Attacks the model with images from the dataset on which the model achieves clean predictions with
    confidences in the provided interval [min_confidence, max_confidence]. Only if the original
    prediction is correct an adversary is generated.
    
    Returns an average of the top1, top5 and confidence for all these samples.
    
    The number of iterations for the ILLM attack is calculated by this function according to the heuristic
    from the authors of "Adversarial Examples in the Physical World".
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    epsilons       -- List of hyperparameter for sign method. Has to be scaled to epsilon/255
    alpha          -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    min_confidence -- Minimum confidence to consider
    max_confidence -- Maximum confidence to consider
    
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
                image_adversarial = attack_ILLM(mean, std, model, image_clean, target_class, epsilon, alpha, num_iterations=num_iterations)
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
    result.to_csv("results/ILLM/ILLM-Conf" + str(int(min_confidence*100)) + ".csv") 
    
    return result


def analyze_attack_ILLM(data_loader, mean, std, model, predict, alpha, sample, epsilon_conf, show_tensor_image, idx_to_name, fixed_num_iter=None, save_plot=False, print_output=True):
    '''
    Generates 4 plots: Image, confidence over epsilon, top 5 confidence for clean image, top 5 confidence for adversarial image.
    The epsilons are: 0, 0.5/255, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255, 20/255

    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    alpha             -- Hyperparameter for iterative step as absolute value. Has to be scaled to alpha/255
    epsilon_conf      -- Epsilon for which to show the distribution in the last plot
    show_tensor_image -- Converts tensor to image. From helper module
    idx_to_name       -- Function to return the class name from a class index. From module helper
    fixed_num_iter    -- Fixed number of iterations for ILLM. Calculates the recommended number for each epsilon if not given
    save_plot         -- Saves the plot to folder ILLM if True
    print_output      -- Prints stats if True
    '''  

    # Get data
    image_clean, class_index = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    class_index.unsqueeze_(0)

    # Predict clean example
    _, confidences_clean, gradient = predict(model, image_clean, class_index, return_grad=True)
    
    epsilons = [0, 0.5/255, 1/255, 2/255, 4/255, 8/255, 12/255, 16/255, 20/255]

    conf_list = []
    acc_list = []

    print("Epsilon \t Iterations \t Accuracy \t Confidence \t Label")
    
    for epsilon in epsilons:
        if fixed_num_iter == None:
            num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))
        else:
            num_iterations = fixed_num_iter
        
        conf_adv, acc, predicted_label = single_attack_stats_ILLM(data_loader, mean, std, model, predict, epsilon, alpha, sample, idx_to_name, num_iterations)
        conf_list.append(conf_adv)
        acc_list.append(acc)
        
        if print_output == True:
            print(str(epsilon*255) + "\t\t\t" + str(num_iterations) + "\t\t\t" + str(acc) + "\t" + str(conf_adv) + "\t" + predicted_label) 
    
    # Compute top 5 confidences for selected epsilon
    ## Number of iterations
    if num_iterations == None:
        num_iterations = int(np.min([np.ceil( (epsilon/alpha) + 4 ), np.ceil( 1.25 * epsilon/alpha ) ]))

    image_adv = attack_ILLM(mean, std, model, image_clean, class_index, epsilon_conf, alpha, num_iterations=num_iterations)    
    
    _, confidences_adv, _ = predict(model, image_adv, class_index, return_grad=False)
    
     
    # Plot
    samples = [1, 2, 3, 4, 5]
    
    fig, axs = plt.subplots(1, 4, figsize=(20,5))

    ## First image: Clean image
    im = show_tensor_image(image_clean)

    axs[0].imshow(im)
    axs[0].axis('off')

    ## Second image: Confidence dist over epsilon and correct/incorrect
    axs[1].plot(np.array(epsilons)*255, conf_list, "-^", color='orange', label='Confidence')
    axs[1].plot(np.array(epsilons)*255, acc_list, "s", color='navy', label='1: Corr, 0: False')
    axs[1].set_ylim(0, 1.1)
    axs[1].set_xlabel("Epsilon *255", fontsize=10)
    axs[1].legend()

    ## Third image: Clean image top 5 confidence
    axs[2].bar(samples, confidences_clean, color='orange')
    axs[2].set_ylim(0, 1.1)
    axs[2].set_xlabel("Epsilon *255", fontsize=10)

    ## Fourth image: Adversarial image selected epsilon top 5 confidence
    axs[3].bar(samples, confidences_adv, color='orange')
    axs[3].set_ylim(0, 1.1)
    axs[3].set_xlabel("Epsilon *255", fontsize=10)
    
    if save_plot is True:
        fig.tight_layout()
        fig.savefig("plots/ILLM/ILLM-Individual_Images-Sample_" + str(sample) + ".png")