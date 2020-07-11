# Functions for the fast gradient sign method attack
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
from IPython.display import clear_output


def attack_FGSM(mean, std, image, epsilon, grad_x):
    '''
    Applies Fast Gradient Sign Method (FGSM) attack on the input image.
    
    Inputs:
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    image          -- Image data as tensor of shape (1, 3, 224, 224)  
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    grad_x         -- Gradient obtained from prediction with image on model
    
    Returns:
    image_tilde    -- Adversarial image as tensor
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


def visualize_attack_FGSM(data_loader, mean, std, model, predict, epsilon, sample, summarize_attack, folder=None):
    '''
    Generates an example using FGSM. Prints infos and plots clean, generated perturbance and resulting 
    adversarial image side-by-side.
    
    Inputs:
    data_loader      -- Pytorch data loader object
    mean             -- Mean from data preparation
    std              -- Standard deviation from data preparation
    model            -- Network under attack   
    predict          -- Predict function from module helper   
    epsilon          -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    sample           -- Index of sample 
    summarize_attack -- Function from module helper to describe attack
    folder           -- If given image will be saved to this folder
    '''

    # Get data
    image_clean, target_class = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    target_class.unsqueeze_(0)

    # Predict clean example
    labels, confidences, gradient = predict(model, image_clean, target_class, return_grad=True)
    label_clean = labels[0]
    conf_clean = confidences[0]
    
    # Compute adversarial image and predict for it.
    image_adv = attack_FGSM(mean, std, image_clean, epsilon, gradient)    
    labels, confidences, _ = predict(model, image_adv, target_class, return_grad=False)
    label_adv = labels[0]
    conf_adv = confidences[0]
    
    # Plot
    summarize_attack(image_clean, image_adv, conf_clean, conf_adv, label_clean, label_adv, target_class, sample,
                        folder=folder)


def get_attack_series(data_loader, mean, std, model, predict, epsilons, sample, show_tensor_image, save=False):
    '''
    Generates four adversaries with the specified epsilons and displays them along with the clean image.
    
    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    epsilon           -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    sample            -- Index of sample 
    show_tensor_image -- Converts tensor to image. From helper module
    save              -- Saves image series to folder FGSM is True

    Returns:
    output            -- Generated image series
    '''
      
    assert(len(epsilons) == 4)
    
    image_clean, target_class = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    target_class.unsqueeze_(0)

    # Predict clean example
    predicted_classes, confidences, gradient = predict(model, image_clean, target_class, return_grad=True)
    
    # Compute adversarial image and predict for it.
    image_0 = show_tensor_image(image_clean)
    image_1 = show_tensor_image(attack_FGSM(mean, std, image_clean, epsilons[0], gradient))
    image_2 = show_tensor_image(attack_FGSM(mean, std, image_clean, epsilons[1], gradient))
    image_3 = show_tensor_image(attack_FGSM(mean, std, image_clean, epsilons[2], gradient))
    image_4 = show_tensor_image(attack_FGSM(mean, std, image_clean, epsilons[3], gradient))

    # Put images side-by-side
    images = [image_0, image_1, image_2, image_3, image_4]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)
    output = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    
    for im in images:
      output.paste(im, (x_offset,0))
      x_offset += im.size[0]

    if save is True:
        output.save("plots/FGSM/FGSM-sample_" + str(sample) + "_series.png")
    
    return output


def confidence_range_attack_FGSM(data_loader, mean, std, model, predict, min_confidence, max_confidence):
    '''
    Attacks the model with images from the dataset on which the model achieves clean predictions with
    confidences in the provided interval [min_confidence, max_confidence]. Only if the original
    prediction is correct an adversary is generated.
    Returns an average of the top1, top5 and confidence for all these samples.
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    min_confidence -- Minimum confidence to consider
    max_confidence -- Maximum confidence to consider
    
    Returns:
    result         -- Dataframe with top1, top5 and confidence for prediction
    '''
    
    # Set perturbation
    epsilons = [0, 0.1/255, 0.2/255, 0.3/255, 0.4/255, 0.5/255, 0.7/255, 1/255, 2/255,
                4/255, 10/255, 20/255, 30/255, 40/255, 50/255, 60/255]

    # Take list results
    result = pd.read_csv("results/Clean-Predictions.csv", index_col=0)

    # Filter correct predictions
    samples = result.loc[result["Accuracy Top 1"] == 1]

    # Filter confidence
    samples = samples.loc[samples["Confidence"] > min_confidence]
    samples = samples.loc[samples["Confidence"] <= max_confidence]

    # Get samples
    samples = list(samples["Sample"])

    # Predict
    accurcy_top1 = []
    accurcy_top5 = []
    confidence_adversarial = []

    for epsilon in epsilons: 
        acc_sub_adver_top1 = []
        acc_sub_adver_top5 = []
        conf_sub_adver = []    
        i = 1

        for sample in samples:
            image_as_tensor, label = data_loader.dataset[sample]
            image_as_tensor.unsqueeze_(0)
            label.unsqueeze_(0)

            clear_output(wait=True)       
            print("Running for epsilon {:.2f}".format(epsilon*255))
            print("Sample: "+ str(i))
            print("Accuracy adversarial: {:.2f}".format(np.mean(acc_sub_adver_top1)))
            print("Confidence adversarial: {:.2f}".format(np.mean(conf_sub_adver)))

            # Predict with clean image
            gradient, corr, _, _, _ = predict(model, image_as_tensor, label, return_grad=True)

            # Generate adversarial example only if initial prediction was correct
            if corr == 1:            
                perturbed_data = attack_FGSM(mean, std,image_as_tensor, epsilon, gradient)
                _, top1, top5, conf, _ = predict(model, perturbed_data, label)
                acc_sub_adver_top1.append(top1)
                acc_sub_adver_top5.append(top5)
                conf_sub_adver.append(conf)

            else:
                acc_sub_adver_top1.append(0)
                acc_sub_adver_top5.append(0)
                conf_sub_adver.append(0)

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
    result.to_csv("results/FGSM/FGSM-conf" + str(int(min_confidence*100)) + ".csv") 
    
    return result


def single_attack_stats_FGSM(data_loader, mean, std, model, predict, epsilon, sample, idx_to_name):
    '''
    Computes FGSM attack and returns info about success.
    
    Inputs:
    data_loader    -- Pytorch data loader object
    mean           -- Mean from data preparation
    std            -- Standard deviation from data preparation
    model          -- Network under attack   
    predict        -- Predict function from module helper   
    epsilon        -- Hyperparameter for sign method. Has to be scaled to epsilon/255
    sample         -- Index of sample 
    idx_to_name    -- Function to return the class name from a class index. From module helper
    
    Returns:
    conf_adv       -- Confidence of adversary
    corr           -- Integer to indicate if predicted adversarial class is correct (1) or not (0)
    class_name_adv -- Label of adversarial class
    '''
    
    # Get data
    image_clean, class_index = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    class_index.unsqueeze_(0)

    # Predict clean example
    _, _, gradient = predict(model, image_clean, class_index, return_grad=True)
              
    # Compute adversarial image and predict for it.
    image_adv = attack_FGSM(mean, std, image_clean, epsilon, gradient)
    predicted_classes, confidences, _ = predict(model, image_adv, class_index, return_grad=False)
    
    # Check if attack successfully changed the class
    if predicted_classes[0] == class_index.squeeze().numpy():
        corr_adv = 1
    else: 
        corr_adv = 0
        
    conf_adv = confidences[0]
    
    
    class_name_adv = idx_to_name(predicted_classes[0])
        
    return conf_adv, corr_adv, class_name_adv


def analyze_attack_FGSM(data_loader, mean, std, model, predict, sample, epsilon_conf, show_tensor_image, idx_to_name, save_plot=False):
    '''
    Generates 4 plots: Image, conf over epsilon, top 5 conf for clean image, top 5 conf for adversarial image.
    
    Inputs:
    data_loader       -- Pytorch data loader object
    mean              -- Mean from data preparation
    std               -- Standard deviation from data preparation
    model             -- Network under attack   
    predict           -- Predict function from module helper   
    sample            -- Index of sample
    epsilon_conf      -- Epsilon for which to show the distribution in the last plot
    show_tensor_image -- Converts tensor to image. From helper module
    idx_to_name       -- Function to return the class name from a class index. From module helper
    save_plot         -- Saves the plot to folder FGSM if True
    ''' 

    # Get data
    image_clean, class_index = data_loader.dataset[sample]
    image_clean.unsqueeze_(0)
    class_index.unsqueeze_(0)

    # Predict clean example
    _, confidences_clean, gradient = predict(model, image_clean, class_index, return_grad=True)
    
    epsilons = [0, 0.1/255, 0.2/255, 0.3/255, 0.4/255, 0.5/255, 0.7/255, 1/255, 2/255, 3/255, 4/255, 5/255, 
                6/255, 8/255, 10/255, 12/255, 14/255, 16/255, 18/255, 20/255]

    conf_list = []
    acc_list = []

    print("Epsilon \t Accuracy \t Confidence \t Label")

    for epsilon in epsilons:
        conf_adv, acc, predicted_label = single_attack_stats_FGSM(data_loader, mean, std, model, predict, epsilon, sample, idx_to_name)
        conf_list.append(conf_adv)
        acc_list.append(acc)
        print(str(epsilon*255) + "\t\t\t" + str(acc) + "\t" + str(conf_adv) + "\t" + predicted_label) 
    
    # Compute top 5 confidences for selected epsilon
    image_adv = attack_FGSM(mean, std, image_clean, epsilon_conf, gradient)  
    _, confidences_adv, _ = predict(model, image_adv, class_index, return_grad=False)
    
       
    # Plot
    samples = [1, 2, 3, 4, 5]
    
    fig, axs = plt.subplots(1, 4, figsize=(15,3.75))

    ## First image: Clean image
    im = show_tensor_image(image_clean)

    axs[0].imshow(im)
    axs[0].axis('off')

    ## Second image: Confidence dist over epsilon and correct/incorrect
    axs[1].plot(np.array(epsilons)*255, conf_list, "-^", color='orange', label='Confidence')
    axs[1].plot(np.array(epsilons)*255, acc_list, "s", color='navy', label='1: Corr, 0: False')
    axs[1].set_ylim(0, 1.1)
    axs[1].xaxis.set_tick_params(labelsize=13)
    axs[1].yaxis.set_tick_params(labelsize=13)
    axs[1].set_xlabel("Epsilon * 255", fontsize=15)
    axs[1].legend()

    ## Third image: Clean image top 5 confidence
    axs[2].bar(samples, confidences_clean, color='orange')
    axs[2].set_ylim(0, 1.1)
    axs[2].xaxis.set_tick_params(labelsize=13)
    axs[2].yaxis.set_tick_params(labelsize=13)
    axs[2].set_xlabel("Top 5 classes", fontsize=15)

    ## Fourth image: Adversarial image selected epsilon top 5 confidence
    axs[3].bar(samples, confidences_adv, color='orange')
    axs[3].set_ylim(0, 1.1)
    axs[3].xaxis.set_tick_params(labelsize=13)
    axs[3].yaxis.set_tick_params(labelsize=13)
    axs[3].set_xlabel("Top 5 classes", fontsize=15)
    
    if save_plot is True:
        fig.tight_layout()
        fig.savefig("plots/FGSM/FGSM-individual_images-sample_" + str(sample) + ".png")


def all_samples_attack_FGSM(model, data_loader, predict, mean, std, epsilons):
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
            image_adv = attack_FGSM(mean, std, image_clean, epsilon, gradient)
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
    results.to_csv("results/FGSM/FGSM-all_samples.csv")
    
    return top1, top5, conf


def iterate_epsilons_FGSM(data_loader, mean, std, model, predict, sample, idx_to_name, single_attack_stats_FGSM):
    '''
    For a given sample number generates
    
    Inputs:
    data_loader              -- Pytorch data loader object
    mean                     -- Mean from data preparation
    std                      -- Standard deviation from data preparation
    model                    -- Network under attack   
    predict                  -- Predict function from module helper   
    sample                   -- Index of sample
    idx_to_name              -- Function to return the class name from a class index. From module helper
    single_attack_stats_FGSM -- Attacks model and returns stats
    '''
     
    epsilons = [0, 0.1/255, 0.2/255, 0.3/255, 0.4/255, 0.5/255, 0.7/255, 1/255, 2/255, 3/255, 4/255, 5/255, 
                6/255, 8/255, 10/255, 12/255, 14/255, 16/255, 18/255, 20/255]

    conf_list = []
    acc_list = []

    print("Epsilon \t Accuracy \t Confidence \t Label")

    for epsilon in epsilons:
        conf_adv, acc, predicted_label = single_attack_stats_FGSM(data_loader, mean, std, model, predict, epsilon, sample, idx_to_name)
        conf_list.append(conf_adv)
        acc_list.append(acc)
        print(str(epsilon*255) + "\t\t\t" + str(acc) + "\t" + str(conf_adv) + "\t" + predicted_label)

    plt.plot(np.array(epsilons)*255, conf_list, "-^", color='orange', label='Confidence')
    plt.plot(np.array(epsilons)*255, acc_list, "s", color='navy', label='1: Corr, 0: False')
    plt.xlabel("Epsilon *255", fontsize=15)
    plt.legend()
    plt.savefig("plots/FGSM/FGSM-confidence_levels-sample" + str(sample)+ ".png")
    plt.show()