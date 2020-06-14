# Helper functions for the adversarial example notebooks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def idx_to_name(idx):
    '''
    Converts the output class index from the googleNet to the respective name.
    
    Input:
    idx  -- Class index as integer
    
    Returns:
    name -- Class names corresponding to idx as string
    '''
    
    ## Load dictionary from file    
    names = pd.read_csv("./data/ImageNet_subset/categories.csv")
    
    ## Retrieve class name for idx
    name = names.iloc[idx]["CategoryName"]
    
    return name


def show_tensor_image(tensor):
    '''
    De-normalizes an image as a tensor and converts it back into an 8bit image object.
    
    Inputs:
    tensor -- PyTorch tensor of shape (1, 3, 224, 224)
    
    Returns:
    image  -- De-normalized image object
    '''
    
    ## Detach computation graph and remove batch dimension
    tensor = tensor.detach().clone()    
    tensor.squeeze_()
    
    ## De-normalize tensor image
    invert_preprocess = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    ])
      
    image = invert_preprocess(tensor)      
    image = np.array(image.detach())
    
    ## Rescale to range 0-255 and convert datatype into 8bit
    image = image * 255    
    image = np.uint8(image)
    
    ## Swap axes to get the expected shape (224, 224, 3)
    image = np.swapaxes(image, 0, 2)
    
    ## Rotate and flip the image, then convert to image object
    image = np.flipud(np.rot90(image))
    
    image = Image.fromarray(image)
    
    ## Show image
    plt.imshow(image)
        
    return image


def predict(model, image, target_label, return_grad=False):
    '''
    Predicts the class of the given image and compares the prediction with the provided label.
    
    Inputs:
    model           -- net
    image           -- Input image as tensor of shape (1, 3, 224, 224)
    target_label    -- Target label as tensor of shape (1)
    return_grad     -- Returns gradient if set True
    
    Returns:
    gradient        -- None if return_grad=False. Otherwise the gradient from the prediction 
                       as a tensor of shape ()
    top_1           -- Integer of value 1 if class is correct, otherwise 0
    top_5           -- Integer of value 1 if target class is among the 5 most confident predicted classes
    confidence      -- Confidence of prediction
    predicted_label -- Predicted label as integer
    '''      
        
    if return_grad == True:
        image.requires_grad=True
        prediction = model(image)
               
        # Zero gradients
        model.zero_grad()

        # Calculate loss using the class index for pandas and get gradient
        loss = F.nll_loss(prediction, target_label)
        loss.backward()
        gradient = image.grad.data
        
    else:           
        gradient = None
        with torch.no_grad():
            prediction = model(image)
   

    ## Get class index and confidence for prediction 
    prediction = torch.nn.functional.softmax(prediction[0].cpu().detach(), dim=0).numpy()
    
    
    ## Get class label indices corresponding to the five highest confidences
    predicted_class_index = prediction.argsort()[-5:][::-1]

        
    ## Get largest confidences
    confidence = prediction[predicted_class_index[0]]
    
    
    ## Calculate if prediction is correct        
    if predicted_class_index[0] == target_label:
        top_1 = 1
        
    else:
        top_1 = 0
     
    
    ## Calculate top 5 accuracy
    if target_label.numpy() in predicted_class_index:
        top_5 = 1
    else:
        top_5 = 0
    
    
    return gradient, top_1, top_5, confidence, predicted_class_index[0]


def plot_examples(image_clean, image_adv, conf_clean, conf_adv, label_clean, label_adv, label_target):
    '''
    Plots the clean and adversarial image side-by-side. Prints predicted labels and confidences for both.
    
    Inputs:
    image_clean     -- Clean image as tensor of shape (1, 1, 28, 28)
    image_adv       -- Adversarial image as tensor of shape (1, 1, 28, 28)
    conf_clean      -- Confidence for the clean image
    conf_adv        -- Confidence for the adversarial image
    label_clean     -- Predicted label from the clean image
    label_adv       -- Predicted label from the adversarial image
    label_target    -- Target label as tensor of shape (1)
    '''
   
    ## Get label names from index
    name_target = idx_to_name(label_target.detach().numpy()[0])
    name_clean = idx_to_name(label_clean)
    name_adv = idx_to_name(label_adv)
    
    ## Isloate perturbance
    perturbance = image_adv - image_clean
    
    ## Text
    print("\t\t\tClean image\t Adversarial image\n")    
    print("Actual class: \t\t{}\t\t\t{}".format(name_target, name_target ))
    print("Predicted class: \t{}\t\t\t{}".format(name_clean, name_adv ))
    print("Confidence: \t\t{:.2f}%\t\t\t\t{:.2f}%\n".format(conf_clean*100, conf_adv*100))
    
    ## Plots
    plt.subplot(221)
    plt.title("Clean example", fontsize=30)
    show_tensor_image(image_clean)
    plt.subplot(222)
    plt.title("Perturbance", fontsize=30)
    show_tensor_image(perturbance)
    plt.subplot(223)
    plt.title("Adversarial example", fontsize=30)
    show_tensor_image(image_adv)