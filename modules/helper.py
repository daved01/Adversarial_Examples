# Helper functions for the adversarial example notebooks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def idx_to_name(class_index):
    '''
    Converts the output class index from the googleNet to the respective name.
    
    Input:
    class_index  -- Class index as integer
    
    Returns:
    name         -- Class names corresponding to idx as string
    '''
    
    # Load dictionary from file    
    names = pd.read_csv("./data/ImageNet_subset/categories.csv")
    
    # Retrieve class name for idx
    name = names.iloc[class_index]["CategoryName"]
    
    return name

def show_tensor_image(tensor):
    '''
    De-normalizes an image as a tensor and converts it back into an 8bit image object.
    
    Inputs:
    tensor -- PyTorch tensor of shape (1, 3, 224, 224)
    
    Returns:
    image  -- De-normalized image object
    '''
    
    # Detach computation graph and remove batch dimension
    tensor = tensor.detach().clone()    
    tensor.squeeze_()
    
    # De-normalize tensor image
    invert_preprocess = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
    ])
      
    image = invert_preprocess(tensor)      
    image = np.array(image.detach())
    
    # Rescale to range 0-255 and convert datatype into 8bit
    image = image * 255    
    image = np.uint8(image)
    
    # Swap axes to get the expected shape (224, 224, 3)
    image = np.swapaxes(image, 0, 2)
    
    # Rotate and flip the image, then convert to image object
    image = np.flipud(np.rot90(image))
    
    image = Image.fromarray(image)
        
    return image

def predict(model, image, target_label, return_grad=False):
    '''
    Predicts the class of the given image and compares the prediction with the provided label.
    
    Inputs:
    model             -- net
    image             -- Input image as tensor of shape (1, 3, 224, 224)
    target_label      -- Target label as tensor of shape (1)
    return_grad       -- Returns gradient if set True
    
    Returns:
    predicted_classes -- Numpy array of top 5 predicted class indices
    confidences       -- Numpy array of top 5 confidences in descending order
    gradient          -- None if return_grad=False. Otherwise the gradient from the prediction
                         as a tensor of shape ().
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

    # Get class index and confidence for prediction 
    prediction = torch.nn.functional.softmax(prediction[0].cpu().detach(), dim=0).numpy()
   
    # Get top 5 class indices
    predicted_classes = prediction.argsort()[-5:][::-1]
        
    # Get largest confidences
    confidences = prediction[predicted_classes]
    
    return predicted_classes, confidences, gradient

def summarize_attack(image_clean, image_adv, conf_clean, conf_adv, label_clean, label_adv, label_target, idx,
                    folder=None):
    '''
    Summarizes attack by printing info and displaying the image along with the adversary and the isolated
    perturbance. Saves image to the folder.
    
    Inputs:
    image_clean     -- Clean image as tensor of shape (1, 1, 28, 28)
    image_adv       -- Adversarial image as tensor of shape (1, 1, 28, 28)
    conf_clean      -- Confidence for the clean image
    conf_adv        -- Confidence for the adversarial image
    label_clean     -- Predicted label from the clean image
    label_adv       -- Predicted label from the adversarial image
    label_target    -- Target label as tensor of shape (1)
    idx             -- Sample index used for filename of plot export
    folder          -- If not None folder to which the image is saved.
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
    f = plt.figure(figsize=(20,20))
    
    ax = f.add_subplot(221)
    ax2 = f.add_subplot(222)
    ax3 = f.add_subplot(223)
    
    im1 = show_tensor_image(image_clean)
    im2 = show_tensor_image(perturbance)
    im3 = show_tensor_image(image_adv)
    
    ax.imshow(im1)
    ax.set_title("Clean example", fontsize=25)
    ax.axis('off')
    ax2.imshow(im2)
    ax2.set_title("Perturbance", fontsize=25)
    ax2.axis('off')
    ax3.imshow(im3)
    ax3.set_title("Adversarial example", fontsize=25)
    ax3.axis('off')
    
    ## Save figure
    if folder is not None:
        f.tight_layout()
        f.savefig("plots/" + str(folder) + "/" + str(folder) + "-sample_" + str(int(idx)) + "_pair.png")
        
def normed_difference(img1, img2):
    '''
    Returns the Frobenius normed difference between two pytorch tensor images. Applies the norm on the second dimension.
    
    Inputs:
    img1 - Tensor of the first image of shape (1, 3, x0, x1)
    img2 - Tensor of the second image of shape (1, 3, x0, x1)
    
    Returns:
    Tensor of the normed difference (1, x0, x1)
    '''
    
    img_diff = img1 - img2
    return torch.norm(img_diff, p='fro', dim=1)

def avg_normed_difference(img1, img2):
    '''
    Returns the Frobenius normed difference between two pytorch tensor images averaged across the entire image.
    Applies the norm across the second dimension.
    
    Inputs:
    img1 - Tensor of the first image of size (1, 3, x0, x1)
    img2 - Tensor of the second image (1, 3, x0, x1)
    
    Returns:
    The normed difference between the two images averaged over the entire image difference. Numpy array(1)
    '''
    
    img_normed = normed_difference(img1, img2)
    return torch.mean(img_normed, dim=(1,2)).detach().cpu().numpy()

def denormalize_image(input_image, means, stds):
    # Detach computation graph and remove batch dimension
    image = input_image.detach().clone()    
    image.squeeze_(dim=0)
    
    # Calculate means and stds for de-standardization process
    means_denorm = list(map(lambda x,y: -1*x/y, means, stds))
    stds_denorm = list(map(lambda y: 1/y, stds))
    
    # De-standardize tensor image
    invert_preprocess = transforms.Compose([
    transforms.Normalize(mean=means_denorm, std=stds_denorm),
    ])
      
    output_image = invert_preprocess(image).detach()
    
    # De-normalize tensor image
    output_image = output_image * 255
    
    # Add back the batch dimension
    output_image = output_image.unsqueeze_(0)
    
    return output_image