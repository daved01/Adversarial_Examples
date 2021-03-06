# Adversarial Examples

The goal of this repository is to explore how adversarial examples work. For that common attack methods are implemented and their steps explained. Detailed results as well as background information can be found [here](https://daved01.github.io/Adversarial_Examples_GANs/).


**Note**: In order to use this repository, the data has to be downloaded from [Kaggle](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set?select=images).


As dataset 1000 examples from [NIPS 2017: Adversarial Learning Development Set](https://www.kaggle.com/google-brain/nips-2017-adversarial-learning-development-set#categories.csv) are used. These images are similar to the ImageNet dataset.


----------------
## Structure

In the repository there is a notebook for each attack method. The notebook 

- `00_Helper-Functions.ipynb`

contains helper functions which are required in all other notebooks. Copies of these can be found in `modules/helper.py` and `modules/dataset.py`.

The notebook

- `01_Data_Exploration.ipynb`

contains a data exploration and predictions on the clean data.

The implmenetation of the attack methods can be found in the folder `modules`. In the notebooks

- `02_Fast-Gradient-Sign-Method.ipynb`

- `03_Basic-Iterative-Method.ipynb`

- `04_Iterative-Least-Likely-Class-Method.ipynb`

- `05_DeepFool.ipynb`

the attacks are explored.
