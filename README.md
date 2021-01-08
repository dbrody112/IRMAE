# IRMAE
This is an implementation of the Implicit Rank-Minimizing Autoencoder (which can be found at the following site: https://arxiv.org/pdf/2010.00679.pdf).
Specifically, I replicated figure 2 and figure 3 using a synthetic shape dataset of shapes and circles with various centers, colors, and sizes, giving an intrinsic dimensionality of 7. The autoencoder was trying to achieve that same dimensionality in their output to show how the model decreases dimensionality/rank to the lowest possible dimensionality/rank (hence the singular value rank vs. singular value graph). Another function of the autoencoder is the regularization effect and superior representation which can be shown in the first and second figure respectively. Unfortunately in my implementation IMRAE(l=4) overfit. I suspect this may be because I shrinked the squares in the dataset to make the representation easier to learn and interpolation clearer. 

## Singular Value Rank vs. Singular Values
This graph was made by taking the latent covariance (covariance of space between encoder and decoder) and finding the singular values through SVD.

![image1](https://user-images.githubusercontent.com/59486373/98158559-170d8380-1ea9-11eb-8b18-ce316ee7b90e.png)

## Linear Interpolation
This linear interpolation was created by a custom function that can be found in main.py
![image2](https://user-images.githubusercontent.com/59486373/98158963-c6e2f100-1ea9-11eb-8ece-05bca831e7ce.png)

<i>more info can be found in irmae_paper.pdf as well as the raw code (without organization)</i>


