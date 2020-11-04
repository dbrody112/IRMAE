# IRMAE
This is an implementation of the Implicit Rank-Minimizing Autoencoder (which can be found at the following site: https://arxiv.org/pdf/2010.00679.pdf).
Specifically, I replicated figure 2 and figure 3 using a synthetic shape dataset of shapes and circles with various centers, colors, and sizes, giving an intrinsic dimensionality of 7. The autoencoder was trying to achieve that same dimensionality in their output to show how the model decreases dimensionality/rank to the lowest possible dimensionality/rank (hence the singular value vs. matrix rank graph). Another function of the autoencoder is the regularization effect and superior representation which can be shown in the first and second figure respectively. Unfortunately in my implementation IMRAE(l=4) overfit. I suspect this may be because I shrinked the squares in the dataset to make the representation easier to learn and interpolation clearer. 

## Matrix Rank vs. Singular Values
![image1](https://user-images.githubusercontent.com/59486373/98158559-170d8380-1ea9-11eb-8b18-ce316ee7b90e.png)

## Linear Interpolation
![image2](https://user-images.githubusercontent.com/59486373/98158963-c6e2f100-1ea9-11eb-8ece-05bca831e7ce.png)


