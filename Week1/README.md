print(score)
[0.02931464004876325, 0.9912]

-------------------------------------------------------------------------------------------------------------------------------------------
                                                        ASSIGNMENT 1
-------------------------------------------------------------------------------------------------------------------------------------------

CONVOLUTION:
--
   To extract Features from an image.
   Process of scanning an image with a filter/kernel matrix using elementwise matrix multiplication to get single pixel as output. The convolution process results in Feature map.

KERNEL/FILTER:
--
  Represented in Matrix of '1's and '0's. Smaller than the input image matrix.
  Each filter captures different characteristics of the image which ultimately helps to recognize the features.
  Example : Edges - Horizontal, vertical, Diagonal edges.

EPOCHS:
--
   A single pass through the full training set(Forward and backward pass)

1x1 CONVOLUTION:
--
   Dimensionality reduction for efficient computations(depth-wise)
   If the input has 16 channels, the 1 x 1 convolution will embed these channels (features) into a single channel.

3X3 CONVOLUTION:
--
   Kernel that helps in extracting feature of the image matrix given.
   Odd indexed kernel for maintaining the symmetry.
   Higher indexed kernel such as 7x7 or 5x5 can be reduced to 3x3, increases the depth(layer) instead of increasing the width.
   Reducing feature map by a dimension of 2x2.
   
FEATURE MAPS:
--
  Result of applying filters to the input image.The number of filters (kernel) the input will result in same amount of feature maps.

ACTIVATION FUNCTION:
--

RECEPTIVE FIELD:
--

