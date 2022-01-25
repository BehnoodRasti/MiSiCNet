# MiSiCNet
MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing

MiSiCNet is a deep learning-based technique for blind hyperspectral unmixing. MiSiCNet copes with higly mixed scenarios and complex datasets with no pure pixels. Unlike all the deep learning-based unmixing methods proposed in the literature, the proposed convolutional encoder-decoder architecture incorporates spatial and geometrical information of the hyperspectral data, in addition to the spectral information. The spatial information is incorporated using convolutional filters and implicitly applying a prior on the abundances. The geometrical information is exploited by incorporating a minimum simplex volume penalty term in the loss function for the endmember extraction. This term is beneficial when there are no pure material pixels in the data, which is often the case in real-world applications. We generated simulated datasets, where we consider two different no-pure pixel scenarios. In the first scenario, there are no pure pixels but at least two pixels on each facet of the data simplex (i.e., mixtures of 2 pure materials). The second scenario is a complex case with no pure pixels and only one pixel on each facet of the data simplex.

To run the code change the path to the correct directory. You need instal the dependencies i. e., torch, numpy, and matplotlib (for ploting).   



https://user-images.githubusercontent.com/61419984/151016911-1fd88441-926b-4494-8c06-f58ede4b9364.mp4



https://user-images.githubusercontent.com/61419984/151016925-819365a8-b78b-4853-aecd-7f1e6a76d14f.mp4

