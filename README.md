# MiSiCNet
MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing

MiSiCNet is a deep learning-based technique for blind hyperspectral unmixing. MiSiCNet copes with higly mixed scenarios and complex datasets with no pure pixels. Unlike all the deep learning-based unmixing methods proposed in the literature, the proposed convolutional encoder-decoder architecture incorporates spatial and geometrical information of the hyperspectral data, in addition to the spectral information. The spatial information is incorporated using convolutional filters and implicitly applying a prior on the abundances. The geometrical information is exploited by incorporating a minimum simplex volume penalty term in the loss function for the endmember extraction. This term is beneficial when there are no pure material pixels in the data, which is often the case in real-world applications. We generated simulated datasets, where we consider two different no-pure pixel scenarios. In the first scenario, there are no pure pixels but at least two pixels on each facet of the data simplex (i.e., mixtures of 2 pure materials). The second scenario is a complex case with no pure pixels and only one pixel on each facet of the data simplex.

To run the code change the path to the correct directory. You need instal the dependencies i. e., torch, numpy, and matplotlib (for ploting).   


![GifMaker_20220125172826580](https://user-images.githubusercontent.com/61419984/151018568-b380b6da-d782-4079-aff6-81af290cf5bb.gif)
![GifMaker_20220125173129571](https://user-images.githubusercontent.com/61419984/151018587-409377a3-8f65-4866-a302-a7d3df9a6c7e.gif)
