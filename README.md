# MiSiCNet
MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing

If you use this code please cite the following paper: B. Rasti, B. Koirala, P. Scheunders and J. Chanussot, "MiSiCNet: Minimum Simplex Convolutional Network for Deep Hyperspectral Unmixing," in IEEE Transactions on Geoscience and Remote Sensing, doi: 10.1109/TGRS.2022.3146904.

MiSiCNet is a deep learning-based technique for blind hyperspectral unmixing. MiSiCNet copes with highly mixed scenarios and complex datasets with no pure pixels. Unlike all the deep learning-based unmixing methods proposed in the literature, the proposed convolutional encoder-decoder architecture incorporates spatial and geometrical information of the hyperspectral data, in addition to the spectral information. The spatial information is incorporated using convolutional filters and implicitly applying a prior on the abundances. The geometrical information is exploited by incorporating a minimum simplex volume penalty term in the loss function for the endmember extraction. This term is beneficial when there are no pure material pixels in the data, which is often the case in real-world applications. We generated simulated datasets, where we considered two different no-pure pixel scenarios. There are no pure pixels in the first scenario but at least two pixels on each facet of the data simplex (i.e., mixtures of 2 pure materials). The second scenario is a complex case with no pure pixels and only one pixel on each facet of the data simplex.

All the datasets used in the manuscript are uploaded here except WDC which you can download here here https://www.dropbox.com/s/tj65r3c027nbpth/Y_clean.mat?dl=0

To run the code, change the path to the correct directory. You need to install the dependencies, i. e., torch, numpy, and matplotlib (for plotting), scipy.io, tqdm (for simulated datasets). You need to select the value of lambda (we use 100 for real datasets but for your dataset might be diiferent) and rmax (the number of endmembers). Here are the results of training over iterations. The gifs show how the endmembers and abundances converge over the iterations for a highly mixed scenario and a noisy simulated dataset (20 dB).

To speed up the processing time you can make the PLOT flag False or set show_every=1000.


<img src="https://user-images.githubusercontent.com/61419984/151823999-119f1597-7d44-4b78-8ad7-190b9540d8c7.png" width="400" height="200"><img src="https://user-images.githubusercontent.com/61419984/151020437-d22dc981-2a46-44de-9ef9-a3dd09873b14.gif" width="400" height="200"><img src="https://user-images.githubusercontent.com/61419984/151022010-822e93ab-65b9-4376-b168-c626b2a253bb.gif" width="400" height="200">




