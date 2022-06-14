# MLP-NeuralNet-From-Scratch

This project was made for the course "Neural Networks & Deep Learning". Its task was to build a Multi-Layer Perceptron from scratch, train it for image classification and compare the results to K-Nearest Neighbors and Nearest Centroid models.

The MLP is capable of **batch** and **stochastic** backpropagation training, each layer of neurons can have either a **Sigmoid**, a **LReLU** or a **Softmax** activation function. The **learning rate** can be set to constant or decaying, where it's divided in half every *n* steps.

The model was trained and evaluated in the [MNIST Fashion](https://github.com/zalandoresearch/fashion-mnist) dataset.
After some hyper-parameter tuning, the model with the best parameters surpassed both KNN and NC in accuracy and predict time with **87.9% accuracy**. 

A more detailed report about the MLP code and the comparison results is available here [Neural_Networks_MLP.pdf](https://github.com/Kouspan/MLP-NeuralNet-From-Scratch/files/8900560/Neural_Networks_MLP.pdf) (in greek).

#### Comments for running the MLP_Testing Notebook
With the process_data(...) function the dataset is loaded from the ubyte files and processed with PCA and OneHotEncoder.
For ease of use, the processed data are saved in .npy form in the './Data/Processed/' folder. 
Either create the above folder or change the save path.  
