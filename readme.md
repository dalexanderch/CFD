# CFD Upsampler

The goal of our tool is to upsample CFD simulations from a small grid to a larger grid in the hope of saving on computation time. My model is based on the SRCNN architecture. You can find the paper here : 
https://arxiv.org/pdf/1501.00092.pdf
The results are not very impressive yet. It's not clear that the original idea was viable. However it might be the case that I have a problem somewhere in my process that I have not been able to spot. 

# Todo List 

- Regenrate the data properly.
- Finish training on mnist
- Train on imagenet
- Try different loss functions
- Show the results in a clean presentation

# Results on MNIST

We use mean square error as a loss function.
- Using classical bilinear interpolation : 0.0252
- Using learned model : 0.002

0.0252/0.002 = 12.6, therefore the learned model is about 13 times better than the classical interpolation method. 

# Results on Cifar10

We use mean square error as a loss function
- Using classical bilinear interpolation : 0.063
- Using learned model : 0.0050

0.063/0.0050 = 12.6, therefore the learned model is about 13 times better than the classical interpolation method. 

# Results on Cifar100

We use mean square error as a loss function
- Using classical bilinear interpolation : 0.0062
- Using learned model : 0.0055

0.063/0.0050 = 1.24, therefore the learned model is about 24% times better than the classical interpolation method. 

## Requirements

You need the following packages : 
- tensorflow (make sure to also have tensorflow-gpu otherwise it will only use the cpu)
- keras (normally it should already be in the tensorflow packages but I had issues so better make sure it's there)
- Pillow (it's a library to manipulate images)
- Numpy (should be already installed in most environments)
- Sklearn (should be already installed in most environments)

## Training the network

- For some reason, transferring files to google cloud from my local computer is extremely slow. Therefore, I'm using github as an intermediary. However, there is a limit on the number of files you can upload, therefore I compressed my data folder into a data.zip file that you need to unzip. 

- In order to train the network you then run upsample.py. If the training is really slow, your gpu is not being used. Make sure that tensorflow-gpu is installed and that all required packages and drivers are up to date. Setting up an environment from scratch is pretty hard, so I strongly suggest using prebuilt images from your favorite cloud provider. 
I personally use the official tensorflow image from google cloud : https://cloud.google.com/deep-learning-vm/. The parameters are the following : 
	- Epochs : The script expects an integer. One Epoch is when an ENTIRE dataset is passed forward and backward through the neural network only ONCE.
	- Batch size : The script expects an inteher. The batch size is the total number of training examples present in a single batch. 

- In order to generate custom data, you need to run generate_data.py which is located in /CFD/palabos-v2.0r0/examples/showCases/cylinder2d. The parameters are the following :
	- Number of batches (there will be an equal number of small batches and big batches. Each batch corresponds to a 2 seconds simulation with a given Reynolds number, one with a small grid size, the other with a bigger grid size so we get parallel data)
	- Upsampling rate (an integer to decide how large the big grid is going to be relative to the small grid. The trained model use was trained with a value of 2)
	- Mean Reynolds number. In order to generate the data, we will sample our Reynolds numbers with the provided mean and the provided variance (see below).
	- Variance : as explained above
	- N : an integer used to build the grid. 
	- lx : N*lx corresponds to the dimensionality of the x-axis.
	- ly: N*ly corresponds to the dimensioanlity of the y-axis. 

An already trained network is already provided (upsample.h5).

## Running a simulation

In order to run a simulation you need to use the file run.py in /CFD/palabos-v2.0r0/examples/showCases/cylinder2d. The parameters are the following : 

- N : like above
- Re : like above
- lx : like above
- ly : like above

## Upsampling a simulation 

In order to upsample a simulation, you need to use the file predict.py in CFD/ 
The parameters are :
 
- pathSmall = path of small grid images (the already trained network was trained for small images with size 100x100)
- pathBig = path to save upsampled images


