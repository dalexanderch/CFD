# CFD Upsampler

The goal of our tool is to upsample CFD simulations from a small grid to a larger grid in the hope of saving on computation time. 

## Requirements

You need the following packages : 
- tensorflow (make sure to also have tensorflow-gpu otherwise it will only use the cpu)
- keras (normally it should already be in the tensorflow packages but I had issues so better make sure it's there)
- Pillow (it's a library to manipulate images)
- Numpy (should be already installed in most environments)
- Sklearn (should be already installed in most environments)

## Training the network

For some reason, transferring files to google cloud from my local computer is extremely slow. Therefore, I'm using github as an intermediary. However, there is a limit on the number of files you can upload, therefore I compressed my data folder into a data.zip file that you need to unzip. 
In order to train the network you then run upsample.py. If the training is really slow, your gpu is not being used. Make sure that tensorflow-gpu is installed and that all required packages and drivers are up to date. Setting up an environment from scratch is pretty hard, so I strongly suggest using prebuilt images from your favorite cloud provider. 
I personally use the official tensorflow image from google cloud : https://cloud.google.com/deep-learning-vm/
An already trained network is already provided (upsample.h5).

