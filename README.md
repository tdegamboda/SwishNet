# SwishNet


The following repository consists of all the source code used in order to produce the results reported in Report.pdf. For the purpose of presenting my code and results I have formatted all the answers into IPython notebooks however, they are maintained in such a way that can be easily exported into an executable Python script.

This repository contains the following directories and subdirectories:

*  *Report.pdf* contains all the written answers for the task

*   **data**
    - For the purpose of compressing this repo, I have deleted all datasets however in order to run any code, please download the MUSAN and GTZAN corpus and name the directories *musan* and *music_speech* respectively.
    - This directory would contain all raw and derived dataset

*   **docs**
    - This directory contains the task, along with the SwishNet paper

    - **plots**

        - This directory contains all the plots obtained of the history over training and fine-tuning

*   **model_params**
    - This directory contains all the saved model parameters in the form of h5 files that can be easily loaded into the model on evaluation. NOTE: trained model parameters had to be deleted for memory requirements
*   **Q1**
    - *my_implementation.ipynb* contains the Tensorflow implementation of SwishNet
    - *keras_implementation.ipynb* contains the Keras implementation of SwishNet taken directly from https://github.com/i7p9h9/swishnet
*   **Q2**
    - *EDA_audio.ipynb* contains an exploration of the libraries used for training i.e. Librosa, python_speech_features, etc.
    - *preprocessing.ipynb* contains the preprocessing pipeline used to preprocess the MUSAN corpus
*   **Q3**
    - *training.ipynb* contains the training script 
*   **Q4**
    - *gtzan_preprocessing.ipynb* contains the preprocessing pipeline used to preprocess the GTZAN dataset
    - *gtzan_evaluation.ipynb* contains the evaluation script
*   **utils**
    - This directory is formatted as a Python module in order to easily import reused code
    - *SGDRScheduler.py* contains the source code for application of warm restarts and cosine annealing
    - *SwishNet.py* contains the source code for my implementation of the SwishNet in Tensorflow for easy imports during training and evaluation