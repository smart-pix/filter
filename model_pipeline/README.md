train.ipynb -> Adapted from GDG's notebook. Used to process data, train the algorithm, and save the model file + csv files. For the 16x16 pipeline (i.e. using 16x16 pixel array simulation datasets, matching with the 16x16 pixel array of CMSPIX28), the train and test csv data and labels files can be loaded directly. These are the same files that goes into the ipynb files of the filter algorithm and repo.
The capability to add gaussian noise to the data also has been added (can be utilized using the `inject_noise` bool).

prepare_weights.py -> Uses the output csv files (b5,c5,b2,c2) from above notebook and produces the final csv file (b5_w5_b2_w2_pixel_bin) that will be sent to the ASIC.

run.ipynb -> produces compouts needed for ASIC DNN data-taking. The "useYProfilePulsing" variable can be set to False, if you want to generate compouts using 16x16 simulation datasets. The functions in utils.py have been updated to take care of all subsequent measures to account for the the new pixel array size (default was previous 21x13).

efficiency.ipynb -> analyzes DNN + model outputs and produces performance plots/results.


[`Depreciated!`] pre_processing.ipynb -> Processes parqueted simulation datasets to quantized versions and produces csv files needed for training and evaluating the NN algorithm.

[`Depreciated!`] SimpleNN_Qkeras_DNN.ipynb -> train algorithm on processed data and save model file + csv file to program weights and biases on chip.