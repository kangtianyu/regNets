# Guide of Using RegNets
1. Introduction
RegNets is the neural network structure I showed in the paper. This github project is the code I used for testing algorithm and hyper-parameters in my paper. It is based on Java Neural Network Framework "Neuroph". I do some foundational level extentions and modifications, so the "Neuroph" include in this project may not work with other projects.
2. Project Setup
If you are using Eclipse, you can use eclipse to setup the project automaticly.
Otherwise, you need set the project working directory to regNets root directory, and include all file in src directory as the source file.
All data used in test programs are in the data directory. Because some data file is very large, not all raw data I used in my paper is in the data directory.
All hyper-parameters are configed in RegNetsInstance.java file, you can play with those hyper-parameters and inspect the results.
3. Main Test Program
    1. IndependentTest
    This is a test which using one dataset as the training set and the other as the testing set. It will creat severl threads with the same setting, compute simultaneously and output the statistics and average results.
    2. CrossValidationTest
    This is a test which do a cross validation on a single datasets. It will creat severl threads with different training set and testing set and compute them simultaneously. The output is the statistics and average results of the cross validation tests.
    3. BootStrapTest
    This is a Boot Strap test progress, which will output the most common high frequency genes. 