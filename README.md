## Description of the model file

* DGSR is an implementation of the SAC-GNN model
* single_gc_mc is an implementation of the GC-MC model
* single_mgnn is an implementation of the MGNN model
* single_ngcf is an implementation of the NGCF model
* The code of ISRMM is available at https://github.com/HduDBSI/ISRMM
* The code of LSRCC is available at Https://github.com/HduDBSI/LSRCC
  
## Description of the SAC-GNN

### Description of the main framework

1. The main model structure is implemented in file MAIGNet.py </br>
2. ALCGnet is invocated by MAIGNet，to achieve dual-graph structure learning</br>
3. Attention and related layers are implemented in MAIGNet
4. Input data in ALCGNet to use LM
5. The training process is implemented in cold_main.py, including data preprocessing, similarity pairing, training and test set reading, metrics calling, and training process. 


### Description of experiment

1. In order to conduct Parameters  experiment, hyperparameter is setted in parser_init.py
2. cold_main_no,cold_main_no_pop，MAIGNet_no_att,MAIG_no_gate are related to ablation experiments, read the notes for the specific experiments targeted
3. metric, data_read and model_test are file related to metrics calculation, file reading and model testing
4. pop_test file is related to popularity baselines experiment.
5. visual plotting and tsne dimensionality methods are included in graph_calculate file.

### Description of data

Files ending in txt are processed sequences or vectors. The details are as follows:</br>

1. The file at the end of vec is the persistent vector obtained after description has been processed by lm, Please call it via the method provided in lm_features.
2. The end of the description is the service description file after splitting the words.
3. Files with percentage names are the split test set and training set, corresponding by suffix number.


## Description on specific versions of libraries

The libraries SAC-GNN used are as follows:</br>

1. PyTroch 2.1.3
2. gensim 4.3.2
3. bert_serving 1.10.0
4. numpy 2.2.5
5. scikit-learn 1.7.0rc1