# EGFR

## Quick Start

```bash
git clone https://github.com/mehradans92/EGFR.git && cd EGFR
conda create --name EGFR python=3.9
conda activate EGFR
pip install -r requirements.txt
```


```python
from utils import get_features
from ml import RNNModel

test_smiles = 'C=CC(=O)Nc1ccc2ncnc(Nc3cc(F)c(Cl)c(Cl)c3)c2c1'
X_test_input = get_features(test_smiles)

loaded_model = RNNModel.load('saved_results/RNN_weights.h5', 'saved_results/RNN_config.json')
loaded_model.compile()
EGFR_potency_probability = loaded_model.predict(X_test_input)

```

## Summary

Task: Classifying the potency value (pIC50) for novel compounds targeting Epidermal Growth Factor Receptor (EGFR) kinase. If -log(IC50) > 8, compound is considered active (y=1), otherwise inactive (y=0). Where IC50 represents the compound/substance concentration required for 50% inhibition. The [dataset](https://raw.githubusercontent.com/volkamerlab/teachopencadd/master/teachopencadd/talktorials/T002_compound_adme/data/EGFR_compounds_lipinski.csv) is imbalanced (negative: positive  ~4.2), and total number of compounds is 4635. Adding `bias regularizer` to the final layer, or over sampling the minority class did not change the performance much.

Baseline model: Support Vector Machine

Designed model: Recurrent Neural Network with Bi-directional LSTMs units

See [this notebook](main.ipynb) for more details.

### Results Summary

#### 5-fold Cross Validation Performance Benchmarking
See [this notebook](CV.ipynb) for more details.

| Model                             | Features                       | Accuracy    | AUROC           | F1 score     |
|-----------------------------------|--------------------------------|-------------|---------------|--------------  |
| SVM (baseline)                    | ECFP-4                         | 0.81 ± 0.01 | 0.82 ± 0.02   | 0.59 ± 0.01    |
| RNN (Bi-LSTM)                     | Smiles, ECFP-4, Mordred        | 0.86 ± 0.01 | 0.89 ± 0.01   | 0.60 ± 0.02    |

#### 5-fold Cross Validation Ablation Study

| Model                             | Features                       | Accuracy    | AUROC           | F1 score     |
|-----------------------------------|--------------------------------|-------------|---------------|--------------|
| RNN (Bi-LSTM)                     | Smiles                         | 0.81 ± 0.01 | 0.80 ± 0.01  | 0.30 ± 0.16   |
| RNN (Bi-LSTM)                     | Smiles, ECFP-4                 | 0.85 ± 0.01 | 0.89 ± 0.01  | 0.59 ± 0.03   |
| RNN (Bi-LSTM)                     | Smiles, ECFP-4, Mordred        | 0.86 ± 0.01 | 0.89 ± 0.01  | 0.60 ± 0.02   |


## Methods

### Model Architecture
<img src="https://github.com/mehradans92/EGFR/assets/51170839/24668171-8a33-4c1a-88f1-bcb7e3e98964" alt="model_architecture" width="1100"/>

```
Model: "Bi-LSTM"
__________________________________________________________________________________________________
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 87)]                 0         []                            
                                                                                                  
 embedding (Embedding)       (None, 87, 8)                472       ['input_1[0][0]']             
                                                                                                  
 bidirectional (Bidirection  (None, 87, 32)               3200      ['embedding[0][0]']           
 al)                                                                                              
                                                                                                  
 bidirectional_1 (Bidirecti  (None, 32)                   6272      ['bidirectional[0][0]']       
 onal)                                                                                            
                                                                                                  
 layer_normalization (Layer  (None, 32)                   64        ['bidirectional_1[0][0]']     
 Normalization)                                                                                   
                                                                                                  
 dropout (Dropout)           (None, 32)                   0         ['layer_normalization[0][0]'] 
                                                                                                  
 input_2 (InputLayer)        [(None, 1024)]               0         []                            
                                                                                                  
 concatenate (Concatenate)   (None, 1056)                 0         ['dropout[0][0]',             
                                                                     'input_2[0][0]']             
                                                                                                  
 dense (Dense)               (None, 32)                   33824     ['concatenate[0][0]']         
                                                                                                  
 layer_normalization_1 (Lay  (None, 32)                   64        ['dense[0][0]']               
 erNormalization)                                                                                 
                                                                                                  
 dropout_1 (Dropout)         (None, 32)                   0         ['layer_normalization_1[0][0]'
                                                                    ]                             
                                                                                                  
 input_3 (InputLayer)        [(None, 60)]                 0         []                            
                                                                                                  
 concatenate_1 (Concatenate  (None, 92)                   0         ['dropout_1[0][0]',           
 )                                                                   'input_3[0][0]']             
                                                                                                  
 dense_1 (Dense)             (None, 8)                    744       ['concatenate_1[0][0]']       
                                                                                                  
 layer_normalization_2 (Lay  (None, 8)                    16        ['dense_1[0][0]']             
 erNormalization)                                                                                 
                                                                                                  
 dropout_2 (Dropout)         (None, 8)                    0         ['layer_normalization_2[0][0]'
                                                                    ]                             
                                                                                                  
 dense_2 (Dense)             (None, 1)                    9         ['dropout_2[0][0]']           
                                                                                                  
==================================================================================================
Total params: 44665 (174.47 KB)
Trainable params: 44665 (174.47 KB)
Non-trainable params: 0 (0.00 Byte)
__________________________________________________________________________________________________
```

### Mordred Feature Selection

Variant Mordred descriptors were selected via recursive feature elimination (RFE) cross-validation using a random forests classifier.
By none-variant, I mean descriptors that had zero values or zero variance for all compounds were removed.
This RFE was done by adjusting the number of features from 10 to 300 with increments of 10, and finding the number of optimal features that result the highest test AUROC by the classifier.
See [this notebook](mordred_feature_selection.ipynb) for more details.

<img src="https://github.com/mehradans92/EGFR/assets/51170839/0eeb9a05-1f2f-4e3d-ba5b-3873b3cbbe58" alt="Pearson" width="1100"/>

Test AUROC had a non-monotonic behavior when the number of features were increased. Total number of 60 mordred features were selected and their indices are stored [here](https://github.com/mehradans92/EGFR/blob/main/saved_results/non_zero_std_cols_mordred_indices.json).

![RFE_Mordred_features](https://github.com/mehradans92/EGFR/assets/51170839/5a825b92-51c7-4dd2-a75b-9be914e20953)
