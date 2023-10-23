# EGFR

## Quick Start

```bash
git clone https://github.com/mehradans92/EGFR.git && cd EGFR
conda create --name valance python=3.9
conda activate valance
pip install - r requirements.txt
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

Task: Classifying the potency value (pIC50) for novel compounds targeting Epidermal Growth Factor Receptor (EGFR) kinase. If -log(IC50) > 8, compund is considered active, otherwise inactive. Where, IC50 represents the compound/substance concentration required for 50% inhibition. Dataset is imalanced (negative:positive ~4.2). Adding `bias regularizer` to the final layer, or over sampling the minority class did not change the performance much.

Baseline model: Logistic Regression

Designed model: Recurrent Neural Network

See [this notebook](main.ipynb) for more deatails.

### Results Summary

#### 5-fold Cross Validation Performance Benchmarking
See [this notebook](CV.ipynb) for more deatails.

| Model                             | Features                       | Accuracy    | AUROC           | F1 score     |
|-----------------------------------|--------------------------------|-------------|---------------|--------------  |
| SVM (baseline)                    | ECFP-4                         | 0.81 ± 0.01 | 0.82 ± 0.02   | 0.59 ± 0.01    |
| RNN (Bi-LSTM)                     | Smiles, ECFP-4, Mordred        | 0.86 ± 0.01 | 0.89 ± 0.01   | 0.10 ± 0.03    |

#### 5-fold Cross Validation Ablational Analysis

| Model                             | Features                       | Accuracy    | AUROC           | F1 score     |
|-----------------------------------|--------------------------------|-------------|---------------|--------------|
| RNN (Bi-LSTM)                     | Smiles                         | 0.81 ± 0.01 | 0.80 ± 0.01  | 0.30 ± 0.16   |
| RNN (Bi-LSTM)                     | Smiles, ECFP-4                 | 0.85 ± 0.01 | 0.89 ± 0.01  | 0.59 ± 0.03   |
| RNN (Bi-LSTM)                     | Smiles, ECFP-4, Mordred        | 0.86 ± 0.01 | 0.89 ± 0.01  | 0.60 ± 0.02   |


## Methods

### Model Architecture
<img src="https://github.com/mehradans92/EGFR/assets/51170839/019aacc3-fcbb-4a24-8e96-7fd17c064a46" alt="model_architecture" width="1100"/>

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

Variant Mordred descriptors were selected via recursive feature elimination (RFE) cross-validation using a Random forests classifier.
This was done by adjusting the number of features from 10 to 300 with increments of 10 and finding the the number of optimal features that results the highest test AUROC by the classifier.
See [this notebook](mordred_feature_selection.ipynb) for more deatails.

<img src="https://github.com/mehradans92/EGFR/assets/51170839/30a02b4b-31b0-4369-9d2d-3d38cebf74af" alt="Pearson" width="1100"/>

