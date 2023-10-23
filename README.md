# EGFR

## Summary

Task: Classifying the potency value (pIC50) for novel compounds targeting Epidermal Growth Factor Receptor (EGFR) kinase. If -log(IC50) > 8, compund is considered active, otherwise inactive. Where, IC50 represents the compound/substance concentration required for 50% inhibition.

Baseline model: Logistic Regression
Features: morgan2 finger prints

Designed model: Recurrent Neural Network
Features: 

```python

from ml import RNNModel
loaded_model = RNNModel.load('saved_results/RNN_weights.h5', 'saved_results/RNN_config.json')
loaded_model.compile()
loss, accuracy, auc, f1 = loaded_model.evaluate(X_test_input, y_test)

```