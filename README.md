# Modifications for ECNP
Modifications made to this code for use in the paper "Predicting Enzyme-Compound Interactions for Enzyme-Catalysed Reactions".
## Pretrain.py
We add the ability to use our ECMap dataset and to run 10 seeds using a predetermined 10-fold cross validation train/test split.

## Finetune_bec.py
We again add the ability to use our 10-fold cross-validation train/test split. 
We also save the predictions in a format easy to use with the ECNP repository.

# Original README:
# BEC-Pred
Data and code of enzymatic reaction prediction model BEC-Pred.

## Installation
Use python version 3.9 and install packages from `requirements.txt`.
Once those packages are installed run:
`pip install rxnfp --no-deps`

Make sure to unzip `mlm_train.zip` and move `mlm_train.txt` to `data/`

## Trained Models

The trained models are stored in `BECPred/model`.
