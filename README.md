Use the main script for parametric models (EXP_BernoulliGamma.py).

For new configurations, modify the CFG and CFGD files
- CFG_* for the model architecture and training options (example in CFG_BernoulliGammaUNET.py).
- CFGD_* for the data transformations (example in CFGD_StandardTransforms4Prediction_incX1D.py).

The script writes some stats after training and plots some maps automatically from the validation period for a quickcheck.

It also hashes the folders so there is no overwrite. 

All the data necessary to reproduce the results in Legasa et al. (2026) can be found on Zenodo: https://doi.org/10.5281/zenodo.20610814.
