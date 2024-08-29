# CCVAN

CCVAN: A Conditional Molecular Generation Model based on conditional VAE and Wasserstein GAN

1. Conditional Property Generation

First, you need to prepare a file that contains the SMILES and properties of all molecules, with each line representing a molecule’s SMILES and its properties, saved as a CSV file.

Then, use the following command to train the model and generate a custom number of molecules with the desired target properties:

python CCVAN-prop.py
The results will be saved in a file named generate.csv in the current directory.

2. Unconditional Molecular Generation

For this task, you only need to prepare a file containing the SMILES of all molecules, also saved with a CSV suffix.

Then, use the following command to train the model and generate a custom number of molecules:

python CCVAN-noprop.py
The results will be saved in a file named generate.csv in the current directory.

3. Result Visualization

Use the following command to generate all the figures from the paper:

python visualize.py
