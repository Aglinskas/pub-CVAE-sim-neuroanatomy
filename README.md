# pub-CVAE-sim-neuroanatomy
Repository for the code and data accompanying the manuscript "Understanding heterogeneity in psychiatric disorders: a method for identifying subtypes and parsing comorbidity" by Aidas Aglinskas, Alicia Bergeron & Stefano Anzellotti. 

# Contents
* **specfile.txt** contains the parameters of the Anaconda enviroment

* **package-list.txt** lists the python packages and their versions used in the analyses

* **Results-latents/** Contains the latent-vectors data used in analyses and training-log data (loss, MSE, variance explained) generated during model training
* **Results/** Contains .npy files with stored results
* **Misc/** Contains miscellaneous files, such as the base atlas used to generate training data
* **Data/** Contains training and reconstruction data, as well as .csv recipes that parametrize training data
* **Code/**
  * **01-generate-data.ipynb** # Generates training data. can take in .csv recipe to recreate data used for model training
  * **02-train-CVAE-ensemble-baseline.ipynb** Train the baseline CVAE model
  * **03-train-VAE-ensemble-baseline.ipynb** Train the baseline VAE model
  * **04-train-CVAE-ensemble-N500.ipynb** Train the CVAE model using only N=500 subjects
  * **05-train-CVAE-ensemble-N200.ipynb** Train the CVAE model using only N=200 subjects
  * **06-train-CVAE-ensemble-8D.ipynb** Train the CVAE model with 8-dimensional latent-space
  * **07-train-CVAE-ensemble-2-subtypes.ipynb** Train the CVAE model using 2-subtype data
  * **08-train-VAE-ensemble-2-subtypes.ipynb** Train the VAE model using 2-subtype data
  * **09-train-CVAE-ensemble-3-subtypes.ipynb** Train the CVAE model using 3-subtype data
  * **10-train-CVAE-ensemble-5-subtypes.ipynb** Train the CVAE model using 5-subtype data
  * **11-train-CVAE-ensemble-2-subtypes-N2000.ipynb** Train the CVAE model using 2-subtype, N=2000 data
  * **12-train-CVAE-ensemble-3-subtypes-N2000.ipynb** Train the CVAE model using 3-subtype, N=2000 data
  * **13-train-CVAE-ensemble-5-subtypes-N2000.ipynb** Train the CVAE model using 5-subtype, N=2000 data
  * **14-train-CVAE-ensemble-comorbidity.ipynb** Train the CVAE model using comorbid data (ASD,ADHD & ASD+ADHD)
  * **15-calc-J-diff.ipynb** Calculate the ratio between shared and disorder-speciifc deformations (2.4)
  * **16-extract-repl-data.ipynb** Train the CVAE model using using 2-subtype, N=2000 data
  * **17-extract-CVAE-synth-twins.ipynb** Use a trained CVAE model to extract latent-representations using independent test data
  * **18-paper-analysis-correlation-analyses.ipynb** Statistical tests of correlations with ground truth measures
  * **19-paper-analysis-clustering-analyses.ipynb** Statistical tests of clustering accuracy, comparing model-inferred subtypes with ground truth subtypes
  * **20-paper-analysis-generalization.ipynb** Analyze generalization to independent data
  * **21-make-comorb-data.ipynb** Generate synthetic comorbidity training data
  * **22-COMORB-ensemble-init.ipynb** Train CVAE to disentagle comorbidity
  * **23-paper-analysis-comorbidity-disentagling.ipynb** Disentagling comorbidity analyses and plots
  * **CVAE_funcs.py** Helper functions
  * **initialize_CVAE-COMORB.py** Function that trains the comorbidity CVAE
  * **initialize_CVAE-dim8.py** Function that trains the 8-dimensional CVAE
  * **initialize_CVAE.py** Function that trains the CVAE
  * **initialize_VAE.py** Function that trains the VAE
  * **make_models.py** helper functions
  * **rsa_funcs.py** helper functions
  * **slurm-01-train-CVAE-ensemble.sh** SLURM job that trains the CVAE models
  * **slurm-02-train-VAE-ensemble.sh** SLURM job that trains the VAE models
  * **xx-model-plate-CVAE-8D.ipynb** Architecture details of the 8-dimensional CVAE model
  * **xx-model-plate-CVAE.ipynb** Architecture details of the CVAE model
  * **xx-model-plate-VAE.ipynb** Architecture details of the VAE model
