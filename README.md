# KGNAS
## Python & Library Version
### python 3.9.2
### pytorch 1.8.0
### cuda 11.3


## Code Description
### fitness_model.py: Individual performance prediction.
### material.py: Store the search space content.
### model_search.py: Model class.
### operations.py: All candidate operation.
### 


## Operating Guide
### In pso_model.py, set the parameters you need in args. 
### In test_model, set the classifier you need to evaluate the individual fitness. 
### The list_ind variable of test_ADA.py contains the vector of the data augmentation strategy found, and the model is set to the classifier used to select the test.
### In material.py, set up other possible sets of brain regions.

## Dataset acquisition
### ABIDE I: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
### ADHD-200: https://fcon_1000.projects.nitrc.org/indi/adhd200/
### AAL & CC200 atlas: http://preprocessed-connectomes-project.org/abide/Pipelines.html
