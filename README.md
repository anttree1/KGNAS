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
### para_ab.py & para_w.py & para_w12.py: Parameter sensitivity experiment.
### para_sw.py & test_SW: The small-worldness experiment.
### pso_model.py: The main program for architecture search.
### test_KDE.py: Experiments with KDE graphs.
### test_SC.py: Difference matrix experiment.
### test_model.py: Test architecture.
### utils.py: Utility classes.


## Operating Guide
### In model_search.py, you can modify the model class.
### In fitness_model.py, you can modify the fitness evaluation strategy.
### In pso_model.py, set the parameters you need and then search.
### In test_model, Enter the search results and evaluate the performance. 

## Dataset acquisition
### ABIDE I: https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html
### ADHD-200: https://fcon_1000.projects.nitrc.org/indi/adhd200/
### AAL & CC200 atlas: http://preprocessed-connectomes-project.org/abide/Pipelines.html
