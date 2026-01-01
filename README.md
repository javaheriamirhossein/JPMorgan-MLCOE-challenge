# Discrete Choice Models for Credit Card Offers

This repository contains the code and report for the JPMorgan MLCOE Challenge on *Discrete Choice Models for Credit Card Offers*. It is built on top of `choice-learn` framework and TensorFlow. It includes synthetic experiments, real-data experiments, baseline comparisons, and unit tests. 



## Repository Structure

### Notebooks

- **`Synthetic-data-generation.ipynb`** (answer (b))
  - Generates synthetic datasets for DeepHalo in the *featureless* setting.  

- **`Synthetic-experiment.ipynb`**  (answers (b) and (c))
  - Trains and evaluates DeepHalo on the synthetic *featureless* data.  
  - Contains the synthetic experiment setup and metrics.

- **`Real-experiment-ModeCanada.ipynb`**  (answer (c))
  - Loads the ModeCanada dataset via `choice-learn`.  
  - Trains and evaluates *featured* DeepHalo on this dataset.  

- **`Real-experiment-SwissMetro.ipynb`**  (answer (c))
  - Loads the SwissMetro dataset via `choice-learn`.  
  - Trains and evaluates *featured* DeepHalo on this dataset.  

- **`Comparison-with-baseline.ipynb`**  (answers (f)).
  - Compares DeepHalo against:  
    - Classical discrete choice models (e.g., MNL / conditional logit, nested/halo variants as implemented in `choice-learn`)  
    - Machine-learning baselines (e.g., SVM, gradient boosting methods) 

### DeepHalo Core Implementation

The `DeepHalo/` folder contains the core model definitions:

- **`DeepHalo_choice_learn.py`**  
  - Defines the `DeepHaloChoiceModel` class integrating DeepHalo into the `choice-learn` API.  


- **`Featured_DeepHalo.py`**  
  - Implements the *featured* DeepHalo core network for inputs with item features. 


- **`Featureless_DeepHalo.py`**  
  - Implements the *featureless* DeepHalo core model.

### Tests

The `tests/` folder contains:

- **`DeepHalo_Tests.py`**  
  - Defines the `TestDeepHaloChoiceModel` class for unit tests  

---

