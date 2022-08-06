# Assessing the Robustness in Predictive Process Monitoring through Adversarial Attacks

Complementary code to reproduce the work of "Assessing the Robustness in Predictive Process Monitoring through Adversarial Attacks"


![Methodology_rev-1](https://user-images.githubusercontent.com/75080516/183251884-2b80c28a-fafb-4c7a-929c-59decbae9bbb.png)
_Figure: The robustness assessment framework that describes the different adversarial attacks, the method of application and how the evaluation is performed.
The notation A* indicates that both attack A1 and A2 can be used._

An overview of the files:

### Preprocessing files 

The preprocessing and hyperoptimalisation are derivative work based on the code provided by [Outcome-Oriented Predictive Process Monitoring: Review and Benchmark](https://github.com/irhete/predictive-monitoring-benchmark).
We would like to thank the authors for the high quality code that allowed to fastly reproduce the provided work.
- dataset_confs.py
- DatasetManager.py
- EncoderFactory.py

### Hyperoptimalisation of parameters
- Hyperoptimalisation_ML_Attack
- Hyperoptimalisation_DL_Attack (GC).ipynb

### Training of the Machine Learning Models
*Logistic Regression (LR), Logit Leaf Model (LLM), Generalized Logistic Rule Regression (GLRM), Random Forest (RF) and XGBoost (XGB)*
- Experiment_ML_Attack.py
- Experiment_ML_Attack_Test.py

### Training of the Deep Learning Models (with Google Colab)
*Long short-term memory neural networks (LSTM) and Convolutional Neural Network( CNN)*
- Experiment_DL_Attack (GC).py
- Experiment_ML_Attack_Test (GC).py

We acknowledgde the work provided by [Building Interpretable Models for Business Process Prediction using Shared and Specialised Attention Mechanisms](https://github.com/ZhipengHe/Shared-and-Specialised-Attention-based-Interpretable-Models) for their attention-based bidirectional LSTM architecture to create the long short-term neural networks with attention layers visualisations.

Finally, the folders contain additional figures and plots that have (not) been used in the paper.
