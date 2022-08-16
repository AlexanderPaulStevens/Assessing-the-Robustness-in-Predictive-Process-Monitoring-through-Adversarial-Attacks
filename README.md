# Assessing the Robustness in Predictive Process Monitoring through Adversarial Attacks

Complementary code to reproduce the work of "Assessing the Robustness in Predictive Process Monitoring through Adversarial Attacks"


![Methodology_rev-1](https://user-images.githubusercontent.com/75080516/183251884-2b80c28a-fafb-4c7a-929c-59decbae9bbb.png)
_This figure contains the robustness assessment framework as introduced in the paper. The framework describes the different adversarial attacks, the method of application and how the evaluation is performed._

An overview of the folders and files is given below. Note that the hyperoptimalisation and experiments for the deep learning models are performed with the use of [Google Colab](https://colab.research.google.com/?utm_source=scs-index). The authors are thankful for this easy-to-use jupyter notebook that provides computing on a GPU.

## Files

### Preprocessing files 

The preprocessing and hyperoptimalisation are derivative work based on the code provided by [Outcome-Oriented Predictive Process Monitoring: Review and Benchmark](https://github.com/irhete/predictive-monitoring-benchmark).
We would like to thank the authors for the high quality code that allowed to fastly reproduce the provided work.
- dataset_confs.py
- DatasetManager.py
- EncoderFactory.py

### Hyperoptimalisation of parameters
- Hyperoptimalisation_ML_Attack
- Hyperoptimalisation_DL_Attack (GC).ipynb

### Adversarial Training Machine Learning Models and Deep Learning Models
*Logistic Regression (LR), Logit Leaf Model (LLM), Generalized Logistic Rule Regression (GLRM), Random Forest (RF) and XGBoost (XGB)*
- Experiment_ML_Attack.py
*Long short-term memory neural networks (LSTM) and Convolutional Neural Network( CNN)*
- Experiment_DL_Attack (GC).py

### Adversarial Examples Machine Learning Models Deep Learning Models 
*Logistic Regression (LR), Logit Leaf Model (LLM), Generalized Logistic Rule Regression (GLRM), Random Forest (RF) and XGBoost (XGB)*
- Experiment_ML_Attack_Test.py

### Adversarial Examples Machine Learning Models
*Long short-term memory neural networks (LSTM) and Convolutional Neural Network( CNN)*
- Experiment_ML_Attack_Test (GC).py

### Experimental_evaluation.ipynb

This jupyter notebook file contains the code to obtain the plots as displayed in the paper (+ an additional plot to show the differences in label flips, which was omitted from the paper).

## Folders

### labeled_logs_csv_processed

This folder contains cleaned and preprocessed event logs that are made available by this GitHub repository: [Benchmark for outcome-oriented predictive process monitoring](https://github.com/irhete/predictive-monitoring-benchmark). They provide 22 event logs, and we have selected 13 of them. The authors' GitHub repository provide a [Google drive link](https://drive.google.com/open?id=154hcH-HGThlcZJW5zBvCJMZvjOQDsnPR) to download these event logs

### PDF
The folder PDF contains the high-resolution figures (PDF format) that have been used in the paper

## Acknowledgements

We acknowledgde the work provided by [Building Interpretable Models for Business Process Prediction using Shared and Specialised Attention Mechanisms](https://github.com/ZhipengHe/Shared-and-Specialised-Attention-based-Interpretable-Models) for their attention-based bidirectional LSTM architecture to create the long short-term neural networks with attention layers visualisations. In this paper, they use the attributes resource, activity and time and the two former categorical attributes are one-hot encoded (OHE). We have extended this work by allowing different case and event attributes. Here, the categorical attributes are OHE and the dynamic attributes are inserted into an LSTM (to learn the dynamic behaviour). 


