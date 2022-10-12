# gan_ensembles

> Ensembles of Generative Adversarial Networks as a Data Augmentation Technique for Alzheimer research.
> Machine Learning Engineer Nanodegree at Udacity - Capstone Project

### Setup

```sh
# install dependencies
conda create -n gan_ensembles python=3.6.5
conda activate gan_ensembles
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
# start jupyter server
jupyter notebook
```

### Notebooks

Notebooks in this project were designed to be executed in sequence:

1) Download MRIs dataset - http://localhost:8888/notebooks/download_dataset.ipynb
2) Download MRIs dataset - http://localhost:8888/notebooks/data_exploration_viz.ipynb
3) Data preprocessing and Data splitting - http://localhost:8888/notebooks/data_preproc_split.ipynb
4) DCGAN implementation and Control Model (CM) training - http://localhost:8888/notebooks/dcgan_control_model.ipynb
5) DCGAN Ensemble Model 1 (eGANs) training - http://localhost:8888/notebooks/dcgan_ensemble_model_1.ipynb
6) DCGAN Ensemble Model 2 (seGANs) training - http://localhost:8888/notebooks/dcgan_ensemble_model_2.ipynb
7) DCGAN Ensemble Model 3 (cGANs) training - http://localhost:8888/notebooks/dcgan_ensemble_model_3.ipynb
8) Metrics visualisation and conclusion - http://localhost:8888/notebooks/metrics_viz_outro.ipynb

### Docs

Original proposal, corrected proposal, and final project report included in [capstone_project_docs](capstone_project_docs) folder.
