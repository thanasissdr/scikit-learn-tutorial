# GENERAL

The purpose of this repo is just for educational purposes on how to structure a project. More specifically, the task is to create
a class where we apply a machine learning algorithm for a classification task and we assess its performance.


## INSTRUCTIONS

- Create a conda environment, e.g.
```cmd
conda create -n scikit-learn-tutorial python=3.8.10 ipykernel
```
- Install the necessary packages for the project.
```cmd
pip install -r requirements.txt
```

- In order to use absolute imports we can do:
```cmd
pip install -e .
```
That means we can use absolute imports from our **ROOT** path. 