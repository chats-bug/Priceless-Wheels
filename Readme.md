# Welcome to Pricing Wheels! 
*A simple used car price prediction tool*

## Table of Contents
1. [Introduction](#introduction)
2. [Roadmap](#roadmap)


## Introduction
Welcome to Priceless Wheels! In this project, our goal is to build a model that can accurately predict the price of a used vehicle based on various factors such as make, model, year, mileage, and condition. The automobile industry is one of the largest and most competitive industries in the world, with millions of vehicles being sold each year. The price of a vehicle can have a significant impact on a consumer's purchasing decision and it is important for both buyers and sellers to have an understanding of the market value of a vehicle. By using machine learning algorithms and data analysis, we aim to provide a reliable and robust model that can assist in determining the fair market value of a vehicle. Join us on this exciting journey as we delve into the world of vehicle price prediction.


## Dataset
The dataset is collected from the Cardekho website. A complete overview of the dataset can be found at this [kaggle link](https://www.kaggle.com/datasets/sukritchatterjee/used-cars-dataset-cardekho). The dataset is made publicly available for research and educational purposes.
The scrapper can be found is `src/scrapper` directory.


## Installation and Dependencies
This project uses some of the most common libraries such as `pandas`, `matplotlib`, `scikit-learn` and many more. To install the dependencies, run the following command:
```
pip install -r requirements.txt
```

## Usage
To run the project locally, follow these steps:
1. Clone the project ad cd into it 
```
git clone [project-url]
cd priceless-wheels
```
2. Setup a new python environment and install the dependecies.
3. Run the `setup.py` file; this will run the preprocessing steps on the data and make it ready for the model.
4. Cd into the `model_training` directory and run the `training.py` file. This will train the model and save the model in `data/models`.
5. Feed the appropriate data into `testing.py` file and get the predictions.


## Model
The final predictive model is an ensemble of 2 gradient boosting algorithms: A CatBoost Regressor and a LightGBM Regressor. These were chosen because of a multitue of reasons - only one of them uses oblivious or symmetric trees, and other such factors which lead to two slightly different models that can be ensembled together (whcih is apparent from their respective feature importances, even though the performance is boradly similar, the important features are vastly different between the 2 models, therefore making them less correlated and helping in overall variance reduction).


## Hyperparameter Tuning
We used an open source HPO library called `optuna`. Bayesian Optimized Hyperband along with a TPE sampler was used for optimizing the hyperparameters.


## Results
The model achieved a mean average error (MAE) of INR 76,000, and a MAPE of ~10.2%. Considering the location choice, and the competence of buyers and sellers to negotiate a deal, a varation of 10% can be expected.

## Roadmap
- [x] Data Collection
- [x] Data Cleaning and Preprocessing
- [x] Exploratory Data Analysis
- [x] Feature Engineering
- [x] Model Building
- [x] Hyperparameter Tuning
- [ ] Model Evaluation
- [ ] Deployment
