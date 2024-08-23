# Mobile Price Prediction using Multiple Linear Regression

This repository contains a project that applies **Multiple Linear Regression** to predict the price of mobile phones based on various features such as Ratings, RAM, ROM, Mobile Size, Camera specifications, and Battery Power.

## Project Overview

The objective of this project is to predict mobile phone prices using a dataset containing information on multiple mobile features. We apply multiple linear regression, a statistical method that models the relationship between multiple independent variables and a dependent variable (price).

## Dataset

The dataset includes the following features:
- **Ratings**: Average user rating of the mobile.
- **RAM**: Random Access Memory in GB.
- **ROM**: Read-Only Memory (internal storage) in GB.
- **Mobile Size**: Screen size in inches.
- **Primary Cam**: Megapixels of the primary camera.
- **Selfie Cam**: Megapixels of the front camera.
- **Battery Power**: Battery capacity in mAh.
- **Price**: Target variable representing the price of the mobile.

## Installation

1. Clone the repository:
```bash
   git clone https://github.com/himpar21/Multiple-Linear-Regression
   cd Multiple-Linear-Regression
```   
2. Install the required Python libraries:
```bash
  pip install pandas numpy scikit-learn matplotlib
```
3. Run the script:
```bash
python multiplelinear.py
```
## Code Overview
#### multiple_linear_regression.py
- This script performs the following:
1) Loads and preprocesses the dataset.
2) Splits the dataset into training and testing sets.
3) Trains a Multiple Linear Regression model.
4) Makes predictions on the test set.
5) Evaluates the model using metrics such as R² Score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
6) Visualizes the actual vs. predicted values.
7) Outputs the regression equation and some sample predictions.
  
## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib

## Results
- R² Score: 0.4332
- Mean Squared Error (MSE): 239,357,657.43
- Root Mean Squared Error (RMSE): 15,471.19
