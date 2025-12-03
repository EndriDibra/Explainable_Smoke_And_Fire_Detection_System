# Author: Endri Dibra 
# Project: Training an ML model using Sensor Fusion Dataset for smoke and fire detection

# Importing the required libraries 
import os
import time
import logging
import numpy as np
import pandas as pd
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
   
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    matthews_corrcoef, cohen_kappa_score, RocCurveDisplay, ConfusionMatrixDisplay
)

# Importing shap for model explainability
import shap

# Monkey patch for SHAP compatibility with NumPy 2.0+
if not hasattr(np, 'int'):
   
    np.int = int

if not hasattr(np, 'long'):

    np.long = int

if not hasattr(np, 'float'):

    np.float = float

if not hasattr(np, 'complex'):

    np.complex = complex

if not hasattr(np, 'bool'):

    np.bool = bool

shap.maskers._MASKERS_SUPPORTS_INT = True

import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="In the future `np.bool` will be defined as the corresponding NumPy scalar.")

# Setting up logging for robustness
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Path to Fusion Dataset
fusionDatasetPath = "Fusion_Dataset.csv"

# Creating directory for saving XAI plots if it does not exist
os.makedirs("Results/XAI_Plots", exist_ok=True)

# Creating directory for saving models if it does not exist
os.makedirs("Models", exist_ok=True)

# Creating directory for saving results if it does not exist
os.makedirs("Results", exist_ok=True)

# Checking if dataset file exists
if not os.path.exists(fusionDatasetPath):

    logger.error(f"Error! Dataset file '{fusionDatasetPath}' not found.")

else:

    # Loading Fusion Dataset
    try:

        fusionDf = pd.read_csv(fusionDatasetPath)
        logger.info(f"Dataset loaded successfully: {fusionDf.shape}")

    except Exception as e:

        logger.error(f"Error loading dataset: {e}")
        raise

# Validating columns
featureCols = ['Temperature', 'Humidity', 'TVOC', 'ECO2', 'Raw H2', 'Raw Ethanol', 'Pressure',
               'PM1.0', 'PM2.5', 'NC0.5', 'NC1.0', 'NC2.5', 'Ptabular', 'Pfire', 'Psmoke', 'Pdetection']

targetCols = ['FireAlert', 'SmokeAlert']

missingFeatures = [col for col in featureCols if col not in fusionDf.columns]

missingTargets = [col for col in targetCols if col not in fusionDf.columns]

if missingFeatures or missingTargets:

    logger.error(f"Missing columns: Features={missingFeatures}, Targets={missingTargets}")

    raise ValueError("Required columns missing in dataset.")

# Extracting features and targets
X = fusionDf[featureCols].copy()
y = fusionDf[targetCols].copy()

# Checking for infinite values and handling
if np.isinf(X.values).any():

    logger.warning("Infinite values detected in features; replacing with NaN.")

    X.replace([np.inf, -np.inf], np.nan, inplace=True)

# Handling missing values with imputation
imputer = SimpleImputer(strategy='median')
XImputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)

logger.info(f"Imputed {XImputed.isnull().sum().sum()} missing values.")

# Scaling features for consistency (though RF is scale-invariant, good practice)
scaler = StandardScaler()
XScaled = pd.DataFrame(scaler.fit_transform(XImputed), columns=XImputed.columns, index=XImputed.index)

# Ensuring targets are binary
for target in targetCols:

    y[target] = y[target].astype(int).clip(0, 1)

    uniqueVals = y[target].unique()

    if not set(uniqueVals).issubset({0, 1}):

        logger.warning(f"Target '{target}' has non-binary values: {uniqueVals}. Clipping to 0/1.")

# Splitting dataset into training and testing sets
try:

    XTrain, XTest, yTrain, yTest = train_test_split(
       
        XScaled, y, test_size=0.2, random_state=42, stratify=y['FireAlert']
    )

    logger.info(f"Data split: Train={XTrain.shape}, Test={XTest.shape}")

except Exception as e:

    logger.error(f"Error in data splitting: {e}")
    raise

# Initializing Random Forest classifiers for each target
rfFire = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rfSmoke = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

# Training the models with cross-validation for robustness check
cvScoresFire = cross_val_score(rfFire, XTrain, yTrain['FireAlert'], cv=5, scoring='roc_auc')
cvScoresSmoke = cross_val_score(rfSmoke, XTrain, yTrain['SmokeAlert'], cv=5, scoring='roc_auc')

logger.info(f"CV ROC-AUC - Fire: {cvScoresFire.mean():.4f} (+/- {cvScoresFire.std() * 2:.4f})")
logger.info(f"CV ROC-AUC - Smoke: {cvScoresSmoke.mean():.4f} (+/- {cvScoresSmoke.std() * 2:.4f})")

# Fitting the models
try:

    rfFire.fit(XTrain, yTrain['FireAlert'])
    rfSmoke.fit(XTrain, yTrain['SmokeAlert'])

    logger.info("Models trained successfully.")

except Exception as e:

    logger.error(f"Error training models: {e}")
    raise

# Predicting
yPredFire = rfFire.predict(XTest)
yPredSmoke = rfSmoke.predict(XTest)
yProbaFire = rfFire.predict_proba(XTest)[:, 1]
yProbaSmoke = rfSmoke.predict_proba(XTest)[:, 1]


# Function for evaluating and printing metrics with error handling
def evaluateModel(yTrue, yPred, yProba, label="Model"):

    print(f"\n{label} Metrics:\n")

    try:

        print(f"Accuracy: {accuracy_score(yTrue, yPred):.4f}")
        print(f"Precision: {precision_score(yTrue, yPred, zero_division=0):.4f}")
        print(f"Recall: {recall_score(yTrue, yPred, zero_division=0):.4f}")
        print(f"F1-score: {f1_score(yTrue, yPred, zero_division=0):.4f}")
        print(f"ROC-AUC: {roc_auc_score(yTrue, yProba):.4f}")
        print(f"Matthews Correlation Coefficient: {matthews_corrcoef(yTrue, yPred):.4f}")
        print(f"Cohen's Kappa: {cohen_kappa_score(yTrue, yPred):.4f}")

    except Exception as e:

        logger.warning(f"Error computing some metrics for {label}: {e}")

        print("Some metrics unavailable due to edge cases (e.g., no positive predictions).")
    
    print("\nConfusion Matrix:\n", confusion_matrix(yTrue, yPred))
    print("\nClassification Report:\n", classification_report(yTrue, yPred, zero_division=0))

    # Plotting ROC Curve
    try:

        RocCurveDisplay.from_predictions(yTrue, yProba)
        plt.title(f"ROC Curve - {label}")
        plt.savefig(f"Results/ROC_{label}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:

        logger.warning(f"Error plotting ROC for {label}: {e}")

    # Plotting Confusion Matrix
    try:

        cm = confusion_matrix(yTrue, yPred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {label}")
        plt.savefig(f"Results/CM_{label}.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    except Exception as e:

        logger.warning(f"Error plotting CM for {label}: {e}")


# Evaluating FireAlert model
evaluateModel(yTest['FireAlert'], yPredFire, yProbaFire, "FireAlert")

# Evaluating SmokeAlert model
evaluateModel(yTest['SmokeAlert'], yPredSmoke, yProbaSmoke, "SmokeAlert")

# Logging feature importances for interpretability
importancesFire = pd.DataFrame({'Feature': featureCols, 'Importance': rfFire.feature_importances_}).sort_values('Importance', ascending=False)
importancesSmoke = pd.DataFrame({'Feature': featureCols, 'Importance': rfSmoke.feature_importances_}).sort_values('Importance', ascending=False)

logger.info(f"Top features FireAlert: \n{importancesFire.head()}")
logger.info(f"Top features SmokeAlert: \n{importancesSmoke.head()}")

# Saving feature importances to CSV
importancesFire.to_csv("Results/RF_FireAlert_Importances.csv", index=False)
importancesSmoke.to_csv("Results/RF_SmokeAlert_Importances.csv", index=False)

# Saving trained models and preprocessors
try:

    dump(rfFire, "Models/RF_FireAlert_Model.joblib")
    dump(rfSmoke, "Models/RF_SmokeAlert_Model.joblib")

    dump(imputer, "Models/Imputer.joblib")
    dump(scaler, "Models/Scaler.joblib")

    print("\nTrained Random Forest models and preprocessors saved: RF_FireAlert_Model.joblib, RF_SmokeAlert_Model.joblib, Imputer.joblib, Scaler.joblib")

    logger.info("Models and preprocessors saved successfully.")

except Exception as e:

    logger.error(f"Error saving models: {e}")


# Function to compute SHAP values for the model (limiting to 1000 rows for efficiency)
def computeShapValues(modelObj, XTest):

    # Subsetting for speed (first 1000 rows or full if smaller)
    subset_size = min(1000, len(XTest))
    XTestSubset = XTest.iloc[:subset_size] if hasattr(XTest, 'iloc') else XTest[:subset_size]
    
    # Using TreeExplainer for RandomForest
    explainer = shap.TreeExplainer(modelObj)
    shapValues = explainer.shap_values(XTestSubset)
    
    return shapValues, XTestSubset


# Function to plot SHAP bar summary using built-in SHAP function
def plotShapBarSummary(shapValues, XExplain, modelName, featureNames):

    # Handling binary format (for TreeExplainer on binary classification)
    if isinstance(shapValues, list):

        shapValues = shapValues[1]  # Positive class SHAP values

    # Built-in SHAP bar plot for global feature importance (mean |SHAP|)
    plt.figure()
    shap.summary_plot(shapValues, XExplain, feature_names=featureNames, plot_type="bar", show=False)
    plt.title(f"SHAP Bar Summary - Top Feature Contributions for {modelName}")
    plt.tight_layout()
    plt.savefig(f"Results/XAI_Plots/SHAP_bar_summary_{modelName}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP bar summary plot saved for {modelName}")


# Function to compute and save mean |SHAP| importance to CSV (for tabular view)
def saveShapImportanceCSV(shapValues, featureNames, modelName):

    # Handling binary format
    if isinstance(shapValues, list):

        shapValues = shapValues[1]

    # Comput ingmean absolute SHAP values
    shapImportance = np.abs(shapValues).mean(axis=0)

    # Creating DataFrame
    importanceDF = pd.DataFrame({

        'feature': featureNames,
        'importance': shapImportance

    }).sort_values('importance', ascending=False)

    # Saving to CSV
    csvPath = f"Results/XAI_{modelName}_importance.csv"
    importanceDF.to_csv(csvPath, index=False)
    print(f"Saved SHAP importance to {csvPath}")

    # Print top 5
    print(f"\nTop 5 SHAP Features for {modelName}:")
    print(importanceDF.head())

    return importanceDF

# Computing and showcasing SHAP for both models (limiting to 1000 rows)
print("\nComputing SHAP for FireAlert Model")

featureNames = featureCols
startTime = time.time()

try:

    shapValuesFire, XExplainSubsetFire = computeShapValues(rfFire, XTest)
    plotShapBarSummary(shapValuesFire, XExplainSubsetFire, "FireAlert_RF", featureNames)
    importanceDFFire = saveShapImportanceCSV(shapValuesFire, featureNames, "FireAlert_RF")

    print(f"SHAP computation for FireAlert complete (Time: {time.time() - startTime:.1f}s)")

except Exception as e:

    print(f"Error computing SHAP for FireAlert: {e} (Time: {time.time() - startTime:.1f}s)")

print("\nComputing SHAP for SmokeAlert Model")

startTime = time.time()

try:

    shapValuesSmoke, XExplainSubsetSmoke = computeShapValues(rfSmoke, XTest)
    plotShapBarSummary(shapValuesSmoke, XExplainSubsetSmoke, "SmokeAlert_RF", featureNames)
    importanceDFSmoke = saveShapImportanceCSV(shapValuesSmoke, featureNames, "SmokeAlert_RF")

    print(f"SHAP computation for SmokeAlert complete (Time: {time.time() - startTime:.1f}s)")

except Exception as e:

    print(f"Error computing SHAP for SmokeAlert: {e} (Time: {time.time() - startTime:.1f}s)")


print("\nAnalysis complete. Check 'Results/' for outputs, 'Models/' for saved artifacts, and 'Results/XAI_Plots/' for SHAP visualizations.")