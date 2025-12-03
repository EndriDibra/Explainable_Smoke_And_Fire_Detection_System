# Author: Endri Dibra 
# Project: Smoke and Fire Detection with Sensor Readings

# Importing the required libraries
import os
import time

# Importing shap for model explainability
import shap

# Importing optuna for hyperparameter optimization
import optuna
import warnings
import numpy as np
import pandas as pd
import seaborn as sns

# Importing joblib for saving and loading models
from joblib import dump
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split, cross_val_score

# Importing evaluation metrics for model assessment
from sklearn.metrics import (
   
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
    roc_curve, auc, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay
)

# Importing ensemble models for machine learning
from sklearn.ensemble import (

    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, HistGradientBoostingClassifier
)

from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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

# Ignoring warnings to reduce output clutter
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)


# Creating directory for saving plots if it does not exist
os.makedirs("Plots", exist_ok=True)

# Creating directory for saving models if it does not exist
os.makedirs("Models", exist_ok=True)

# Creating directory for saving XAI plots if it does not exist
os.makedirs("XAI_Plots", exist_ok=True)

# Loading dataset from CSV file
dataFrame = pd.read_csv("Dataset.csv")

# Renaming columns for easier access
dataFrame.rename(columns={

    "Humidity[%]": "Humidity",
    "Temperature[C]": "Temperature",
    "TVOC[ppb]": "TVOC",
    "eCO2[ppm]": "ECO2",
    "Pressure[hPa]": "Pressure"

}, inplace=True)

# Dropping irrelevant columns from dataset
dropCols = ["Unnamed: 0", "CNT", "UTC"]
dataFrame.drop(columns=dropCols, inplace=True, errors='ignore')

# Checking for missing values in dataset
if dataFrame.isna().sum().sum() > 0:

    # Filling missing values with median if present
    dataFrame.fillna(dataFrame.median(), inplace=True)

# Creating correlation heatmap for features
plt.figure(figsize=(12,10))
sns.heatmap(dataFrame.corr(), annot=True, cmap="viridis")
plt.title("Feature Correlation Matrix")

# Saving correlation heatmap to file
plt.savefig("Plots/correlation_matrix.png", dpi=300)

# Closing current figure to free memory
plt.close()

# Separating features and target variable
X = dataFrame.drop("Fire Alarm", axis=1)
y = dataFrame["Fire Alarm"]

# Splitting dataset into training and testing sets with stratification
XTrain, XTest, yTrain, yTest = train_test_split(

    X, y, test_size=0.2, random_state=42, stratify=y
)

# Creating standard scaler for deep learning model
scalerDL = StandardScaler()

# Fitting and transforming training data for deep learning
XTrainDL = scalerDL.fit_transform(XTrain)

# Transforming test data using fitted scaler
XTestDL = scalerDL.transform(XTest)


# Defining function to create Keras neural network model
def createKerasModel(inputDim):

    # Creating sequential neural network model
    model = Sequential([

        Dense(64, activation='relu', input_shape=(inputDim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compiling model with Adam optimizer and binary crossentropy loss
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Creating dictionary of models for evaluation
modelDictionary = {

    "Dummy": DummyClassifier(),

    "LogisticRegression": Pipeline([

        ("Scaler", StandardScaler()),
        ("Model", LogisticRegression(max_iter=2000, solver="liblinear"))
    ]),

    "RidgeClassifier": Pipeline([

        ("Scaler", StandardScaler()),
        ("Model", RidgeClassifier())
    ]),

    "KNN": Pipeline([

        ("Scaler", StandardScaler()),
        ("Model", KNeighborsClassifier())
    ]),

    "SVC": Pipeline([

        ("Scaler", StandardScaler()),
        ("Model", SVC(probability=True))
    ]),

    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
    "ExtraTree": ExtraTreeClassifier(),
    "RandomForest": RandomForestClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "HistGradientBoosting": HistGradientBoostingClassifier(),
    "XGBoost": XGBClassifier(eval_metric="logloss", use_label_encoder=False),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier(verbose=0),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(32,16), max_iter=600),
    "KerasNN": createKerasModel(XTrain.shape[1])
}


# Defining function to evaluate models and save metrics
def evaluateModels(modelDict):

    # Creating list to store results
    resultsList = []

    # Iterating over each model in dictionary
    for modelName, modelObj in modelDict.items():

        print(f"Training {modelName}...")

        # Recording start time for training
        startTime = time.time()

        # Handling Keras neural network training separately
        if modelName == "KerasNN":

            earlyStop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

            modelObj.fit(

                XTrainDL, yTrain,
                validation_split=0.2,
                shuffle=True,
                epochs=50,
                batch_size=32,
                callbacks=[earlyStop],
                verbose=0
            )

            # Predicting binary classes
            predictions = (modelObj.predict(XTestDL) > 0.5).astype(int)

            # Predicting probabilities
            probas = modelObj.predict(XTestDL).flatten()

        else:

            # Fitting model on training data
            modelObj.fit(XTrain, yTrain)

            # Predicting binary classes
            predictions = modelObj.predict(XTest)

            try:

                # Attempting to predict probabilities
                probas = modelObj.predict_proba(XTest)[:,1]

            except:

                probas = None

        # Calculating AUC score if probabilities exist
        aucScore = roc_auc_score(yTest, probas) if probas is not None else None

        # Recording end time for training
        endTime = time.time()

        # Appending model metrics to results list
        resultsList.append([

            modelName,
            accuracy_score(yTest, predictions),
            precision_score(yTest, predictions, zero_division=0),
            recall_score(yTest, predictions, zero_division=0),
            f1_score(yTest, predictions, zero_division=0),
            aucScore,
            endTime - startTime
        ])

        # Creating evaluation plots if probabilities exist
        if probas is not None:

            # Calculating ROC curve
            fpr, tpr, _ = roc_curve(yTest, probas)

            plt.figure()
            plt.plot(fpr, tpr, label=f'AUC = {aucScore:.3f}')
            plt.plot([0,1],[0,1],'--',color='gray')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve: {modelName}')
            plt.legend(loc='lower right')
            plt.savefig(f'Plots/ROC_{modelName}.png', dpi=300)
            plt.close()

            # Calculating precision-recall curve
            precision, recall, _ = precision_recall_curve(yTest, probas)

            plt.figure()
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curve: {modelName}')
            plt.savefig(f'Plots/PR_{modelName}.png', dpi=300)
            plt.close()

            # Creating calibration curve
            prob_true, prob_pred = calibration_curve(yTest, probas, n_bins=10)

            plt.figure()
            plt.plot(prob_pred, prob_true, marker='o')
            plt.plot([0,1],[0,1],'--', color='gray')
            plt.xlabel('Mean Predicted Probability')
            plt.ylabel('Fraction of Positives')
            plt.title(f'Calibration Curve: {modelName}')
            plt.savefig(f'Plots/Calibration_{modelName}.png', dpi=300)
            plt.close()

        # Creating and saving confusion matrix plot
        cm = confusion_matrix(yTest, predictions)
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(cmap='Blues')

        plt.title(f'Confusion Matrix: {modelName}')
        plt.savefig(f'Plots/ConfusionMatrix_{modelName}.png', dpi=300)
        plt.close()

    # Converting results list to pandas DataFrame
    resultsDF = pd.DataFrame(resultsList, columns=[

        "Model","Accuracy","Precision","Recall","F1 Score","AUC Score","Training Time"
    ])

    # Saving model metrics summary to CSV
    resultsDF.to_csv("Models/model_metrics_summary.csv", index=False)

    print("Saved model metrics summary CSV.")

    # Returning results sorted by F1 score in descending order
    return resultsDF.sort_values(by="F1 Score", ascending=False)

# Running evaluation for all models
evaluationResults = evaluateModels(modelDictionary)

print("=== Model Ranking ===")
print(evaluationResults)

# Selecting best model based on F1 score
bestModelName = evaluationResults.iloc[0]["Model"]
bestModel = modelDictionary[bestModelName]

print(f"Selected Best Model: {bestModelName}")

# Saving best model depending on type
if bestModelName != "KerasNN":

    dump(bestModel, f"Models/best_AutoML_model_{bestModelName}.joblib")

else:

    bestModel.save("Models/best_AutoML_model_KerasNN.keras")


# Defining Optuna objective function for XGBoost hyperparameter tuning
def objectiveOptuna(trial):

    # Suggesting hyperparameters
    paramGrid = {

        "n_estimators": trial.suggest_int("n_estimators",100,600),
        "max_depth": trial.suggest_int("max_depth",3,20),
        "learning_rate": trial.suggest_float("learning_rate",0.01,0.3)
    }

    # Creating XGBoost model with suggested hyperparameters
    model = XGBClassifier(**paramGrid, eval_metric="logloss", use_label_encoder=False)

    # Computing cross-validation score
    score = cross_val_score(model, XTrain, yTrain, cv=3, scoring="f1").mean()

    # Returning negative F1 score for minimization
    return -score

# Creating Optuna study for hyperparameter tuning
optunaStudy = optuna.create_study(direction="minimize")

# Running optimization for 20 trials
optunaStudy.optimize(objectiveOptuna, n_trials=20)
print("Best Hyperparameters:", optunaStudy.best_params)

# Creating best XGBoost model using optimized hyperparameters
bestXGB = XGBClassifier(**optunaStudy.best_params, eval_metric="logloss", use_label_encoder=False)
bestXGB.fit(XTrain, yTrain)

# Saving tuned XGBoost model
dump(bestXGB, "Models/best_XGBoost_model_Tuned.joblib")

# Predicting test data with tuned XGBoost
predictions = bestXGB.predict(XTest)
print(classification_report(yTest, predictions))


# Function to compute SHAP values for the best model (limiting to 1000 rows)
def computeShapValues(modelObj, modelName, XTrain, XTest, XTrainDL=None, XTestDL=None):

    # Subsetting for speed (first 1000 rows)
    XTestSubset = XTest.iloc[:1000] if hasattr(XTest, 'iloc') else XTest[:1000]
    XTestDLSubset = XTestDL[:1000] if XTestDL is not None else None
    shapValues = None

    if modelName == "KerasNN":

        # Using GradientExplainer for DL
        backgroundDL = XTrainDL[:100]
        explainer = shap.GradientExplainer(modelObj, backgroundDL)
        shapValues = explainer.shap_values(XTestDLSubset)[0] 

    elif modelName in ["LogisticRegression", "RidgeClassifier"]:

        # Using LinearExplainer: Exact and fast for linear models
        background = XTrain[:100]  
        innerModel = modelObj.named_steps['Model'] if hasattr(modelObj, 'named_steps') else modelObj
        explainer = shap.LinearExplainer(innerModel, background)
        shapValues = explainer.shap_values(XTestSubset)

    elif modelName in ["XGBoost", "LightGBM", "CatBoost"]:

        # Using TreeExplainer for boosting
        explainer = shap.TreeExplainer(modelObj)
        shapValues = explainer.shap_values(XTestSubset)


    elif any(name in modelName for name in ["RandomForest", "GradientBoosting", "DecisionTree", "ExtraTree", "HistGradientBoosting"]):

        # Using TreeExplainer for scikit trees
        explainer = shap.TreeExplainer(modelObj)
        shapValues = explainer.shap_values(XTestSubset)

    else:

        # Using KernelExplainer fallback 
        background = shap.kmeans(XTrain, 10).data 

        if hasattr(modelObj, 'predict_proba'):

            explainer = shap.KernelExplainer(modelObj.predict_proba, background)

        else:

            explainer = shap.KernelExplainer(modelObj.predict, background)

        shapValues = explainer.shap_values(XTestSubset)

    return shapValues


# Function to compute and plot SHAP feature importance (mean abs SHAP values) for a model
def plotShapImportance(shapValues, XExplain, modelName, featureNames, isDL=False, classIdx=1):

    # Extracting values if it's an Explanation object (for robustness across SHAP versions)
    if hasattr(shapValues, 'values'):

        shapValues = shapValues.values

    # Handling multi-class or binary formats
    if isinstance(shapValues, list):

        shapValues = shapValues[classIdx]

    elif isinstance(shapValues, np.ndarray) and shapValues.ndim == 3:

        shapValues = shapValues[:, :, classIdx]

    # Computing mean absolute SHAP values for feature importance
    shapImportance = np.abs(shapValues).mean(axis=0)

    # Creating DataFrame for sorting
    importanceDF = pd.DataFrame({

        'feature': featureNames,
        'importance': shapImportance
    
    }).sort_values('importance', ascending=True)

    # Plotting horizontal bar chart for top contributing features
    plt.figure(figsize=(10, 6))
    plt.barh(importanceDF['feature'], importanceDF['importance'])
    plt.xlabel('Mean |SHAP Value|')
    plt.title(f'Top Feature Contributions (SHAP) for {modelName}')
    plt.gca().invert_yaxis()  # Highest importance on top
    plt.tight_layout()
    plt.savefig(f"XAI_Plots/SHAP_importance_bar_{modelName}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"SHAP importance plot saved for {modelName}")

    return importanceDF


# Computing and showcasing SHAP only for the Best Model (limiting to 1000 rows)
print("\nComputing SHAP for Best Model")

featureNames = X.columns.tolist()
startTime = time.time()

print(f"Computing SHAP for {bestModelName}...")

try:

    shapValues = computeShapValues(bestModel, bestModelName, XTrain, XTest, XTrainDL, XTestDL)

    isDL = (bestModelName == "KerasNN")

    XExplainSubset = XTest.iloc[:1000] if not isDL else XTestDL[:1000]

    importanceDF = plotShapImportance(shapValues, XExplainSubset, bestModelName, featureNames, isDL=isDL)

    # Saving SHAP importance results to CSV
    importanceDF.to_csv("XAI_Results.csv", index=False)
    print("Saved XAI results to XAI_Results.csv")

    # Printing top 5 for the best model
    print(f"\nTop 5 Features for {bestModelName} (Time: {time.time() - startTime:.1f}s)")
    print(importanceDF.tail(5))

except Exception as e:

    print(f"Error computing SHAP for {bestModelName}: {e} (Time: {time.time() - startTime:.1f}s)")