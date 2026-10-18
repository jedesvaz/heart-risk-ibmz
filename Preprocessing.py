# Function created to preprocess and select the most significative indicators for the dataset "heart_disease_health_indicators_BRFSS2015.csv" #
# This function has been created by the team "Biodatitos" on the "IBM Z Datathon 2026" #

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

def preprocess(dataset): # Receives the complete dataset
    df = dataset
    df = df.drop_duplicates().reset_index(drop=True) # Drop of duplicates

    Indicators = df.drop(columns="HeartDiseaseorAttack")
    Ground_Truth = df["HeartDiseaseorAttack"]

    # Training split
    X_train, X_temp, Y_train, Y_temp = train_test_split(Indicators, Ground_Truth, train_size=0.50, stratify=Ground_Truth, random_state=25)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp, Y_temp, test_size=0.40, stratify=Y_temp, random_state=25)

    # Mutual information: captures nonlinear relations
    discrete_features = [c != "BMI" for c in X_train.columns]
    mi = pd.Series(mutual_info_classif(X_train, Y_train, discrete_features=discrete_features, random_state=25),
                   index=X_train.columns)

    # Random Forest (with class_weight for the unbalanced)
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=50,
                                class_weight="balanced_subsample", n_jobs=-1, random_state=25)
    rf.fit(X_train, Y_train)
    rf_imp = pd.Series(rf.feature_importances_, index=X_train.columns)

    # Permutation importance
    perm = permutation_importance(rf, X_val, Y_val, scoring="average_precision",
                                  n_repeats=10, n_jobs=-1, random_state=25)
    perm_imp = pd.Series(perm.importances_mean, index=X_train.columns)

    imp = pd.DataFrame({"mutual_info": mi, "random_forest": rf_imp, "permutation": perm_imp})
    norm = imp.clip(lower=0)
    norm = norm / norm.sum()

    weights = {"mutual_info": 1, "random_forest": 1, "permutation": 2}
    imp["score"] = sum(norm[c] * w for c, w in weights.items()) / sum(weights.values())

    # Ranking formation
    ranking = imp.sort_values("score", ascending=False)
    top = ranking.head(10).index.tolist()

    # Most relevant Indicators
    acum = ranking["score"].cumsum()
    top_95 = acum[acum <= 0.95].index.tolist()
    top_85 = acum[acum <= 0.85].index.tolist()
    top_70 = acum[acum <= 0.70].index.tolist()

    return {"top_95": top_95, "top_85": top_85, "top_70": top_70} # Returns the most significant indicators for the 95%, 85% and 70% 
                                                                  # of the combined correlation score