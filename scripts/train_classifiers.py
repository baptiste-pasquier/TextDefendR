import click
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from xgboost import XGBClassifier

from nlp_adversarial_attacks.reactdetect.utils.magic_vars import PRIMARY_KEY_FIELDS


@click.command()
@click.option(
    "--path_to_pickle",
    type=click.Path(exists=True),
    default="data_tcab/whole_feature_dataset.pickle",
    help="Path to the pickle file containing the dataset",
)
@click.option(
    "--objective",
    type=str,
    default="binary",
    help="Whether to optimize for binary or multiclass classification",
)
@click.option(
    "--model_type",
    type=str,
    default="xgboost",
    help="Which model to use between xgboost and lr",
)
@click.option("--feature_set", type=str, default="tlc", help="Which feature set to use")
@click.option(
    "--scaler", type=str, default="StandardScaler", help="Which scaler to use"
)
@click.option(
    "--pca", type=bool, default=True, help="Whether to try PCA as a preprocessing step"
)
@click.option(
    "--n_trials", type=int, default=100, help="Number of trials to run with optuna"
)
@click.option(
    "--n_jobs", type=int, default=-1, help="Number of jobs to run in parallel"
)
def main(
    path_to_pickle, objective, model_type, feature_set, scaler, pca, n_trials, n_jobs
):
    # Check arguments
    assert objective in [
        "binary",
        "multiclass",
    ], "Objective must be either binary or multiclass"
    assert model_type in ["xgboost", "lr"], "Model type must be either xgboost or lr"
    assert feature_set in [
        "bert",
        "t",
        "tl",
        "tlc",
        "canine",
        "tlc_canine",
    ], "Feature set must be either bert, t, tl, tlc, canine, or tlc_canine"
    assert scaler in [
        "None",
        "StandardScaler",
        "MinMaxScaler",
    ], "Scaler must be either None, StandardScaler, or MinMaxScaler"
    assert pca in [True, False], "PCA must be either True or False"

    # Load the data
    print("-- Loading data")
    df = pd.read_pickle(path_to_pickle)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # Separate the different features
    print("-- Processing the data")
    tp_features = df.columns[df.columns.str.startswith("tp_")].tolist()
    tp_bert_features = df.columns[df.columns.str.startswith("tp_bert_")].tolist()
    lm_features = df.columns[df.columns.str.startswith("lm_")].tolist()
    tm_features = df.columns[df.columns.str.startswith("tm_")].tolist()
    canine_features = df.columns[df.columns.str.startswith("canine_tp_bert_")].tolist()
    feature_sets = {
        "bert": tp_bert_features,
        "t": tp_features,
        "tl": tp_features + lm_features,
        "tlc": tp_features + lm_features + tm_features,
        "canine": canine_features,
        "tlc_canine": tp_features + lm_features + tm_features + canine_features,
    }
    features = feature_sets[feature_set]

    # Split the data into X and y
    var_df = df.loc[:, features]
    id_vars = ["unique_id"] + PRIMARY_KEY_FIELDS
    index_df = df.loc[:, id_vars]
    index_df["label"] = np.where(index_df["attack_name"] == "clean", "clean", "attack")
    del df

    # Split train and test sets
    label_var = "label" if objective == "binary" else "attack_name"
    train_idx, test_idx = train_test_split(
        var_df.index,
        test_size=0.2,
        random_state=42,
    )
    X_train, X_test = var_df.loc[train_idx], var_df.loc[test_idx]
    y_train, y_test = (
        index_df.loc[train_idx, label_var],
        index_df.loc[test_idx, label_var],
    )
    del var_df

    # Encode the labels
    le = LabelEncoder().fit(y_train)
    y_train_enc, y_test_enc = le.transform(y_train), le.transform(y_test)
    # int_to_label = {i: label for i, label in enumerate(le.classes_)}

    # Define the objective function
    def objective_function(trial):
        if model_type == "lr":
            lr_c = trial.suggest_float("lr_c", 1e-5, 1e5, log=True)
            classifier_obj = sklearn.linear_model.LogisticRegression(
                C=lr_c, max_iter=100
            )
        elif model_type == "xgboost":
            xgb_max_depth = trial.suggest_int("xgb_max_depth", 2, 16, log=True)
            xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 10, 100, log=True)
            classifier_obj = XGBClassifier(
                max_depth=xgb_max_depth, n_estimators=xgb_n_estimators
            )

        # Preprocessing steps (scaler and PCA)
        print("-- Building pipeline")
        steps = []
        if scaler == "None":
            steps.append(("scaler", "passthrough"))
        elif scaler == "StandardScaler":
            steps.append(("scaler", StandardScaler()))
        elif scaler == "MinMaxScaler":
            steps.append(("scaler", MinMaxScaler()))

        if pca:
            pca_n_components = trial.suggest_int("pca_n_components", 25, 250, log=True)
            steps.append(("pca", PCA(n_components=pca_n_components)))

        steps.append(("classifier", classifier_obj))
        pipe = Pipeline(steps)
        print(pipe)

        print("-- Training")
        pipe.fit(X_train, y_train_enc)

        print("-- Evaluating")
        y_pred = pipe.predict(X_test)
        accuracy = accuracy_score(y_test_enc, y_pred)

        return accuracy

    # Create a study object and optimize the objective function.
    study_name = f"{objective}_{model_type}_{feature_set}_{scaler.lower()}_{'pca' if pca else 'nopca'}"
    storage_name = f"sqlite:///{study_name}.db"
    print("-- Lauching study")
    print(f"-- Saving in {storage_name}")
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="maximize",
    )
    study.optimize(
        objective_function, n_trials=n_trials, show_progress_bar=True, n_jobs=n_jobs
    )


if __name__ == "__main__":
    main()
