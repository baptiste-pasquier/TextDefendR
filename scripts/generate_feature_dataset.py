import time
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from nlp_adversarial_attacks.reactdetect.utils.magic_vars import PRIMARY_KEY_FIELDS


def vars_values_for_a_line(holder, i, variables, len_sub_variables):
    if not holder[i]["deliverable"]:
        values = [np.nan] * len_sub_variables
        return values
    values = []
    for var in variables:
        values.extend(holder[i]["deliverable"][var][0].tolist())
    return values


def generate_deliverable_variables(holder):
    variables = list(holder[0]["deliverable"].keys())
    sub_variables = []
    for var in variables:
        sub_variables.extend(
            [f"{var}_{j}" for j in range(holder[0]["deliverable"][var].shape[-1])]
        )
    return variables, sub_variables


def generate_vars_df(holder, disable_tqdm=False):
    variables, sub_variables = generate_deliverable_variables(holder)
    var_values = []
    for i in tqdm(holder, disable=disable_tqdm):
        values = vars_values_for_a_line(holder, i, variables, len(sub_variables))
        var_values.append(values)
    var_values = np.array(var_values)
    return pd.DataFrame(var_values, columns=sub_variables)


def generate_index_df(holder):
    unique_id = np.array([holder[i]["unique_id"] for i in holder]).reshape(-1, 1)
    primary_key = np.array([holder[i]["primary_key"] for i in holder])
    index = ["unique_id"] + PRIMARY_KEY_FIELDS
    df_values = np.hstack([unique_id, primary_key])
    index_df = pd.DataFrame(df_values, columns=index)
    return index_df


def get_df_from_holder(holder, disable_tqdm=False):
    index_df = generate_index_df(holder)
    var_df = generate_vars_df(holder, disable_tqdm=disable_tqdm)
    final_df = pd.concat([index_df, var_df], axis=1)

    return final_df


@click.command(help="Generate the dataframe from the joblib file")
@click.option(
    "--joblib_path",
    type=str,
    default="data_tcab/reprs/samplewise/fr+small_distilcamembert_allocine_ALL_ALL.joblib",
    help="Path to the joblib file",
    show_default=True,
)
@click.option(
    "--use_canine",
    type=bool,
    is_flag=True,
    default=False,
    help="Add CANINE features to the dataframe",
    show_default=True,
)
@click.option(
    "--canine_path",
    type=str,
    default="data_tcab/reprs/samplewise/fr+canine_distilcamembert_allocine_ALL_TP.joblib",
    help="Path to the canine joblib file",
    show_default=True,
)
@click.option(
    "--disable_tqdm",
    type=bool,
    is_flag=True,
    default=False,
    help="Disable tqdm progress bar",
    show_default=True,
)
def main(joblib_path, use_canine, canine_path, disable_tqdm):
    start = time.time()
    print("--- loading holder")
    with open(joblib_path, "rb") as f:
        holder = joblib.load(f)
    print()

    print("--- generating dataframe from holder")
    df = get_df_from_holder(holder, disable_tqdm=disable_tqdm)
    print()

    if use_canine:
        print("--- loading canine holder")
        canine_holder = joblib.load(canine_path)
        print()

        print("--- generating caine dataframe")
        canine_df = get_df_from_holder(canine_holder, disable_tqdm=disable_tqdm)
        canine_df = canine_df.set_index("unique_id")
        canine_df = canine_df[canine_df.columns[canine_df.columns.str.startswith("tp")]]
        canine_df.columns = [
            "canine_" + col
            for col in canine_df.columns[canine_df.columns.str.startswith("tp")]
        ]
        print()

        print("--- merging both dataframes")
        df_with_canine = pd.merge(
            df, canine_df, left_on="unique_id", right_index=True, how="left"
        )
        num_missing = (
            df_with_canine["canine_tp_num_lowercase_after_punctuation_0"].isna().sum()
        )
        print(f"Number of missing values: {num_missing}")
        print()

    print("--- saving to disk")
    df_file_path = Path("data_tcab/whole_feature_dataset.pickle")
    df_file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(df_file_path)
    print(f"saved in {df_file_path}")

    if use_canine:
        df_with_canine_file_path = Path(
            "data_tcab/whole_feature_dataset_with_canine.pickle"
        )
        df_with_canine_file_path.parent.mkdir(parents=True, exist_ok=True)
        df_with_canine.to_pickle(df_with_canine_file_path)
        print(f"saved in {df_with_canine_file_path}")

    print(f"\nTotal time: {time.strftime('%H:%M:%S', time.gmtime(time.time()-start))}")


if __name__ == "__main__":
    main()
