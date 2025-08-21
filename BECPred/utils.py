from argparse import ArgumentParser
from itertools import product
import pandas as pd
import os
import json


def get_dataset_ecnp(args):
    data_name = args.data_name if not args.test else args.test
    if data_name == "ecmap":
        data = pd.read_csv("../data/ECmap/all_reactions.csv")
        data = data.drop_duplicates(["rdkit_reactants", "ec_num"])
    else:
        raise ValueError(f"Unknown dataset: {data_name}")
    return data


def split_ecnp(config, args):
    ec_lvl = config.get("overseer_lvl", 3)
    ec_min_size = config.get("overseer_min", 40)
    splits_path = f"../data/ecnp_kfold/{args.data_name}_{ec_lvl}_{ec_min_size}/"
    os.makedirs(splits_path, exist_ok=True)
    if os.path.exists(split_file := os.path.join(splits_path, f"{args.seed}.json")):
        with open(split_file) as s_file:
            return json.load(s_file)
    else:
        raise FileNotFoundError(f"{split_file} does not exist")


def ecnp_split_to_moe(data_frame, ecnp_split, config, ec_to_i):
    ec_lvl = config["overseer_lvl"]
    ecnp_split = ecnp_split.copy()
    r_train, r_val, r_test = set(ecnp_split["train"]), set(ecnp_split["val"]), set(ecnp_split["test"])
    train, e_train, val, e_val, test, e_test = [], [], [], [], [], []
    for row in data_frame.itertuples():
        reactant = row.rdkit_reactants
        reaction = row.rdkit_reaction
        ec_num = ".".join(row.ec_num.split(".")[:ec_lvl])
        pos = ec_lvl
        while ec_num not in ec_to_i and pos > 0:
            ec_num = ec_num.split(".")
            pos = pos - 1
            ec_num[pos] = "-"
            ec_num = ".".join(ec_num)
        if pos <= 0:
            continue
        elif reactant in r_train:
            train.append(reaction)
            e_train.append(ec_num)
        elif reactant in r_val:
            val.append(reaction)
            e_val.append(ec_num)
        elif reactant in r_test:
            test.append(reaction)
            e_test.append(ec_num)
    ecnp_split["train"], ecnp_split["val"], ecnp_split["test"] = train, val, test
    ecnp_split["e_train"], ecnp_split["e_val"], ecnp_split["e_test"] = e_train, e_val, e_test
    return ecnp_split


def get_arguments():
    parser = ArgumentParser()
    parser.add_argument("--data-name", type=str, default="ecmap", help="Which dataset to use, uspto, pubchem, envipath, baeyer")
    parser.add_argument("--tokenizer", type=str, default="regex", help="Style of tokenizer, selfies, regex")
    parser.add_argument("--split-path", type=str, default="", help="Predetermined split file with train, val and test SMILES")
    parser.add_argument("--max-len", type=int, default=380, help="Maximum encoded length to consider")
    parser.add_argument("--min-len", type=int, default=0, help="Minimum encoded length to consider")
    parser.add_argument("--debug", action="store_true", default=False, help="Whether to set parameters for debugging")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to use")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size to target")
    parser.add_argument("--weights-dataset", type=str, default="", help="Pretrained weights based on what dataset")
    parser.add_argument("--score-all", action="store_true", help="Whether to group same reactants together")
    parser.add_argument("--folds", type=int, default=10, help="How many folds to use for cross validation, default 10")
    parser.add_argument("--preprocessor", type=str, default="rdkit",
                        help="Type of preprocessing to use, envipath or rdkit")
    parser.add_argument("--seed", default=1, type=int, help="Seed to use for randomness")
    parser.add_argument("--test", default="", type=str, help="Dataset to use for testing instead of training")
    parser.add_argument("-tm", "--train-many", action="store_true", help="Whether to train many models")
    parser.add_argument("--workers", type=int, default=20)
    arguments = parser.parse_args()
    return arguments


def expand_search_space(search_space_dict):
    keys = list(search_space_dict.keys())
    values = list(search_space_dict.values())
    return [dict(zip(keys, combo)) for combo in product(*values)]


def write_to_file(file_path, data):
    with open(file_path, "w") as f:
        [f.write(l + "\n") for l in data]
    return file_path
