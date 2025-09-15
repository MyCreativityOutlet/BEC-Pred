import copy
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import logging
import random
import sklearn
from rxnfp.models import SmilesClassificationModel
from tqdm import trange

from BECPred.utils import get_arguments, split_ecnp, format_data

logger = logging.getLogger(__name__)


def load_model(path):
    return SmilesClassificationModel("bert", path, use_cuda=torch.cuda.is_available(), args={"silent": True})


def evaluate(args):
    model = load_model("")

    df = pd.read_pickle('../data/final_df_ec.pkl')
    df = df.loc[df['split']=='test']
    print(df[:5])
    test_df = df.rxn
    test_reactions = test_df.values.tolist()
    y_true = df.class_id
    y_true = y_true.values.tolist()
    print(y_true[:5])
    y_true = y_true

    y_preds = model.predict(test_reactions)

    y_preds = pd.Series(y_preds)
    y_pred = y_preds.values.tolist()
    y_pred = y_pred[0]


    def f1_multiclass(y_true,y_pred):
          return sklearn.metrics.f1_score(y_true,y_pred, average='weighted')

    def prec_multiclass(y_true,y_pred):
          return sklearn.metrics.precision_score(y_true,y_pred, average='weighted')

    def rec_multiclass(y_true,y_pred):
          return sklearn.metrics.recall_score(y_true,y_pred, average='weighted')

    # for y1_true, y1_pred in zip(y_true, y_pred):
    prec=prec_multiclass(y_true,y_pred)
    rec=rec_multiclass(y_true,y_pred)
    acc=sklearn.metrics.accuracy_score(y_true,y_pred)
    mcc=sklearn.metrics.matthews_corrcoef(y_true,y_pred)
    f1=f1_multiclass(y_true,y_pred)

    print(prec,rec,acc,mcc,f1)
    # print(y_preds)


def runtime_evaluate(args):
    config = {"overseer_min": 40, "overseer_lvl": 3}
    runtime_path = "outputs/runtimes.json"
    batch_sizes = [1, 10, 100, 1000]
    time_inner = {b: [] for b in batch_sizes}
    times = {"BECPred": copy.deepcopy(time_inner)}
    seeds = range(10)
    try:
        for seed in seeds:
            args.seed = seed
            model = load_model(f"../models/refine_{seed}")
            split = split_ecnp(config, args)
            all_train_data = split["train"]
            for size in batch_sizes:
                train_data = all_train_data[:size * 50]
                for i in trange(0, len(train_data), size):
                    start_time = time.time()
                    _ = model.predict(train_data[i: i+size])
                    end_time = time.time() - start_time
                    times["BECPred"][size].append(end_time)
    except RuntimeError as e:
        print(f"Failed to evaluate runtime on batch size: {size} error:\n{e}")
    with open(runtime_path, "w") as r_file:
        json.dump(times, r_file)


def main(args):
    runtime_evaluate(args)
    pass


if __name__ == "__main__":
    arguments = get_arguments()
    main(arguments)
