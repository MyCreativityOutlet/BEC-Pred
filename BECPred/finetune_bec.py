import json
import os.path

import numpy as np
import pandas as pd
import torch
import logging
import pkg_resources
import sklearn
from rxnfp.models import SmilesClassificationModel

from BECPred.utils import get_arguments, expand_search_space, split_ecnp

logger = logging.getLogger(__name__)


def format_data(reactants, labels, i_to_ec):
    df = {"text": [], "labels": []}
    for reaction, enzyme_enc in zip(reactants, labels):
        enzyme_enc = np.array(enzyme_enc)
        for ec_id in np.nonzero(enzyme_enc)[0]:
            df["text"].append(reaction)
            df["labels"].append(i_to_ec[ec_id.item()])
    return pd.DataFrame(df)


def main(args):
    config = {"overseer_min": 40, "overseer_lvl": 3}
    split = split_ecnp(config, args)
    ec_to_i = split["ec_to_i"]
    i_to_ec = {i: ec for ec, i in ec_to_i.items()}
    final_train_df = format_data(split["train"], split["e_train"], i_to_ec)
    eval_df = format_data(split["val"], split["e_val"], i_to_ec)
    test_df = format_data(split["test"], split["e_test"], i_to_ec)
    # df = pd.read_pickle('../data/final_df_ec.pkl')
    # print(df[['rxn', 'class_id']].head())
    # train_df = df.loc[df['split']=='train']
    # eval_df = df[['rxn', 'class_id']].loc[df['split']=='val']
    # eval_df.columns = ['text', 'labels']
    #
    # all_train_reactions = train_df.rxn.values.tolist()
    # corresponding_labels = train_df.class_id.values.tolist()
    # final_train_df = pd.DataFrame({'text': all_train_reactions, 'labels': corresponding_labels})
    # final_train_df = final_train_df.sample(frac=1.)
    output_dir = f'../models/refine_{args.seed}'
    model_args = {
        'wandb_project': None, 'num_train_epochs': 1, 'overwrite_output_dir': True,
        'learning_rate': 1e-5, 'gradient_accumulation_steps': 1,
        'regression': False, "num_labels": 155, "fp16": False,
        "evaluate_during_training": True, 'manual_seed': args.seed,
        "max_seq_length": 512, "train_batch_size": 8,"warmup_ratio": 0.00,
        'output_dir': output_dir,
        'thread_count': 4,
        }

    # optional
    model_path = f"../models/pretrain_{args.seed}/"
    model = SmilesClassificationModel("bert", model_path, num_labels=model_args["num_labels"], args=model_args, use_cuda=torch.cuda.is_available())

    # optional
    # train_model_path =  pkg_resources.resource_filename("best_model")

    def f1_multiclass(labels, preds):
          return sklearn.metrics.f1_score(labels, preds, average='weighted')

    def prec_multiclass(labels, preds):
          return sklearn.metrics.precision_score(labels, preds, average='weighted')

    def rec_multiclass(labels, preds):
          return sklearn.metrics.recall_score(labels, preds, average='weighted')

    model.train_model(final_train_df, eval_df=eval_df, prec=prec_multiclass, rec=rec_multiclass, acc=sklearn.metrics.accuracy_score, mcc=sklearn.metrics.matthews_corrcoef, f1=f1_multiclass, args={"process_count": 20})
    result, model_outputs, wrong_predictions = model.eval_model(test_df, prec=prec_multiclass, rec=rec_multiclass, acc=sklearn.metrics.accuracy_score, mcc=sklearn.metrics.matthews_corrcoef, f1=f1_multiclass)
    print(result)
    print(model_outputs)
    print(type(model_outputs[0]))
    with open(os.path.join(output_dir, "test_results.json"), "w") as t_file:
        json.dump({"result": result, "model_outputs": model_outputs}, t_file)


if __name__ == "__main__":
    arguments = get_arguments()
    if arguments.train_many:
        search_space = {"seed": list(range(10))}
        search_space = expand_search_space(search_space)
        for search in search_space:
            for k, v in search.items():
                if k in vars(arguments):
                    setattr(arguments, k, v)
                else:
                    raise ValueError(f"Search space key {k} not in args")
            main(arguments)
            torch.cuda.empty_cache()
    else:
        main(arguments)
