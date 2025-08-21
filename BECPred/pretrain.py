import logging
from rxnfp.models import SmilesLanguageModelingModel
import torch
from BECPred.utils import get_arguments, expand_search_space, get_dataset_ecnp, split_ecnp, ecnp_split_to_moe, \
    write_to_file

logger = logging.getLogger(__name__)

def main(args):
    config = {
      "architectures": [
        "BertForMaskedLM"
      ],
      "attention_probs_dropout_prob": 0.1,
      "hidden_act": "gelu",
      "hidden_dropout_prob": 0.2,
      "hidden_size": 512,
      "initializer_range": 0.02,
      "intermediate_size": 512,
      "layer_norm_eps": 1e-12,
      "max_position_embeddings": 512,
      "model_type": "bert",
      "num_attention_heads": 4,
      "num_hidden_layers": 12,
      "pad_token_id": 0,
      "type_vocab_size": 2,
      "overseer_min": 40,
      "overseer_lvl": 3
    }
    vocab_path = '../data/vocab.txt'
    data = get_dataset_ecnp(args)
    split = split_ecnp(config, args)
    ec_to_i = split["ec_to_i"]
    moe_split = ecnp_split_to_moe(data, split, config, ec_to_i)
    train_file = write_to_file(f"../data/pretrain_txt/ecnp_train_{args.seed}.txt", moe_split["train"])
    eval_file = write_to_file(f"../data/pretrain_txt/ecnp_eval_{args.seed}.txt", moe_split["val"])
    test_file = write_to_file(f"../data/pretrain_txt/ecnp_test_{args.seed}.txt", moe_split["test"])

    args = {'config': config,
            'vocab_path': vocab_path,
            'train_batch_size': 64,
            'manual_seed': args.seed,
            "fp16": False,
            "num_train_epochs": 30,
            'max_seq_length': 512,
            'evaluate_during_training': True,
            'overwrite_output_dir': True,
            'output_dir': f'../models/pretrain_{args.seed}',
            'learning_rate': 1e-4
           }

    model = SmilesLanguageModelingModel(model_type='bert', model_name=None, args=args, use_cuda=True)
    model.train_model(train_file=train_file, eval_file=eval_file)


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
