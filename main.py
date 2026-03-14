import logging
from typing import Dict, List, Tuple

import click
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm
import json
import os

import wandb
from components.evaluator import GPTEvaluator, NullEvaluator
from components.proposer import (
    LLMProposer,
    LLMProposerDiffusion,
    VLMFeatureProposer,
    VLMProposer,
)
from components.ranker import CLIPRanker, LLMRanker, NullRanker, VLMRanker
from components.set_diff import SetDiff

def load_config(config: str) -> Dict:
    base_cfg = OmegaConf.load("configs/base.yaml")
    cfg = OmegaConf.load(config)
    final_cfg = OmegaConf.merge(base_cfg, cfg)
    args = OmegaConf.to_container(final_cfg)
    args["config"] = config
    if args["wandb"]:
        wandb.init(
            project=args["project"],
            name=args["data"]["name"],
            group=f'{args["data"]["group1"]} - {args["data"]["group2"]} ({args["data"]["purity"]})',
            config=args,
        )
    return args


def load_data(args: Dict) -> Tuple[List[Dict], List[Dict], List[str]]:
    data_args = args["data"]

    df = pd.read_csv(f"{data_args['root']}/{data_args['name']}.csv")

    if data_args["subset"]:
        old_len = len(df)
        df = df[df["subset"] == data_args["subset"]]
        print(
            f"Taking {data_args['subset']} subset (dataset size reduced from {old_len} to {len(df)})"
        )

    dataset1 = df[df["group_name"] == data_args["group1"]].to_dict("records")
    dataset2 = df[df["group_name"] == data_args["group2"]].to_dict("records")
    group_names = [data_args["group1"], data_args["group2"]]

    if data_args["purity"] < 1:
        logging.warning(f"Purity is set to {data_args['purity']}. Swapping groups.")
        assert len(dataset1) == len(dataset2), "Groups must be of equal size"
        n_swap = int((1 - data_args["purity"]) * len(dataset1))
        dataset1 = dataset1[n_swap:] + dataset2[:n_swap]
        dataset2 = dataset2[n_swap:] + dataset1[:n_swap]
    return dataset1, dataset2, group_names


def gen_captions(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    images, sampled_dataset1, sampled_dataset2 = proposer.get_captions(dataset1, dataset2)
    if args["wandb"]:
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return sampled_dataset1, sampled_dataset2

def propose(args: Dict, dataset1: List[Dict], dataset2: List[Dict]) -> List[str]:
    proposer_args = args["proposer"]
    proposer_args["seed"] = args["seed"]
    proposer_args["captioner"] = args["captioner"]

    proposer = eval(proposer_args["method"])(proposer_args)
    hypotheses, logs, images = proposer.propose(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})
        for i in range(len(images)):
            wandb.log(
                {
                    f"group 1 images ({dataset1[0]['group_name']})": images[i][
                        "images_group_1"
                    ],
                    f"group 2 images ({dataset2[0]['group_name']})": images[i][
                        "images_group_2"
                    ],
                }
            )
    return hypotheses


def rank(
    args: Dict,
    hypotheses: List[str],
    dataset1: List[Dict],
    dataset2: List[Dict],
    group_names: List[str],
) -> List[str]:
    ranker_args = args["ranker"]
    ranker_args["seed"] = args["seed"]

    ranker = eval(ranker_args["method"])(ranker_args)

    scored_hypotheses = ranker.rerank_hypotheses(hypotheses, dataset1, dataset2)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(scored_hypotheses))
        wandb.log({"scored hypotheses": table_hypotheses})
        for i in range(5):
            wandb.summary[f"top_{i + 1}_difference"] = scored_hypotheses[i][
                "hypothesis"
            ].replace('"', "")
            wandb.summary[f"top_{i + 1}_score"] = scored_hypotheses[i]["auroc"]

    scored_groundtruth = ranker.rerank_hypotheses(
        group_names,
        dataset1,
        dataset2,
    )
    if args["wandb"]:
        table_groundtruth = wandb.Table(dataframe=pd.DataFrame(scored_groundtruth))
        wandb.log({"scored groundtruth": table_groundtruth})

    return [hypothesis["hypothesis"] for hypothesis in scored_hypotheses]

def get_diffs_from_set_diff(
    args: Dict,
    dataset1: List[Dict],
    dataset2: List[Dict]
) -> List[str]:
    differ_args = args["differ"]
    seed = args["seed"]
    differ = eval(differ_args["method"])(differ_args)

    cls0_diffs, cls0_name, cls1_diffs, cls1_name = differ.get_differences(dataset1, dataset2, seed)
    if args["wandb"]:
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(cls0_diffs))
        wandb.log({f"scored difference {cls0_name}": table_hypotheses})
        top_n_cls0 = min(5, len(cls0_diffs))
        for i in range(top_n_cls0):
            wandb.summary[f"{cls0_name}_top_{i + 1}_difference"] = cls0_diffs[i]["text"].replace('"', "")
            wandb.summary[f"{cls0_name}_top_{i + 1}_score"] = cls0_diffs[i]["correlation"]
    
        table_hypotheses = wandb.Table(dataframe=pd.DataFrame(cls1_diffs))
        wandb.log({f"scored difference {cls1_name}": table_hypotheses})
        top_n_cls1 = min(5, len(cls1_diffs))
        for i in range(top_n_cls1):
            wandb.summary[f"{cls1_name}_top_{i + 1}_difference"] = cls1_diffs[i]["text"].replace('"', "")
            wandb.summary[f"{cls1_name}_top_{i + 1}_score"] = cls1_diffs[i]["correlation"]

    return [diff["text"] for diff in cls0_diffs], cls0_name, [diff["text"] for diff in cls1_diffs], cls1_name

def get_diffs_from_llm(
    args: Dict,
    dataset1: List[Dict],
    dataset2: List[Dict]
) -> List[str]:
    differ_args = args["differ"]
    differ = eval(differ_args["method"])(differ_args)
    hypotheses, logs = differ.get_hypotheses(dataset1, dataset2)
    if args["wandb"]:
        wandb.log({"hypotheses": wandb.Table(dataframe=pd.DataFrame(hypotheses))})
        wandb.log({"logs": wandb.Table(dataframe=pd.DataFrame(logs))})

    return hypotheses

def dual_evaluate(args: Dict, ranked_hypotheses_cls0: List[str], cls0_name: str, ranked_hypotheses_cls1: List[str], cls1_name: str) -> Dict:
    evaluator_args = args["evaluator"]
    evaluator = eval(evaluator_args["method"])(evaluator_args)
    metrics_cls0, evaluated_hypotheses_cls0 = evaluator.evaluate(
        ranked_hypotheses_cls0,
        cls0_name,
        cls1_name
    )

    metrics_cls1, evaluated_hypotheses_cls1 = evaluator.evaluate(
        ranked_hypotheses_cls1,
        cls1_name,
        cls0_name
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses_cls0)
        )
        wandb.log({f"evaluated hypotheses {cls0_name}": table_evaluated_hypotheses})
        prefixed_metrics_cls0 = {f"{k}_cls0":v for k,v in metrics_cls0.items()}
        wandb.log(prefixed_metrics_cls0)

        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses_cls1)
        )
        wandb.log({f"evaluated hypotheses {cls1_name}": table_evaluated_hypotheses})
        prefixed_metrics_cls1 = {f"{k}_cls1":v for k,v in metrics_cls1.items()}
        wandb.log(prefixed_metrics_cls1)

    return metrics_cls0, metrics_cls1

def evaluate(args: Dict, ranked_hypotheses: List[str], group_names: List[str]) -> Dict:
    evaluator_args = args["evaluator"]

    evaluator = eval(evaluator_args["method"])(evaluator_args)

    metrics, evaluated_hypotheses = evaluator.evaluate(
        ranked_hypotheses,
        group_names[0],
        group_names[1],
    )

    if args["wandb"] and evaluator_args["method"] != "NullEvaluator":
        table_evaluated_hypotheses = wandb.Table(
            dataframe=pd.DataFrame(evaluated_hypotheses)
        )
        wandb.log({"evaluated hypotheses": table_evaluated_hypotheses})
        wandb.log(metrics)
    return metrics

def clean_hypo(hypo: str) -> str:
    new_hypo = hypo.replace('\"','')
    return new_hypo

def prepare_hypothesis_corpus(args: Dict, dataset1: List[Dict], dataset2: List[Dict], group_names:List[str]):
    filename = args["hypo_file"]
    a_b = group_names[0]+ "_" + group_names[1]
    b_a = group_names[1]+ "_" + group_names[0]
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            json.dump({}, file, indent=4)

    with open(filename, 'r') as file:
        data = json.load(file)
    
    hypotheses = propose(args, dataset1, dataset2)
    data[a_b] = [clean_hypo(t) for t in hypotheses]

    hypotheses = propose(args, dataset2, dataset1)
    data[b_a] = [clean_hypo(t) for t in hypotheses]

    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    logging.info(f"Saved {len(data[a_b])} hypotheses for {a_b}")
    logging.info(f"Saved {len(data[b_a])} hypotheses for {b_a}")


@click.command()
@click.option("--config", help="config file")
def main(config):
    logging.info("Loading config...")
    args = load_config(config)
    # print(args)
    set_diff_enabled = args['set_diff_enabled']
    find_bottle_neck_exp = args['bottle_neck_exp']
    prep_knowledge_bank = args['prep_knowledge_bank']
    set_diff_with_universal_vocab_enabled = args['set_diff_with_universal_vocab_enabled']
    logging.info("Loading data...")
    dataset1, dataset2, group_names = load_data(args)
    # print(dataset1, dataset2, group_names)

    if prep_knowledge_bank:
        prepare_hypothesis_corpus(args, dataset1, dataset2, group_names)
        return

    logging.info("Proposing hypotheses...")
    if set_diff_enabled:
        captioned_dataset1, captioned_dataset2 = gen_captions(args, dataset1, dataset2)
    elif find_bottle_neck_exp:
        captioned_dataset1, captioned_dataset2 = gen_captions(args, dataset1, dataset2)
        hypotheses = get_diffs_from_llm(args, captioned_dataset1, captioned_dataset2)
    elif set_diff_with_universal_vocab_enabled:
        pass
    else:
        hypotheses = propose(args, dataset1, dataset2)
        # print(hypotheses)

    logging.info("Ranking hypotheses...")
    if set_diff_enabled:
        ranked_hypotheses_cls0, cls0_name, ranked_hypotheses_cls1, cls1_name = get_diffs_from_set_diff(args, captioned_dataset1, captioned_dataset2)
    elif set_diff_with_universal_vocab_enabled:
        ranked_hypotheses_cls0, cls0_name, ranked_hypotheses_cls1, cls1_name = get_diffs_from_set_diff(args, dataset1, dataset2)
    else:
        ranked_hypotheses = rank(args, hypotheses, dataset1, dataset2, group_names)
        # print(ranked_hypotheses)

    logging.info("Evaluating hypotheses...")
    if set_diff_enabled or set_diff_with_universal_vocab_enabled:
        metrics_cls0, metrics_cls1 = dual_evaluate(args, ranked_hypotheses_cls0, cls0_name, ranked_hypotheses_cls1, cls1_name)
        logging.info("Experiment Completed!")
    else:
        metrics = evaluate(args, ranked_hypotheses, group_names)
        # print(metrics)


if __name__ == "__main__":
    main()
