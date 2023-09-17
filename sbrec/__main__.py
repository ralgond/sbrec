import json
from argparse import ArgumentParser
from tqdm import tqdm

import torch
from sbrec.sasrec.train import train_sasrec
from sbrec.sasrec.model import TopKModel, SASRec

import pytorch_lightning as pl

def read_stats(data_dir, dataset):
    with open(f"{data_dir}/{dataset}/{dataset}_stats.json", "r") as f:
        stats = json.load(f)
        train_stats = stats["train"]
        test_stats = stats["test"]
    return train_stats, test_stats, stats["num_items"]

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--config-filename", type=str)
#     parser.add_argument("--config-dir", type=str, default="configs")
#     parser.add_argument("--data-dir", type=str, default="datasets")
#     args = parser.parse_args()

#     with open(f"{args.config_dir}/{args.config_filename}.json", "r") as f:
#         config = json.load(f)

#     train_stats, test_stats, num_items = read_stats(args.data_dir, config["dataset"])

#     if config["model"] == "sasrec":
#         trainer, model, train_loader, test_loader = train_sasrec(config, args.data_dir, train_stats, test_stats, num_items)
#     # elif config["model"] == "gru4rec":
#     #     trainer, model, train_loader, test_loader = train_gru(config, args.data_dir, train_stats, test_stats, num_items)
#     else:
#         #raise ValueError('sasrec or gru4rec must be provided as model')
#         raise ValueError('sasrec must be provided as model')

#     if config["overfit_batches"] > 0:
#         test_loader = train_loader

#     trainer.fit(model, train_loader, test_loader)

#     if config["model"] == "sasrec":
#         model.export(trainer.logger.log_dir)

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--config-filename", type=str)
#     parser.add_argument("--config-dir", type=str, default="configs")
#     parser.add_argument("--data-dir", type=str, default="datasets")
#     args = parser.parse_args()

#     with open(f"{args.config_dir}/{args.config_filename}.json", "r") as f:
#         config = json.load(f)

#     train_stats, test_stats, num_items = read_stats(args.data_dir, config["dataset"])

#     if config["model"] == "sasrec":
#         trainer, model, train_loader, test_loader = train_sasrec(config, args.data_dir, train_stats, test_stats, num_items)

#     # model.load_from_checkpoint("lightning_logs/version_2/checkpoints/sasrec-otto-epoch=4-recall_cutoff_20=0.464.ckpt")

#     trainer.validate(model, test_loader, "lightning_logs/version_2/checkpoints/sasrec-otto-epoch=4-recall_cutoff_20=0.464.ckpt")

#     # print ("==========>begin to predict test dataset")

#     # idx, batch = next(enumerate(test_loader))

#     # model.validation_step(batch, idx)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config-filename", type=str)
    parser.add_argument("--config-dir", type=str, default="configs")
    parser.add_argument("--data-dir", type=str, default="datasets")
    args = parser.parse_args()

    with open(f"{args.config_dir}/{args.config_filename}.json", "r") as f:
        config = json.load(f)

    train_stats, test_stats, num_items = read_stats(args.data_dir, config["dataset"])

    if config["model"] == "sasrec":
        trainer, model, train_loader, test_loader = train_sasrec(config, args.data_dir, train_stats, test_stats, num_items)
    # elif config["model"] == "gru4rec":
    #     trainer, model, train_loader, test_loader = train_gru(config, args.data_dir, train_stats, test_stats, num_items)
    else:
        #raise ValueError('sasrec or gru4rec must be provided as model')
        raise ValueError('sasrec must be provided as model')
    
    model = model.load_from_checkpoint("lightning_logs/version_2/checkpoints/sasrec-otto-epoch=4-recall_cutoff_20=0.464.ckpt",
                                                    hparams_file="lightning_logs/version_2/hparams.yaml")
    model = model.cuda()

    topk_model = TopKModel(model)

    topk_model = topk_model.cuda()

    cnt = 0
    recall_cnt = 0
    with open("datasets/otto/otto_test.jsonl") as fin:
        for idx, line in tqdm(enumerate(fin), total=test_stats['num_sessions']):
            js = json.loads(line.strip())

            clicks = []
            for ev in js['events']:
                clicks.append(ev['aid'])

            clicks = clicks[-1*config["max_session_length"]:]

            label = clicks[-1]
            clicks = clicks[:-1]

            indices, _ = topk_model.forward(torch.tensor(clicks, device="cuda"), torch.tensor(20, device="cuda"))

            if label in indices.tolist():
                recall_cnt += 1

            cnt += 1
            if idx % 10000 == 0:
                if cnt != 0:
                    print ("recall@20:", recall_cnt*1.0/cnt)

