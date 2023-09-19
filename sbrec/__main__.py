import json
import pandas as pd
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

def news_train_main():
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

    if config["overfit_batches"] > 0:
        test_loader = train_loader

    trainer.fit(model, train_loader, test_loader)

def news_pred_main():
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

    model = model.load_from_checkpoint("lightning_logs/version_3/checkpoints/sasrec-news-epoch=14-recall_cutoff_20=0.320.ckpt")

    model = model.cuda()

    topk_model = TopKModel(model)

    topk_model = topk_model.cuda()

    of = open("lightning_logs/version_3/pred.csv", "w+")
    of.write("user_id,article_1,article_2,article_3,article_4,article_5\n")


    with open("datasets/news/news_test.jsonl") as fin:
        for idx, line in tqdm(enumerate(fin), total=test_stats['num_sessions']):
            js = json.loads(line.strip())

            clicks = []
            for ev in js['events']:
                clicks.append(ev['aid'])

            clicks = clicks[-1*config["max_session_length"]:]

            indices, _ = topk_model.forward(torch.tensor(clicks, device="cuda"), torch.tensor(5, device="cuda"))

            # indices = indices + 1

            session_id = js['session']

            l = [str(label) for label in indices.tolist()]
            of.write("{},{}\n".format(session_id, ",".join(l)))
    of.close()

def news_pred_remap_main(in_file_path, out_file_path):
    session_map_df = pd.read_csv("datasets/news/maps_session.csv", sep='\t', names=["origin_id", "id"])
    remap_session = {}
    for idx,row in session_map_df.iterrows():
        remap_session[row["id"]] = row['origin_id']

    item_map_df = pd.read_csv("datasets/news/maps_item.csv", sep='\t', names=["origin_id", "id"])
    remap_item = {}
    for idx,row in item_map_df.iterrows():
        remap_item[row["id"]] = row['origin_id']

    with open (out_file_path, "w+") as of:
        for idx, line in enumerate(open(in_file_path)):
            if idx == 0:
                of.write("user_id,article_1,article_2,article_3,article_4,article_5\n")
                continue
            terms = line.strip("\n").split(",")
            session_id = remap_session[int(terms[0])]
            item_id_list = []
            for item_id in terms[1:]:
                item_id_list.append(remap_item[int(item_id)])
            of.write("{},{}\n".format(session_id, ",".join([str(item_id) for item_id in item_id_list])))

def news_check_main():
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
    
    model = model.load_from_checkpoint("lightning_logs/version_3/checkpoints/sasrec-news-epoch=14-recall_cutoff_20=0.320.ckpt",
                                                    hparams_file="lightning_logs/version_3/hparams.yaml")
    model = model.cuda()

    topk_model = TopKModel(model)

    topk_model = topk_model.cuda()

    cnt = 0
    recall_cnt = 0
    with open("datasets/news/news_test.jsonl") as fin:
        for idx, line in tqdm(enumerate(fin), total=test_stats['num_sessions']):
            js = json.loads(line.strip())

            clicks = []
            for ev in js['events']:
                clicks.append(ev['aid'])

            clicks = clicks[-1*config["max_session_length"]:]

            label = clicks[-1]
            clicks = clicks[:-1]

            if len(clicks) > 0:
                indices, _ = topk_model.forward(torch.tensor(clicks, device="cuda"), torch.tensor(5, device="cuda"))

                if label in indices.tolist():
                    recall_cnt += 1

                cnt += 1
                if idx % 1000 == 0:
                    if cnt != 0:
                        print ("recall@5:", recall_cnt*1.0/cnt)

if __name__ == "__main__":
    news_train_main()
    #news_pred_main()
    #news_pred_remap_main("lightning_logs/version_3/pred.csv", "lightning_logs/version_3/pred_remap.csv")
    #news_check_main()



