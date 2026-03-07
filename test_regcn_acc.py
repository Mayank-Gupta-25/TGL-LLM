import os
import argparse
import yaml
import pickle
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from modules.regcn import REGCN
from modules.utils_llm import Datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="gpu")
    args = parser.parse_args()

    # Load conf
    conf = yaml.safe_load(open("./config.yaml"))
    conf = conf[args.dataset]
    conf["dataset"] = args.dataset
    conf["data_path"] = conf.get("path", "./data/")
    
    # We need n_layers, n_bases etc from pretrain config for REGCN init
    pre_conf = yaml.safe_load(open("./config_pretrain.yaml"))[args.dataset]
    for k, v in pre_conf.items():
        if k not in conf:
            conf[k] = v
            
    conf["gpu"] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    conf["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["num_candidate"] = 3 # Default initialization

    # Load Graph Dict
    with open(os.path.join(conf["data_path"], conf["dataset"], 'graph_dict.pkl'), 'rb') as fp:
        graph_dict = pickle.load(fp)

    for cand_k in [3, 5, 9]:
        conf["num_candidate"] = cand_k
        conf["k"] = cand_k
        conf["rs"] = False # Do not subset 10w, want all tests
        conf["train_sample"] = False
        
        # Load Dataset utilizing utils_llm to get exact candidates
        dataset = Datasets(conf)
        
        # Inject metadata for REGCN
        conf['num_ent'] = dataset.num_ent
        conf['num_rel'] = dataset.num_rel
        
        # Load model only once
        if cand_k == 3:
            model_name = f"./checkpoints/regcn/{args.dataset}/REGCN--convtranse-lr1e-05-wd1e-06-dim200-histlen3-layers2"
            if not os.path.exists(model_name):
                model_name = f"./checkpoints/regcn/{args.dataset}/REGCN"
                
            model = REGCN(conf).to(conf["device"])
            checkpoint = torch.load(model_name, map_location=conf["device"])
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()

        hits = 0
        total = 0
        
        with torch.no_grad():
            for event_id, history, candidates_id in tqdm(dataset.test_loader):
                for i in range(len(history)): # iterate over batch
                    tim_list = history[i].tolist()
                    g_list = [graph_dict[tim].to(conf["device"]) for tim in tim_list]
                    test_triples_input = torch.LongTensor(event_id[i, :4]).to(conf["device"]).unsqueeze(0)
                    
                    # predict_p processes 1 snap at a time
                    final_score, _, _, _, _ = model.predict_p(g_list, test_triples_input)
                    
                    obj_scores = final_score[0] # Just the single object prediction
                    obj_target = event_id[i, 2].item()
                    cand_opts = candidates_id[i, 1:].tolist()
                    all_choices = [obj_target] + cand_opts
                    
                    choice_scores = obj_scores[all_choices]
                    best_idx = torch.argmax(choice_scores).item()
                    
                    if best_idx == 0:
                        hits += 1
                    total += 1
                    
        acc = hits / total if total > 0 else 0
        opts = cand_k + 1
        print(f"Dataset={args.dataset}, K={cand_k} (Acc@{opts}): {acc:.4f}")

if __name__ == '__main__':
    main()
