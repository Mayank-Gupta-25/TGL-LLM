import os
import argparse
import yaml
import torch
from tqdm import tqdm
import random
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from modules.utils_llm import Datasets

def generate_prompt(event, options):
    instruction = "Given the historical context, what is the most likely Object Entity for the Query({}, {}, {}) as the blank space? " \
                  "Please choose the corresponding option.".format(event[0], event[1], "?")
    choices = "\n".join(options)
    prompt = ("Below is an instruction that describes a task, paired with an input that provides further context. "
              "Write a response that appropriately completes the request.\n\n"
              "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ").format(
        instruction=instruction, input=choices)
    return prompt

def test_raw(conf, dataset, k, e2i, r2i):
    device = conf["device"]
    
    i2e = {int(v): k for k, v in e2i.items()}
    i2r = {int(v): k for k, v in r2i.items()}
    
    tokenizer = AutoTokenizer.from_pretrained(conf["base_model"])
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        conf["base_model"],
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    generation_config = GenerationConfig(
        num_beams=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=5,
        do_sample=False,
    )

    count_hit = 0
    count_all = 0

    with torch.no_grad():
        for event_id, history, candidates_id in tqdm(dataset.test_loader):
            for i in range(len(history)):
                sub_id, rel_id, obj_id = event_id[i, 0].item(), event_id[i, 1].item(), event_id[i, 2].item()
                event_text = [i2e[sub_id], i2r[rel_id], i2e[obj_id], str(event_id[i, 3].item())]
                true_label_text = i2e[obj_id]
                
                cand_opts_ids = candidates_id[i, 1:].tolist()
                cand_opts_text = [i2e[cid] for cid in cand_opts_ids]
                
                options = []
                c_idx = random.randint(0, len(cand_opts_text))
                cand_opts_text.insert(c_idx, true_label_text)
                
                label_char = ""
                for j, cand in enumerate(cand_opts_text):
                    options.append(chr(ord('A')+j) + '. ' + cand)
                    if j == c_idx:
                        label_char = chr(ord('A')+j)
                        
                prompt = generate_prompt(event_text, options)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                
                outputs = model.generate(
                    inputs["input_ids"],
                    generation_config=generation_config
                )
                
                gen_tokens = outputs[0][inputs["input_ids"].shape[1]:]
                output_seq = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                
                pattern = re.compile(r'[A-Z]')
                match = pattern.search(output_seq)
                
                if match and match.group() == label_char:
                    count_hit += 1
                count_all += 1
                
    return count_hit / count_all if count_all > 0 else 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="gpu")
    parser.add_argument("-k", "--k_values", type=str, default="3,5,9", help="comma-separated K values")
    args = parser.parse_args()

    conf = yaml.safe_load(open("./config.yaml"))[args.dataset]
    conf["dataset"] = args.dataset
    conf["gpu"] = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    conf["device"] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf["from_pretrain"] = False

    data_path = f"./data/{args.dataset}"
    with open(f"{data_path}/entity2id.json", "r") as f:
        e2i = json.load(f)
    with open(f"{data_path}/relation2id.json", "r") as f:
        r2i = json.load(f)

    k_values = [int(x) for x in args.k_values.split(",")]

    for k in k_values:
        conf["k"] = k
        conf["num_candidate"] = k
        conf["train_sample"] = False
        
        dataset = Datasets(conf)
        conf["num_ent"] = dataset.num_ent
        conf["num_rel"] = dataset.num_rel

        acc = test_raw(conf, dataset, k, e2i, r2i)
        opts = k + 1
        print(f"Dataset={args.dataset}, K={k} (Acc@{opts}): {acc:.4f}")

if __name__ == '__main__':
    main()
