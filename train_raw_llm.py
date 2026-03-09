"""
Raw LLM Fine-Tuning + Testing Script (Table 3 Reproduction)
============================================================
This script fine-tunes Llama-2-7b-chat-hf with LoRA on text-only prompts
(NO graph embeddings) and tests with Acc@K metrics.

This reproduces the "Raw" row from Table 3 of the TGL-LLM paper.
"""
import os
import sys
import json
import yaml
import torch
import random
import re
import numpy as np
import pandas as pd
import argparse
import transformers
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training

def setup_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ======================== DATA ========================

class RawTextDataset(Dataset):
    """Dataset that returns text-only prompts for training."""
    def __init__(self, tokenizer, data_csv, candidates_csv, max_len=512):
        self.tokenizer = tokenizer
        self.data = pd.read_csv(data_csv)[["Subject", "Relation", "Object", "Date"]]
        self.candidates = pd.read_csv(candidates_csv)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        event = self.data.iloc[idx].values.tolist()
        label_text = str(event[2])
        cands = eval(self.candidates.iloc[idx]["Candidates"])
        
        options = []
        c_idx = random.randint(0, len(cands))
        cands.insert(c_idx, label_text)
        
        label_char = ""
        for i, cand in enumerate(cands):
            options.append(chr(ord('A') + i) + '. ' + str(cand))
            if i == c_idx:
                label_char = chr(ord('A') + i)
        
        # Build prompt without trailing space
        prompt = build_prompt(event, options)
        
        # Tokenize separately to ensure no merging
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        answer_ids = self.tokenizer.encode(label_char, add_special_tokens=False)
        eos_id = [self.tokenizer.eos_token_id]
        
        input_ids = prompt_ids + answer_ids + eos_id
        labels = [-100] * len(prompt_ids) + answer_ids + eos_id
        
        # Padding
        if len(input_ids) < self.max_len:
            pad_len = self.max_len - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * pad_len
            labels += [-100] * pad_len
        else:
            input_ids = input_ids[:self.max_len]
            labels = labels[:self.max_len]
            
        attention_mask = [1 if x != self.tokenizer.pad_token_id else 0 for x in input_ids]
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }


def build_prompt(event, options):
    """Build text-only prompt (no graph tokens)."""
    choices = "\n".join(options)
    prompt = (
        f"You are an assistant for event forecasting. "
        f"Predict the missing object from the query.\n\n"
        f"Query: ({event[0]}, {event[1]}, ?, {event[3]})\n\n"
        f"Options:\n{choices}\n\n"
        f"The answer is"
    )
    # Note: no trailing space here. We'll add it if needed or trust tokenizer.
    return prompt


# ======================== TRAIN ========================

class RawTrainer(Trainer):
    def _save(self, output_dir=None, state_dict=None):
        super()._save(output_dir, state_dict)
        self.model.save_pretrained(output_dir)


def train_raw(conf, device):
    dataset_name = conf["dataset"]
    data_dir = f"./data/{dataset_name}"
    
    tokenizer = AutoTokenizer.from_pretrained(conf["base_model"], padding_side="right")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model = AutoModelForCausalLM.from_pretrained(
        conf["base_model"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    for k in [3, 5, 9]:
        train_csv = os.path.join(data_dir, "train.csv")
        cand_csv = os.path.join(data_dir, "candidates", f"K_{k}", f"train_{k}_candidates.csv")
        
        if not os.path.exists(cand_csv):
            print(f"WARNING: {cand_csv} not found, skipping K={k}")
            continue
        
        train_dataset = RawTextDataset(tokenizer, train_csv, cand_csv)
        
        # Subsample to 10000 (coreset size from paper)
        if len(train_dataset) > 10000:
            indices = random.sample(range(len(train_dataset)), 10000)
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        
        output_dir = f"./checkpoints/RawLLM/{dataset_name}_K{k}"
        
        training_args = TrainingArguments(
            warmup_steps=20,
            num_train_epochs=1,
            learning_rate=3e-4,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=32,  # effective batch = 128
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="no",
            save_strategy="epoch",
            output_dir=output_dir,
            run_name=f"RawLLM_{dataset_name}_K{k}",
            fp16=False,
            bf16=True,
            remove_unused_columns=False,
        )
        
        os.environ["WANDB_DISABLED"] = "true"
        
        trainer = RawTrainer(
            model=model,
            train_dataset=train_dataset,
            args=training_args,
            data_collator=lambda data: {
                "input_ids": torch.stack([d["input_ids"] for d in data]),
                "attention_mask": torch.stack([d["attention_mask"] for d in data]),
                "labels": torch.stack([d["labels"] for d in data]),
            },
        )
        
        print(f"\n{'='*60}")
        print(f"Training Raw LLM for {dataset_name} K={k}")
        print(f"{'='*60}")
        
        # Resume from checkpoint if it exists
        resume_from_checkpoint = False
        if os.path.exists(output_dir) and len(os.listdir(output_dir)) > 0:
            # Look for checkpoint-X folders
            checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
            if checkpoints:
                resume_from_checkpoint = True
                print(f"Found existing checkpoints: {checkpoints}. Resuming training...")
        
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.save_model(output_dir)
        print(f"Model saved to {output_dir}")
        
        # Reset for next K (reload fresh LoRA)
        if k < 9:
            model = AutoModelForCausalLM.from_pretrained(
                conf["base_model"],
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model = prepare_model_for_kbit_training(model)
            model = get_peft_model(model, lora_config)


# ======================== TEST ========================

def test_raw(conf, device, k_list=[3, 9]):
    dataset_name = conf["dataset"]
    data_dir = f"./data/{dataset_name}"
    
    tokenizer = AutoTokenizer.from_pretrained(conf["base_model"], padding_side="right")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    generation_config = GenerationConfig(
        num_beams=1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=5,
        do_sample=False,
    )
    
    test_csv = os.path.join(data_dir, "test.csv")
    test_data = pd.read_csv(test_csv)[["Subject", "Relation", "Object", "Date"]]
    
    results = {}
    
    for k in k_list:  # Table 3 only uses Acc@4 and Acc@10
        cand_csv = os.path.join(data_dir, "candidates", f"K_{k}", f"test_{k}_candidates.csv")
        
        if not os.path.exists(cand_csv):
            print(f"WARNING: {cand_csv} not found, skipping K={k}")
            continue
        
        test_candidates = pd.read_csv(cand_csv)
        
        # Load the fine-tuned LoRA model
        checkpoint_dir = f"./checkpoints/RawLLM/{dataset_name}_K{k}"
        
        base_model = AutoModelForCausalLM.from_pretrained(
            conf["base_model"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        
        if os.path.exists(checkpoint_dir):
            model = PeftModel.from_pretrained(base_model, checkpoint_dir)
            print(f"Loaded fine-tuned LoRA from {checkpoint_dir}")
        else:
            model = base_model
            print(f"WARNING: No checkpoint found at {checkpoint_dir}, using base model")
        
        model.eval()
        
        count_hit = 0
        count_all = 0
        
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        
        batch_size = 32
        with torch.no_grad():
            for i in tqdm(range(0, len(test_data), batch_size), desc=f"{dataset_name} K={k}"):
                batch_prompts = []
                batch_labels = []
                
                for j in range(i, min(i + batch_size, len(test_data))):
                    event = test_data.iloc[j].values.tolist()
                    label_text = str(event[2])
                    cands = eval(test_candidates.iloc[j]["Candidates"])
                    
                    options = []
                    c_idx = random.randint(0, len(cands))
                    cands.insert(c_idx, label_text)
                    
                    label_char = ""
                    for m, cand in enumerate(cands):
                        options.append(chr(ord('A') + m) + '. ' + str(cand))
                        if m == c_idx:
                            label_char = chr(ord('A') + m)
                    
                    prompt = build_prompt(event, options)
                    batch_prompts.append(prompt)
                    batch_labels.append(label_char)
                
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                
                outputs = model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                    generation_config=generation_config,
                )
                
                for b_idx in range(len(batch_prompts)):
                    input_len = inputs["input_ids"].shape[1]
                    gen_tokens = outputs[b_idx][input_len:]
                    output_seq = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                    
                    # Pattern to find the first A-Z char
                    pattern = re.compile(r'[A-Z]')
                    match = pattern.search(output_seq)
                    
                    if match and match.group() == batch_labels[b_idx]:
                        count_hit += 1
                    count_all += 1
        
        acc = count_hit / count_all if count_all > 0 else 0
        opts = k + 1
        results[f"Acc@{opts}"] = acc
        print(f"Dataset={dataset_name}, K={k} (Acc@{opts}): {acc:.4f}")
        
        # Cleanup GPU memory
        del model, base_model
        torch.cuda.empty_cache()
    
    return results


# ======================== MAIN ========================

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, required=True, choices=["train", "test"], help="train or test")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="IR, IS, or EG")
    parser.add_argument("-k", "--k", type=int, default=None, help="Specific K to test")
    parser.add_argument("-g", "--gpu", type=str, default="0", help="GPU id")
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    conf = yaml.safe_load(open("./config.yaml"))[args.dataset]
    conf["dataset"] = args.dataset

    setup_seeds()

    if args.option == "train":
        train_raw(conf, device)
    elif args.option == "test":
        k_list = [args.k] if args.k is not None else [3, 9]
        test_raw(conf, device, k_list=k_list)
