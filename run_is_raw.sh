#!/bin/bash
# Quick re-run for IS K=3 and K=5 only
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tgl

# Only run K=3 and K=5 for IS since K=9 was already captured
python3 -c "
import os, argparse, yaml, torch, random, re, json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from modules.utils_llm import Datasets

def generate_prompt(event, options):
    instruction = 'Given the historical context, what is the most likely Object Entity for the Query({}, {}, {}) as the blank space? Please choose the corresponding option.'.format(event[0], event[1], '?')
    choices = '\n'.join(options)
    prompt = ('Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: ').format(instruction=instruction, input=choices)
    return prompt

os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device('cuda:0')

conf = yaml.safe_load(open('./config.yaml'))['IS']
conf['dataset'] = 'IS'
conf['gpu'] = '5'
conf['device'] = device
conf['from_pretrain'] = False

data_path = './data/IS'
with open(f'{data_path}/entity2id.json', 'r') as f:
    e2i = json.load(f)
with open(f'{data_path}/relation2id.json', 'r') as f:
    r2i = json.load(f)
i2e = {int(v): k for k, v in e2i.items()}
i2r = {int(v): k for k, v in r2i.items()}

tokenizer = AutoTokenizer.from_pretrained(conf['base_model'])
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(conf['base_model'], torch_dtype=torch.bfloat16, device_map=device)
model.eval()
generation_config = GenerationConfig(num_beams=1, bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, max_new_tokens=5, do_sample=False)

for k in [3, 5]:
    conf['k'] = k
    conf['num_candidate'] = k
    conf['train_sample'] = False
    dataset = Datasets(conf)
    conf['num_ent'] = dataset.num_ent
    conf['num_rel'] = dataset.num_rel

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
                label_char = ''
                for j, cand in enumerate(cand_opts_text):
                    options.append(chr(ord('A')+j) + '. ' + cand)
                    if j == c_idx:
                        label_char = chr(ord('A')+j)
                prompt = generate_prompt(event_text, options)
                inputs = tokenizer(prompt, return_tensors='pt').to(device)
                outputs = model.generate(inputs['input_ids'], generation_config=generation_config)
                gen_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                output_seq = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                pattern = re.compile(r'[A-Z]')
                match = pattern.search(output_seq)
                if match and match.group() == label_char:
                    count_hit += 1
                count_all += 1
    acc = count_hit / count_all if count_all > 0 else 0
    opts = k + 1
    print(f'Dataset=IS, K={k} (Acc@{opts}): {acc:.4f}')
"
