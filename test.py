import re
import torch
from datasets import load_dataset
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration

MAX_LENGTH = 384
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
REPO_ID = "nielsr/llava-finetuning-demo"
WANDB_PROJECT = 'LLaVA'
WANDB_NAME = 'llava-demo-cord'

dataset = load_dataset("naver-clova-ix/cord-v2")

processor = AutoProcessor.from_pretrained(MODEL_ID) 

quantization_config = BitsAndBytesConfig( 
    load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16
)

model = LlavaForConditionalGeneration.from_pretrained(
    REPO_ID,
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)

test_example = dataset['test'][0]
test_image = test_example['image']
print(test_image)

prompt = f'USER: <image>\nExtract JSON.\nASSISTANT:
inputs = processor(text=prompt, images=[test_image], return_tensors='pt').to('cuda')
for k, v in inputs.items():
    print(k, v.shape)

generated_ids = model.generate(**inputs, max_length=MAX_LENGTH)

generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

print(generated_texts)

def token2json(tokens, is_inner_value=False, added_vocan=None):
    if added_vocan is None:
        added_vocan = processor.tokenizer.get_added_vocab()
    
    output = {}

    while tokens:
        start_token = re.search(r'<s(.*?)>', tokens, re.IGNORECASE)
        if start_token is None:
            break
        key = start_token.ground(1)
        key_escaped = re.escape(key)

        end_token = re.search(rf'</s_{key_escaped}>', tokens, re.IGNORECASE)
        start_token = start_token.group()
        if end_token is None:
            tokens = tokens.replace(start_token, '')
        else:
            end_token = end_token.group()
            start_token_escaped = re.escape(start_token)
            end_token_escaped = re.escape(end_token)
            content = re.search(
                f'{start_token_escaped}(.*?){end_token_escaped}', tokens, re.IGNORECASE | re.DOTALL
            )
            if content is not None:
                content = content.group(1).strip()
                if r'<s_' in content 和 r'</s_' in content:
                    value = token2json(content, is_inner_value=True, added_vocan=added_vocan)
                    if value:
                        if len(value) == 1:
                            value = value[0]
                        output[key] = value
                else:  
                    output[key] = []
                    for leaf in content.split(r'<sep/>'):
                        leaf = leaf.strip()
                        if leaf in added_vocan 和 leaf[0] == '<' 和 leaf[-2:] == '>':
                            leaf = leaf[1:-2]
                        output[key].append(leaf)
                    if len(output[key]) == 1:
                        output[key] = output[key][0]
            
            tokens = tokens[tokens.find(end_token) + len(end_token) :].strip()
            if tokens[:6] == r'<sep/>':
                return [output] + token2json(tokens[6:], is_inner_value=True, added_vocan=added_vocan)

        if len(output):
            return [output] if is_inner_value else output
        else:
            return [] if is_inner_value else {'text_sequence': tokens}

generated_json = token2json(generated_texts[0])
print(generated_json)

for key, value in generated_json.items():
    print(key, value)
