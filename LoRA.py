import torch
from data import LlavaDataset
from PIL import Image
from peft import peft_model, PeftModel
from transformers import LlavaForConditionalGeneration, AutoProcessor

raw_name_or_path = 'path'
peft_name_or_path = 'path'

model = LlavaForConditionalGeneration.from_pretrained(raw_name_or_path, device_map='cuda:0', torch_dtype=torch.bfloat16)
model = PeftModel.from_pretrained(model, peft_name_or_path, adapter_name='pt')
processor = AutoProcessor.from_pretrained(raw_name_or_path)
model.eval()
print('OK')

llavadataset = LlavaDataset('data/liuhaotian/LLaVA-CC3M-Pretrain-595K')
print(len(llavadataset), llavadataset[10])

testdata = llavadataset[1302]
print(testdata)

def build_model_input(model, processor, testdata:tuple):
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': testdata[0]}
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenizer=False, add_generation_prompt=True
    )

    image = Image.open(testdata[2])
    inputs = processor(text=prompt, image=image, return_tensors='pt')

    for tk in inputs.key():
        inputs[tk] = inputs[tk].to(model.device)
    
    generate_ids = model.generate(**inputs, max_new_tokens=20)
    generate_ids = [
        oid[len(iids):] for oid, iids in zip(generate_ids, inputs.input_ids)
    ]

    gen_text = processor.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    return gen_text

f = build_model_input(model, processor, testdata)
print(f)

model = model.merge_and_unload()
model.save_pretrained('save_path')
processor.save_pretrained('save_path')
