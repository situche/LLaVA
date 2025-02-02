from transformers import LlavaForConditionalGeneration, LlavaConfig
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoProcessor

# llava 初始化
clip_model_name_or_path = '/home/gpu/openai/clip-vit-large-patch14-336'
qwen_model_name_or_path = '/home/gpu/Qwen1.5-4B-Chat'

clip_model = AutoModel.from_pretrained(clip_model_name_or_path, device_map='cuda:0')
llm_model = AutoModelForCausalLM.from_pretrained(qwen_model_name_or_path, device_map='cuda:0')
llm_tokenizer = AutoTokenizer.from_pretrained(qwen_model_name_or_path)

# 初始化llava model
vision_config = clip_model.vision_model.config
text_config = llm_model.config
configuration = LlavaConfig(vision_config, text_config)
model = LlavaForConditionalGeneration(configuration)

model.vision_tower.vision_model.embeddings

clip_model.vision_model

# 复制模型权重
model.vision_tower.vision_model = clip_model.vision_model
model.language_model = llm_model
llm_model.model.embed_tokens.weight.data[:, :2]

model.language_model.model.embed_tokens.weight.data[:, :2]

# 复制pad_token_id
model.config.pad_token_id
model.config.pad_token_id = llm_tokenizer.pad_token_id

# 复制image_token_index
model.config.image_token_index
model.config.image_token_index = llm_tokenizer.encode('<image>')[0]

# 保存模型
model.save_pretrained('/data/wangbin/week4/model001')

# 保存processor
llm_tokenizer.save_pretrained('/data/wangbin/week4/model001')
autoprocessor = AutoProcessor.from_pretrained(clip_model_name_or_path)
autoprocessor.save_pretrained('/data/wangbin/week4/model002')
