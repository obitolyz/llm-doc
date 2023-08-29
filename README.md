# 模型部署分享
## FastChat 架构 ⭐️26.8k
fastchat 由四个组件组成，分别为
- gradio web server 用于简单的 UI 交互
- Controller 用于管理 `model worker` 的模型名和模型地址
- openai_api server 提供与 **openai** 兼容的 API
- model worker 运行大模型，定时向 **Controller** 发送心跳

![fastchat_framework](images/fastchat_framework.png)

组件的启动顺序是先启动 `Controller` 模块，用于 model-worker 的注册信息。其次是 `model worker`，运行大模型，最后是 `gradio web server` 和 `openai_api server` 。

### model worker 主要参数
`--gpus`: A single GPU like 1 or multiple GPUs like `0,2`

`--max-gpu-memory`: 单个 gpu 最大的显存使用量

`--controller-address`: Controller 地址

`--limit-worker-concurrency`: 防止 OOM

### gradio web server 主要参数
`--model-list-mode`: **once** or **reload**. 加载模型列表一次或者每次都加载

## conversation template

## openai api

## Baichuan-13B Demo
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat", use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan-13B-Chat", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
messages = []
messages.append({"role": "user", "content": "Which moutain is the second highest one in the world?"})
response = model.chat(tokenizer, messages)
print(response)
```

## huggingface files
```
+--- config.json
+--- configuration_baichuan.py
+--- generation_config.json
+--- generation_utils.py
+--- gitattributes.txt
+--- handler.py
+--- modeling_baichuan.py
+--- pytorch_model-00001-of-00003.bin
+--- pytorch_model-00002-of-00003.bin
+--- pytorch_model-00003-of-00003.bin
+--- pytorch_model.bin.index.json
+--- quantizer.py
+--- README.md
+--- requirements.txt
+--- special_tokens_map.json
+--- tokenization_baichuan.py
+--- tokenizer.model
+--- tokenizer_config.json
```
### pytorch_model.bin.index.json
模型参数保存在哪一个 bin 文件中

![](images/pytorch_model.png)

### tokenizer_config.json
```python
tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan-13B-Chat")
```

上面的代码会从 `auto_map.AutoTokenizer[0]` 或 `tokenizer_class` 动态加载到   `transformers.models` 模块上
```json
{
  "auto_map": {
    "AutoTokenizer": [
      "tokenization_baichuan.BaichuanTokenizer",
      null
    ]
  },
  "bos_token": {
    "__type": "AddedToken",
    "content": "<s>",
    "lstrip": false,
    "normalized": true,
    "rstrip": false,
    "single_word": true
  },
  "clean_up_tokenization_spaces": false,
  "eos_token": {
    "__type": "AddedToken",
    "content": "</s>",
    "lstrip": false,
    "normalized": true,
    "rstrip": false,
    "single_word": true
  },
  "model_max_length": 4096,
  "pad_token": {
    "__type": "AddedToken",
    "content": "<unk>",
    "lstrip": false,
    "normalized": true,
    "rstrip": false,
    "single_word": true
  },
  "tokenizer_class": "BaichuanTokenizer",
  "unk_token": {
    "__type": "AddedToken",
    "content": "<unk>",
    "lstrip": false,
    "normalized": true,
    "rstrip": false,
    "single_word": true
  }
}
```
### generation_config.json
```json
{
  "pad_token_id": 0,
  "bos_token_id": 1,
  "eos_token_id": 2,
  "user_token_id": 195,
  "assistant_token_id": 196,
  "max_new_tokens": 2048,
  "temperature": 0.3,
  "top_k": 5,
  "top_p": 0.85,
  "repetition_penalty": 1.1,
  "do_sample": true,
  "transformers_version": "4.29.2"
}
```