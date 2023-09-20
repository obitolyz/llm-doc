# LLM 模型微调以及训练方案

## 训练框架

### DeepSpeed

### Colossal AI

## 推理框架
### vLLM
## modelscope 阿里
## triton-inference-server
## BentoML

## 微调方法以及原理

### instruction tuning

### Freeze
参数冻结，对原始模型部分参数进行冻结操作，仅训练部分参数，减少显存的使用
```python
freeze_module_name = args.freeze_module_name.split(",")
for name, param in model.named_parameters():
	if not any(nd in name for nd in freeze_module_name):
		param.requires_grad = False # 不计算梯度
```

### P-Tuning

### P-Tuning v2

### Lora

## 从 0 开始训练 LLM 模型的步骤

## 框架
简单介绍功能和使用方法
### MindSpore 华为
### KubeDL 阿里
### Paddlepaddle

## 量化
GPTQ

## Tokenizer Training

## 参考文献
1. [大模型训练入门实战](https://techdiylife.github.io/big-model-training/deepspeed/deepspeed-chat.html)
2. [北大硕士RLHF实践，基于DeepSpeed-Chat成功训练上自己的模型](https://zhuanlan.zhihu.com/p/653285736)
3. [大模型LLM-微调经验分享&总结](https://zhuanlan.zhihu.com/p/620885226)
4. [github - ChatGLM-Finetuning](https://github.com/liucongg/ChatGLM-Finetuning)
5. [【LLM】从零开始训练大模型](https://zhuanlan.zhihu.com/p/636270877)
