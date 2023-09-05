# LLM 模型微调以及训练方案

## 训练框架

### DeepSpeed

### Colossal AI

## 微调方法以及原理

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


## 参考文献
1. [大模型训练入门实战](https://techdiylife.github.io/big-model-training/deepspeed/deepspeed-chat.html)
2. [北大硕士RLHF实践，基于DeepSpeed-Chat成功训练上自己的模型](https://zhuanlan.zhihu.com/p/653285736)
