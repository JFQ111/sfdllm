import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from transformers import AutoConfig, AutoModel, AutoTokenizer


class PatchEmbedding(nn.Module):
    """
    论文中的Patching方法 - 严格按照论文实现
    将时序数据转换为patch embedding
    """
    def __init__(self,patch_len, stride, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        
        # 论文中使用1D卷积进行value embedding
        self.value_embedding = nn.Conv1d(
            in_channels=1, 
            out_channels=d_model,
            kernel_size=patch_len, 
            stride=stride,
            padding=0,
            bias=False
        )
        
    def forward(self, x):
        """
        x: [batch_size, seq_len]
        返回: [batch_size, n_patches, d_model]
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # [batch_size, 1, seq_len]
        
        # Value embedding through 1D convolution
        x = self.value_embedding(x)  # [batch_size, d_model, n_patches]
        x = x.transpose(1, 2)  # [batch_size, n_patches, d_model]
        
        return x


class PositionalEmbedding(nn.Module):
    """
    论文中的Position Embedding - 标准正弦位置编码
    """
    def __init__(self, d_model, max_len=10000):
        super(PositionalEmbedding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model]
        """
        return x + self.pe[:, :x.size(1)]


class DataBasedLLM(nn.Module):
    """
    严格按照论文Figure 4实现的Data-based LLM模型
    """
    def __init__(self, configs):
        super(DataBasedLLM, self).__init__()
        self.configs = configs

        self.patch_stride = getattr(configs, "p_stride", None)
        if self.patch_stride is None:
            self.patch_stride = getattr(configs, "stride", None)
        if self.patch_stride is None:
            raise ValueError("configs.p_stride or configs.stride is required")

        # 计算patch数量
        self.num_patches = (configs.seq_len - configs.patch_len) // self.patch_stride + 1
        
        # 1. 初始化基础模型
        self._init_backbone_model()
        
        # 2. 实例标准化 - 论文Figure 4中的Instance Norm
        self.instance_norm = nn.InstanceNorm1d(1, affine=True)
        
        # 3. Patch Embedding - 论文Figure 4中的Value embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=configs.patch_len,
            stride=self.patch_stride,
            d_model=self.d_model
        )
        
        # 4. Position Embedding - 论文Figure 4中的Position embedding
        self.use_custom_pos = getattr(configs, "use_custom_pos", False)
        self.pos_embedding = (
            PositionalEmbedding(self.d_model, max_len=max(512, self.num_patches + 1))
            if self.use_custom_pos
            else None
        )
        
        # 5. 按论文方式冻结参数
        self._freeze_layers_paper_way()
        
        # 6. 最终分类层 - 论文Figure 4中的Linear
        self.final_layer_norm = nn.LayerNorm(self.d_model)
        self.classifier = nn.Linear(self.d_model, configs.num_classes)
        
        # 7. Dropout
        self.dropout = nn.Dropout(configs.dropout)
        
    def _init_backbone_model(self):
        """
        初始化基础模型
        """
        model_path = getattr(self.configs, "gpt2_path", None)
        if model_path is None:
            model_path = getattr(self.configs, "model_path", None)
        if model_path is None:
            model_path = "../weights_of_models/qwen2.5_0.5b"

        self.backbone_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        self.backbone_config.output_hidden_states = True
        
        self.backbone = AutoModel.from_pretrained(
            model_path,
            config=self.backbone_config,
            trust_remote_code=True
        )
        self.d_model = getattr(self.backbone_config, "hidden_size", None) or getattr(self.backbone_config, "n_embd")
        if self.d_model is None:
            raise ValueError("Cannot infer hidden size from backbone config")
        
        # 获取tokenizer（虽然我们不直接使用，但保持一致性）
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _freeze_layers_paper_way(self):
        """
        严格按照论文3.2.3节的方式冻结参数：
        "we froze the multi-head attention and FFN layers, and only trained 
        the layer norm and position embedding layers"
        """
        for name, param in self.backbone.named_parameters():
            if any(x in name for x in ['attn', 'mlp']):  # 冻结attention和MLP(FFN)
                param.requires_grad = False
            elif any(x in name for x in ['ln_', 'layernorm', 'norm', 'wpe']):  # 训练layer norm和position embedding
                param.requires_grad = True
            else:
                param.requires_grad = False  # 其他默认冻结
                
        # 输出可训练参数统计
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"Backbone total parameters: {total_params:,}")
        print(f"Backbone trainable parameters: {trainable_params:,}")
        print(f"Backbone trainable ratio: {trainable_params/total_params*100:.2f}%")
    
    def forward(self, x):
        """
        严格按照论文Figure 4的流程：
        Vibration signal -> Instance Norm -> Patching -> 
        Value embedding + Position embedding -> Pretrained LLM -> 
        Add & Layer Norm (×6) -> Layer Norm -> Linear
        """
        # 确保输入格式正确
        if len(x.shape) == 3:
            x = x.squeeze(1)  # [batch_size, seq_len]
        
        # 1. Instance Normalization - 论文Figure 4第一步
        x_norm = self.instance_norm(x.unsqueeze(1)).squeeze(1)
        
        # 2. Patching - 论文Figure 4第二步
        patches = self.patch_embedding(x_norm)  # [batch_size, n_patches, d_model]
        
        # 3. Position Embedding - 论文Figure 4第三步
        if self.pos_embedding is not None:
            patches = self.pos_embedding(patches)
        
        # 4. 通过GPT-2 - 论文Figure 4的Pretrained LLM部分
        # GPT-2不需要attention mask，因为我们处理的是嵌入向量
        outputs = self.backbone(inputs_embeds=patches)
        hidden_states = outputs.last_hidden_state  # [batch_size, n_patches, d_model]
        
        # 5. 按论文方式进行特征聚合 - 使用平均池化
        # 论文中提到使用mean pooling
        pooled_features = hidden_states.mean(dim=1)  # [batch_size, d_model]
        
        # 6. Final Layer Norm - 论文Figure 4中的Layer Norm
        pooled_features = self.final_layer_norm(pooled_features)
        
        # 7. Dropout
        pooled_features = self.dropout(pooled_features)
        
        # 8. Final Linear Classification - 论文Figure 4中的Linear
        logits = self.classifier(pooled_features)
        
        return logits
