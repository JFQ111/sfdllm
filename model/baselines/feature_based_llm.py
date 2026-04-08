import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from scipy import signal
import json
import re
from typing import List, Optional

def extract_class_names(dataset) -> Optional[List[str]]:
    candidates: List[List[str]] = []
    if hasattr(dataset, "get_class_names"):
        candidates.append(dataset.get_class_names())
    if hasattr(dataset, "class_names"):
        candidates.append(dataset.class_names)
    if hasattr(dataset, "dataset"):
        sub_names = extract_class_names(dataset.dataset)
        if sub_names:
            candidates.append(sub_names)
    if hasattr(dataset, "datasets"):
        for ds in dataset.datasets:
            sub_names = extract_class_names(ds)
            if sub_names:
                candidates.append(sub_names)
    if not candidates:
        return None
    return max(candidates, key=len)


def extract_sampling_rate(dataset) -> Optional[float]:
    if hasattr(dataset, "target_sampling_rate"):
        return dataset.target_sampling_rate
    if hasattr(dataset, "sampling_rate"):
        return dataset.sampling_rate
    if hasattr(dataset, "args") and hasattr(dataset.args, "sampling_rate"):
        return dataset.args.sampling_rate
    if hasattr(dataset, "dataset"):
        return extract_sampling_rate(dataset.dataset)
    if hasattr(dataset, "datasets"):
        for ds in dataset.datasets:
            rate = extract_sampling_rate(ds)
            if rate is not None:
                return rate
    return None

class FeatureExtractor:
    """
    基于论文Table 1的特征提取器
    提取12个时域特征和12个频域特征
    """

    def __init__(self, sampling_rate=50000):
        self.sampling_rate = sampling_rate

    def extract_time_domain_features(self, x):
        """提取时域特征"""
        features = {}
        N = len(x)

        # p1: Mean value
        features['mean_value'] = np.mean(x)

        # p2: Standard deviation
        features['std_deviation'] = np.std(x, ddof=1)

        # p3: Square root amplitude
        features['square_root_amplitude'] = (np.mean(np.sqrt(np.abs(x)))) ** 2

        # p4: Absolute mean value
        features['abs_mean_value'] = np.mean(np.abs(x))

        # p5: Peak value
        features['peak_value'] = np.max(np.abs(x))

        # p6: Skewness
        features['skewness'] = np.mean(x ** 3)

        # p7: Kurtosis
        features['kurtosis'] = np.mean(x ** 4)

        # p8: Variance
        features['variance'] = np.var(x)

        # p9: Kurtosis index
        features['kurtosis_index'] = features['kurtosis'] / (features['skewness'] ** 2) if features[
                                                                                               'skewness'] != 0 else 0

        # p10: Peak index
        features['peak_index'] = features['peak_value'] / features['std_deviation'] if features[
                                                                                           'std_deviation'] != 0 else 0

        # p11: Waveform index
        features['waveform_index'] = features['std_deviation'] / features['abs_mean_value'] if features[
                                                                                                   'abs_mean_value'] != 0 else 0

        # p12: Pulse index
        features['pulse_index'] = features['peak_value'] / features['abs_mean_value'] if features[
                                                                                             'abs_mean_value'] != 0 else 0

        return features

    def extract_frequency_domain_features(self, x):
        """提取频域特征"""
        features = {}

        # 计算功率谱密度
        freqs, psd = signal.welch(x, fs=self.sampling_rate, nperseg=min(256, len(x) // 4))

        # 移除DC分量
        freqs = freqs[1:]
        psd = psd[1:]

        if len(psd) == 0:
            return {f'freq_feature_{i}': 0.0 for i in range(12)}

        K = len(psd)

        # p13: Frequency mean value
        features['freq_mean_value'] = np.sum(psd) / K

        # p14: Frequency variance
        freq_mean = features['freq_mean_value']
        features['freq_variance'] = np.sum((psd - freq_mean) ** 2) / (K - 1) if K > 1 else 0

        # p15: Frequency skewness
        features['freq_skewness'] = np.sum((psd - freq_mean) ** 3) / (K * features['freq_variance'] ** 1.5) if features[
                                                                                                                   'freq_variance'] > 0 else 0

        # p16: Frequency kurtosis
        features['freq_kurtosis'] = np.sum((psd - freq_mean) ** 4) / (K * features['freq_variance'] ** 2) if features[
                                                                                                                 'freq_variance'] > 0 else 0

        # p17: Gravity frequency
        features['gravity_freq'] = np.sum(freqs * psd) / np.sum(psd) if np.sum(psd) > 0 else 0

        # p18: Frequency standard deviation
        features['freq_std'] = np.sqrt(
            np.sum((freqs - features['gravity_freq']) ** 2 * psd) / (K * np.sum(psd))) if np.sum(psd) > 0 else 0

        # p19: Frequency root mean square
        features['freq_rms'] = np.sqrt(np.sum(freqs ** 2 * psd) / np.sum(psd)) if np.sum(psd) > 0 else 0

        # p20: Average frequency
        features['avg_freq'] = np.sqrt(np.sum(freqs ** 4 * psd) / np.sum(freqs ** 2 * psd)) if np.sum(
            freqs ** 2 * psd) > 0 else 0

        # p21: Regularity degree
        features['regularity_degree'] = np.sum(freqs ** 2 * psd) / (
                    np.sum(psd) * np.sqrt(np.sum(freqs ** 4 * psd))) if np.sum(freqs ** 4 * psd) > 0 else 0

        # p22: Variation parameter
        features['variation_param'] = features['freq_std'] / features['gravity_freq'] if features[
                                                                                             'gravity_freq'] > 0 else 0

        # p23: Eighth-order moment
        features['eighth_moment'] = np.sum((freqs - features['gravity_freq']) ** 3 * psd) / (
                    K * features['freq_std'] ** 3) if features['freq_std'] > 0 else 0

        # p24: Sixteenth-order moment
        features['sixteenth_moment'] = np.sum((freqs - features['gravity_freq']) ** 4 * psd) / (
                    K * features['freq_std'] ** 4) if features['freq_std'] ** 4 > 0 else 0

        return features

    def extract_all_features(self, x):
        """提取所有特征"""
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        time_features = self.extract_time_domain_features(x)
        freq_features = self.extract_frequency_domain_features(x)

        all_features = {**time_features, **freq_features}
        return all_features


class FeatureBasedLLM(nn.Module):
    """
    基于特征的 LLM 轴承故障诊断模型 - 生成式版本 (Generative SFT)
    完全复现论文 Figure 2 和 Section 3.1.2 的 Prompt Engineering 和 SFT 方法
    """
    def __init__(self, configs):
        super(FeatureBasedLLM, self).__init__()
        self.configs = configs
        self.feature_extractor = FeatureExtractor(sampling_rate=configs.sampling_rate)
        self.class_names: List[str] = []
        self.label_map = {}
        self.class_aliases: List[List[str]] = []
        self._init_class_names()

        # 1. 初始化 LLM 模型 (Causal LM)
        self._init_llm_model()

        # 2. 配置 LoRA (Section 3.1.2: "Fine-tuning methods based on LoRA... were employed") [cite: 28, 241]
        self._setup_lora()

    def _init_class_names(self):
        class_names = getattr(self.configs, "class_names", None)
        if class_names:
            self.set_class_names(class_names)
            return
        num_classes = getattr(self.configs, "num_classes", None)
        if num_classes is not None:
            self.set_class_names([f"class_{i}" for i in range(num_classes)])
            return
        self.set_class_names(["normal", "inner", "outer", "ball"])

    def set_class_names(self, class_names: Optional[List[str]]) -> None:
        if not class_names:
            self.class_names = []
            self.label_map = {}
            self.class_aliases = []
            return
        self.class_names = [str(name) for name in class_names]
        self.label_map = {idx: name for idx, name in enumerate(self.class_names)}
        self.class_aliases = self._build_class_aliases(self.class_names)

    def set_sampling_rate(self, sampling_rate: Optional[float]) -> None:
        if sampling_rate is None:
            return
        self.feature_extractor.sampling_rate = sampling_rate

    @staticmethod
    def _normalize_label_name(name: str) -> str:
        return re.sub(r"\s+", " ", str(name).lower().replace("_", " ").strip())

    @staticmethod
    def _compact_label_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", str(name).lower())

    @staticmethod
    def _infer_base_label(name: str) -> Optional[str]:
        lower = str(name).lower()
        if "normal" in lower or "healthy" in lower:
            return "normal"
        if "inner" in lower:
            return "inner"
        if "outer" in lower:
            return "outer"
        if "ball" in lower or "rolling" in lower or "roller" in lower:
            return "ball"
        return None

    def _build_class_aliases(self, class_names: List[str]) -> List[List[str]]:
        base_counts = {}
        base_list = []
        for name in class_names:
            base = self._infer_base_label(name)
            base_list.append(base)
            if base:
                base_counts[base] = base_counts.get(base, 0) + 1

        aliases = []
        for name, base in zip(class_names, base_list):
            tokens = []
            raw = str(name).lower()
            normalized = self._normalize_label_name(name)
            compact = self._compact_label_name(name)
            if raw and raw not in tokens:
                tokens.append(raw)
            if normalized:
                tokens.append(normalized)
            if compact and compact not in tokens:
                tokens.append(compact)
            if base and base_counts.get(base, 0) == 1 and base not in tokens:
                tokens.append(base)
            aliases.append(tokens)
        return aliases

    def _find_class_index(self, keywords: List[str]) -> Optional[int]:
        for idx, name in enumerate(self.class_names):
            name_lower = str(name).lower()
            if any(keyword in name_lower for keyword in keywords):
                return idx
        return None

    def _match_class_from_response(self, response: str) -> int:
        response_lower = response.lower()
        matches = []
        for idx, aliases in enumerate(self.class_aliases):
            for token in aliases:
                if token and token in response_lower:
                    matches.append((len(token), idx))
        if matches:
            matches.sort(key=lambda item: (-item[0], item[1]))
            return matches[0][1]

        for keywords in (["normal", "healthy"], ["inner"], ["outer"], ["ball", "rolling", "roller"]):
            if any(keyword in response_lower for keyword in keywords):
                idx = self._find_class_index(keywords)
                if idx is not None:
                    return idx
        return -1

    def _init_llm_model(self):
        """初始化生成式 LLM"""
        print(f"Loading Causal LM: {self.configs.model_name}...")
        
        # 统一使用 AutoModelForCausalLM，不再使用 BertModel (因为是生成任务)
        # 论文使用的是 ChatGLM2-6B [cite: 476]
        model_path = getattr(self.configs, "model_path", None) or "../weights_of_models/qwen2.5_0.5b"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            padding_side="left" # 推理时生成需要 left padding
        )
        
        # 设置 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 量化配置 (QLoRA) [cite: 241]
        bnb_config = None
        if getattr(self.configs, 'quantize', False):
            bnb_config = BitsAndBytesConfig(
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                load_in_4bit=True
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
        
        # 开启梯度检查点以节省显存
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()

    def _setup_lora(self):
        """配置 LoRA"""
        # 针对不同模型选择 Target Modules
        if "chatglm" in self.configs.model_name.lower():
            target_modules = ["query_key_value"]
        elif "qwen" in self.configs.model_name.lower():
            target_modules = ["q_proj", "v_proj"]
        elif "gpt2" in self.configs.model_name.lower():
            target_modules = ["c_attn"]
        elif "llama" in self.configs.model_name.lower():
            target_modules = ["q_proj", "v_proj"]
        else:
            target_modules = ["q_proj", "v_proj"] # Default

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, # 关键：任务类型必须是 CAUSAL_LM
            inference_mode=False,
            r=8, # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=target_modules
        )
        
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

    def build_prompt_input(self, features_dict):
        """
        构建 Input 部分的文本，参考论文 Figure 2 [cite: 246, 247]
        """
        # 构建特征描述文本
        desc_list = []
        # 按论文 Table 1 顺序或字典顺序遍历特征
        for k, v in features_dict.items():
            readable_key = k.replace('_', ' ')
            desc_list.append(f"The {readable_key} is {v:.4f}")
        
        feature_str = ", ".join(desc_list)
        label_hint = ""
        if self.class_names:
            label_hint = f"Candidate labels: {', '.join(self.class_names)}.\n"

        # 构造符合 ChatGLM 或通用 LLM 的 Prompt
        # 论文 Figure 2 格式:
        # "instruction": ...
        # "input": "The time-domain mean... The frequency-domain mean..."
        # "output": ...
        prompt = (
            f"Instruction: You are a bearing fault diagnosis expert. Based on the following features, you need to conduct fault diagnosis.\n"
            f"Input: {feature_str}.\n"
            f"{label_hint}"
            f"Output:"
        )
        return prompt

    def forward(self, x, labels=None):
        """
        训练时的前向传播
        x: [batch_size, seq_len] 原始振动信号
        labels: [batch_size] 对应的故障类别索引 (0-3)
        """
        # 1. 提取特征
        batch_prompts = []
        batch_full_texts = []
        
        device = x.device
        
        for i in range(len(x)):
            signal_data = x[i]
            y = labels[i].item() if labels is not None else None
            signal_data = signal_data.reshape(-1)
            
            # 提取数值特征
            feats = self.feature_extractor.extract_all_features(signal_data)
            
            # 构建 Prompt (Instruction + Input)
            prompt = self.build_prompt_input(feats)
            batch_prompts.append(prompt)
            
            # 如果是训练阶段，构建完整的 (Prompt + Answer)
            if y is not None:
                label_text = self.label_map.get(y, f"class_{y}")
                diagnosis_str = f" The diagnosis result is {label_text}."
                full_text = prompt + diagnosis_str + self.tokenizer.eos_token
                batch_full_texts.append(full_text)

        # 2. Tokenize & Masking (SFT 的核心)
        # 我们只希望计算 Output 部分的 Loss，Mask 掉 Instruction 和 Input 部分
        if labels is not None:
            self.tokenizer.padding_side = "right" # 训练时通常 right padding
            
            # 编码完整文本
            encodings = self.tokenizer(
                batch_full_texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(device)
            
            input_ids = encodings.input_ids
            attention_mask = encodings.attention_mask
            target_labels = input_ids.clone()
            
            # Mask 掉 Prompt 部分 (设为 -100)
            for i, prompt in enumerate(batch_prompts):
                prompt_len = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
                # 注意：这里是一个简化的 Mask 逻辑，实际可能需要更精细的处理特殊 token
                # 这里的假设是 prompt 出现在序列开头
                target_labels[i, :prompt_len] = -100 
                
                # 同时也 mask 掉 padding
                target_labels[i][attention_mask[i] == 0] = -100

            # 3. 计算 Causal LM Loss
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=target_labels # 传入 labels，模型会自动计算 CrossEntropyLoss
            )
            return outputs.loss
            
        else:
            # 推理阶段不在这里进行，使用 generate_diagnosis
            return None

    @torch.no_grad()
    def generate_diagnosis(self, x):
        """
        推理/测试阶段
        x: [batch_size, seq_len]
        返回: [batch_size] 预测的类别索引
        """
        self.model.eval()
        predictions = []
        
        for i in range(len(x)):
            signal_data = x[i]
            signal_data = signal_data.reshape(-1)
            feats = self.feature_extractor.extract_all_features(signal_data)
            prompt = self.build_prompt_input(feats)
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(x.device)
            
            # 生成文本
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=20, 
                do_sample=False # 确定性生成
            )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 截取 Output 之后的部分
            if "Output:" in response:
                response = response.split("Output:")[-1]
            
            # 解析文本回类别索引
            pred_idx = self._match_class_from_response(response)
            
            predictions.append(pred_idx)
            
        return torch.tensor(predictions).to(x.device)

# --------------------------------------------------------------------------
# 对应的训练器也需要调整，因为模型现在自己计算 Loss，不需要外部 Criterion
# --------------------------------------------------------------------------

class FeatureLLMTrainer:
    def __init__(self, model, configs):
        self.model = model
        self.configs = configs
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=configs.learning_rate)
        self._synced = False

    @staticmethod
    def _unpack_batch(batch):
        if isinstance(batch, dict):
            return batch.get("data"), batch.get("label")
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError("Unsupported batch format")

    def _sync_from_loader(self, loader):
        if self._synced:
            return
        class_names = extract_class_names(loader.dataset)
        if class_names:
            self.model.set_class_names(class_names)
        sampling_rate = extract_sampling_rate(loader.dataset)
        if sampling_rate:
            self.model.set_sampling_rate(sampling_rate)
        self._synced = True
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        self._sync_from_loader(train_loader)
        for batch in train_loader:
            batch_data, batch_labels = self._unpack_batch(batch)
            batch_data = batch_data.to(self.model.model.device)
            batch_labels = batch_labels.to(self.model.model.device).long().view(-1)
            
            self.optimizer.zero_grad()
            
            # forward 直接返回 loss
            loss = self.model(batch_data, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)

    def evaluate(self, test_loader):
        self.model.eval()
        correct = 0
        total = 0

        self._sync_from_loader(test_loader)
        for batch in test_loader:
            batch_data, batch_labels = self._unpack_batch(batch)
            batch_data = batch_data.to(self.model.model.device)
            batch_labels = batch_labels.to(self.model.model.device).long().view(-1)
            
            # 使用 generate 进行推理
            preds = self.model.generate_diagnosis(batch_data)
            
            total += batch_labels.size(0)
            correct += (preds == batch_labels).sum().item()
            
        return 100 * correct / total
