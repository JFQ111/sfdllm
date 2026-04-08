import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BertModel, BertTokenizer, BertConfig,
    LlamaConfig, LlamaModel, LlamaTokenizer,
    GPT2Config, GPT2Model, GPT2Tokenizer,
)
from peft import LoraConfig, get_peft_model, TaskType


class Normalize(nn.Module):
    def __init__(self, affine=False):
        super().__init__()
        self.affine = affine

    def forward(self, x, mode="norm"):
        if mode == "norm":
            self.mean = torch.mean(x, dim=1, keepdim=True)
            self.std = torch.std(x, dim=1, keepdim=True) + 1e-5
            return (x - self.mean) / self.std
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Conv1d(1, d_model, kernel_size=patch_len, stride=stride)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.proj(x)
        return self.dropout(x.permute(0, 2, 1))


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)
        out = self._attention(target_embedding, source_embedding, value_embedding)
        return self.out_projection(out.reshape(B, L, -1))

    def _attention(self, target, source, value):
        B, L, H, E = target.shape
        scale = 1.0 / sqrt(E)
        scores = torch.einsum("blhe,she->bhls", target, source)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,she->blhe", A, value)


def _compute_lags(x, top_k=5):
    q_fft = torch.fft.rfft(x.float(), dim=-1)
    corr = torch.fft.irfft(q_fft * torch.conj(q_fft), dim=-1)
    _, lags = torch.topk(corr, k=top_k, dim=-1)
    return lags


def _round_val(val, decimals=4):
    return round(float(val), decimals)


def _compute_tf_features(signal: torch.Tensor, sampling_rate: int) -> dict:
    if signal.dim() != 1:
        signal = signal.reshape(-1)
    sr = max(1, int(sampling_rate))
    sig = signal.detach().to(dtype=torch.float64)

    mean_val = torch.mean(sig)
    centered = sig - mean_val
    std_val = torch.std(sig, unbiased=False)
    rms = torch.sqrt(torch.mean(centered ** 2))
    abs_mean = torch.mean(torch.abs(centered))
    peak = torch.max(torch.abs(centered))
    ptp = torch.max(sig) - torch.min(sig)
    crest = peak / (rms + 1e-12)
    impulse = peak / (abs_mean + 1e-12)
    var = torch.var(sig, unbiased=False) + 1e-12
    std_safe = torch.sqrt(var)
    skewness = torch.mean((centered / std_safe) ** 3)
    kurt = torch.mean(centered ** 4) / (var ** 2)

    n = sig.numel()
    window = torch.hann_window(n, device=sig.device, dtype=sig.dtype)
    spectrum = torch.abs(torch.fft.rfft(centered * window))
    freq_count = spectrum.numel()
    freqs = torch.arange(freq_count, device=sig.device, dtype=sig.dtype) * (sr / n)
    energy = torch.sum(spectrum ** 2)

    if torch.all(spectrum == 0):
        dominant_freq = spectral_centroid = bandwidth = torch.tensor(0.0, device=sig.device, dtype=sig.dtype)
    else:
        spec_sum = torch.sum(spectrum) + 1e-12
        dominant_freq = freqs[torch.argmax(spectrum)]
        spectral_centroid = torch.sum(freqs * spectrum) / spec_sum
        bandwidth = torch.sqrt(torch.sum(((freqs - spectral_centroid) ** 2) * spectrum) / spec_sum)

    nyquist = sr / 2.0
    band_defs = [
        ("low_energy", 0.0, nyquist / 3.0),
        ("mid_energy", nyquist / 3.0, 2 * nyquist / 3.0),
        ("high_energy", 2 * nyquist / 3.0, nyquist),
    ]
    band_energy = {}
    for name, lo, hi in band_defs:
        mask = (freqs >= lo) & (freqs < hi)
        band_energy[name] = (torch.sum(spectrum[mask] ** 2) / (energy + 1e-12))

    top_k = min(3, spectrum.numel())
    peak_indices = torch.topk(spectrum, k=top_k, largest=True).indices
    top_peaks = freqs[peak_indices] if top_k > 0 else torch.zeros(0, device=sig.device, dtype=sig.dtype)

    raw = {
        "mean": mean_val, "std": std_val, "rms": rms, "peak_to_peak": ptp,
        "abs_mean": abs_mean, "crest_factor": crest, "impulse_factor": impulse,
        "skewness": skewness, "kurtosis": kurt,
        "dominant_freq_hz": dominant_freq, "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": bandwidth, "energy": energy,
        "top3_peak_freq_hz": top_peaks,
    }
    raw.update(band_energy)

    result = {}
    for k, v in raw.items():
        if isinstance(v, torch.Tensor):
            result[k] = [_round_val(x) for x in v.tolist()] if v.numel() > 1 else _round_val(v.item())
        else:
            result[k] = _round_val(v)
    return result


class SFDLLMBackbone(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.patch_len = configs.patch_len
        self.stride = configs.p_stride
        self.sampling_rate = max(1, int(getattr(configs, "sampling_rate", 1)))
        self.num_patch = int((configs.seq_len - configs.patch_len) / configs.p_stride + 1)
        self.normalize = Normalize(affine=False)
        self.projection_dim = 256

        self.model_type = self._infer_model_type(configs)
        base_model, tokenizer, lora_target, lora_task = self._load_backbone(self.model_type, configs)
        self.tokenizer = tokenizer
        self.d_llm = self._infer_hidden_size(base_model.config)

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token or "[PAD]"

        for param in base_model.parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            task_type=lora_task,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=lora_target,
        )
        self.model = get_peft_model(base_model, peft_config)

        self.word_embedding = self.model.get_input_embeddings().weight
        self.mapping = nn.Linear(self.word_embedding.shape[0], configs.num_mapping)
        self.patch_embedding = PatchEmbedding(configs.d_model, configs.patch_len, configs.p_stride, configs.dropout)
        self.reprogramming = ReprogrammingLayer(configs.d_model, configs.n_heads, configs.d_keys, self.d_llm)

        self.dim_reduction = nn.Conv1d(self.d_llm, configs.redimension, 1)
        self.feature_projection = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(configs.redimension * self.num_patch, self.d_llm),
            nn.ReLU(),
            nn.Linear(self.d_llm, self.projection_dim),
        )
        self.text_projection = nn.Linear(self.d_llm, self.projection_dim)
        self.reconstruction_head = nn.Linear(self.d_llm, configs.patch_len)

    def _infer_model_type(self, configs):
        name = str(getattr(configs, "model_name", "")).lower()
        path = str(getattr(configs, "model_path", "")).lower()
        for key in ("bert", "llama", "gpt2", "qwen"):
            if key in name or key in path:
                return key
        raise ValueError(f"Unsupported model: {configs.model_name}")

    def _infer_hidden_size(self, config):
        for key in ("hidden_size", "n_embd", "d_model"):
            val = getattr(config, key, None)
            if val is not None:
                return int(val)
        raise ValueError("Cannot infer hidden size from model config.")

    def _load_backbone(self, model_type, configs):
        if model_type == "bert":
            cfg = BertConfig.from_pretrained(configs.model_path)
            cfg.output_hidden_states = True
            model = BertModel.from_pretrained(configs.model_path, config=cfg)
            tokenizer = BertTokenizer.from_pretrained(configs.model_path)
            return model, tokenizer, ["query", "value"], TaskType.FEATURE_EXTRACTION
        elif model_type == "llama":
            cfg = LlamaConfig.from_pretrained(configs.model_path)
            cfg.num_hidden_layers = configs.llm_layers
            cfg.output_hidden_states = True
            model = LlamaModel.from_pretrained(configs.model_path, config=cfg, attn_implementation="eager")
            tokenizer = LlamaTokenizer.from_pretrained(configs.model_path)
            tokenizer.padding_side = "left"
            return model, tokenizer, ["q_proj", "k_proj", "v_proj", "o_proj"], TaskType.FEATURE_EXTRACTION
        elif model_type == "qwen":
            model = AutoModelForCausalLM.from_pretrained(
                configs.model_path, device_map="auto", torch_dtype=torch.float32, output_hidden_states=True
            )
            tokenizer = AutoTokenizer.from_pretrained(configs.model_path)
            tokenizer.padding_side = "left"
            return model, tokenizer, ["q_proj", "k_proj", "v_proj", "o_proj"], TaskType.CAUSAL_LM
        elif model_type == "gpt2":
            cfg = GPT2Config.from_pretrained(configs.model_path)
            cfg.output_hidden_states = True
            model = GPT2Model.from_pretrained(configs.model_path, config=cfg)
            tokenizer = GPT2Tokenizer.from_pretrained(configs.model_path)
            tokenizer.padding_side = "left"
            return model, tokenizer, ["c_attn", "c_proj"], TaskType.FEATURE_EXTRACTION
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _unwrap_base(self):
        if hasattr(self.model, "base_model"):
            base = self.model.base_model
            return base.model if hasattr(base, "model") else base
        return getattr(self.model, "model", self.model)

    def get_text_embedding(self, text_list):
        device = self.model.device
        tokens = self.tokenizer(
            text_list, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        backbone = self._unwrap_base()
        with torch.no_grad():
            outputs = backbone(**tokens, output_hidden_states=True)
        if "bert" in self.configs.model_name.lower():
            feature = outputs.last_hidden_state[:, 0, :]
        else:
            feature = outputs.hidden_states[-1][:, -1, :]
        return self.text_projection(feature)

    def encode_signal(self, x):
        B, L = x.shape
        x_norm = self.normalize(x, "norm")

        min_vals = torch.min(x_norm, dim=1)[0]
        max_vals = torch.max(x_norm, dim=1)[0]
        medians = torch.median(x_norm, dim=1).values
        lags = _compute_lags(x_norm)
        trends = x_norm.diff(dim=1).sum(dim=1)

        prompts = []
        for b in range(B):
            tf = _compute_tf_features(x_norm[b], self.sampling_rate)
            prompt = (
                f"<|start_prompt|>Dataset: {self.configs.desc}. "
                f"Task: Bearing fault diagnosis from vibration signals. "
                f"Statistics: min {_round_val(min_vals[b])}, max {_round_val(max_vals[b])}, "
                f"median {_round_val(medians[b])}, rms {tf['rms']}, p2p {tf['peak_to_peak']}, "
                f"abs_mean {tf['abs_mean']}, crest {tf['crest_factor']}, impulse {tf['impulse_factor']}, "
                f"skewness {tf['skewness']}, kurtosis {tf['kurtosis']}, "
                f"dominant_freq {tf['dominant_freq_hz']}Hz, centroid {tf['spectral_centroid_hz']}Hz, "
                f"bandwidth {tf['spectral_bandwidth_hz']}Hz, "
                f"band_energy L/M/H {tf['low_energy']}/{tf['mid_energy']}/{tf['high_energy']}, "
                f"top3_peaks {tf['top3_peak_freq_hz']}Hz, "
                f"trend {'up' if trends[b] > 0 else 'down'}, top_lags {lags[b].tolist()}<|<end_prompt>|>"
            )
            prompts.append(prompt)

        prompt_tokens = self.tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(x.device)
        prompt_emb = self.model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_mask = prompt_tokens.attention_mask

        source_emb = self.mapping(self.word_embedding.permute(1, 0)).permute(1, 0)
        x_enc = self.patch_embedding(x_norm)
        x_enc = self.reprogramming(x_enc, source_emb, source_emb)

        llm_in = torch.cat([prompt_emb, x_enc], dim=1)
        llm_mask = torch.cat(
            [prompt_mask, torch.ones((B, x_enc.shape[1]), device=x.device, dtype=prompt_mask.dtype)],
            dim=1,
        )

        if "bert" in self.configs.model_name.lower():
            llm_out = self.model(inputs_embeds=llm_in, attention_mask=llm_mask).last_hidden_state
        else:
            llm_out = self.model(
                inputs_embeds=llm_in, attention_mask=llm_mask, output_hidden_states=True
            ).hidden_states[-1]

        llm_out_patch = llm_out[:, -self.num_patch:, :]

        feat_reduced = self.dim_reduction(llm_out_patch.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        signal_emb = self.feature_projection(feat_reduced)
        rec_patches = self.reconstruction_head(llm_out_patch)

        return signal_emb, rec_patches, x_norm


class SemanticFaultAligner(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.backbone = SFDLLMBackbone(configs)
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        self.lambda_rec = 0.1

    def _patchify(self, x, patch_len, stride):
        return x.unsqueeze(1).unfold(dimension=-1, size=patch_len, step=stride).squeeze(1)

    def forward(self, x, text_descriptions, labels=None):
        signal_emb, rec_patches, x_norm = self.backbone.encode_signal(x)
        signal_emb = F.normalize(signal_emb, p=2, dim=1)

        text_emb = self.backbone.get_text_embedding(text_descriptions)
        text_emb = F.normalize(text_emb, p=2, dim=1)

        logits = torch.matmul(signal_emb, text_emb.t()) / self.temperature

        loss_contrastive = F.cross_entropy(logits, labels) if labels is not None else torch.tensor(0.0)

        gt_patches = self._patchify(x_norm, self.backbone.patch_len, self.backbone.stride)
        min_len = min(rec_patches.shape[1], gt_patches.shape[1])
        loss_rec = F.mse_loss(rec_patches[:, :min_len], gt_patches[:, :min_len])

        total_loss = loss_contrastive + self.lambda_rec * loss_rec
        return total_loss, logits, loss_rec

    def predict(self, x, candidate_descriptions):
        self.eval()
        with torch.no_grad():
            signal_emb, _, _ = self.backbone.encode_signal(x)
            signal_emb = F.normalize(signal_emb, p=2, dim=1)
            text_emb = self.backbone.get_text_embedding(candidate_descriptions)
            text_emb = F.normalize(text_emb, p=2, dim=1)
            similarity = torch.matmul(signal_emb, text_emb.t())
            predictions = torch.argmax(similarity, dim=1)
        return predictions, similarity
