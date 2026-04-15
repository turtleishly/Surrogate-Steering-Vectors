from types import SimpleNamespace
from typing import Optional, Tuple

import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..config import DEFAULT_READER_MODEL_NAME

_hf_reader_model = None
_hf_reader_tokenizer = None
_hf_reader_model_name = None


class SurrogateHFAdapter:
    """Minimal compatibility layer for notebook code expecting TL-like methods."""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self._device = next(model.parameters()).device
        self.cfg = SimpleNamespace(
            device=str(self._device),
            n_ctx=int(getattr(model.config, "max_position_embeddings", 4096)),
            n_layers=int(getattr(model.config, "num_hidden_layers", 0)),
        )

    def to_tokens(self, text: str, prepend_bos: bool = False):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if prepend_bos and self.tokenizer.bos_token_id is not None:
            ids = [self.tokenizer.bos_token_id] + ids
        if len(ids) == 0 and self.tokenizer.eos_token_id is not None:
            ids = [self.tokenizer.eos_token_id]
        return t.tensor([ids], dtype=t.long, device=self._device)

    def to_string(self, token_ids):
        if isinstance(token_ids, t.Tensor):
            token_ids = token_ids.detach().to("cpu").tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def __call__(self, token_tensor: t.Tensor):
        with t.inference_mode():
            out = self.model(input_ids=token_tensor.to(self._device), use_cache=False)
        return out.logits


def get_hf_reader(model_name: str = DEFAULT_READER_MODEL_NAME):
    """Lazy-load (or reuse) HF model/tokenizer."""
    global _hf_reader_model, _hf_reader_tokenizer, _hf_reader_model_name

    if (
        _hf_reader_model is not None
        and _hf_reader_tokenizer is not None
        and _hf_reader_model_name == model_name
    ):
        return _hf_reader_model, _hf_reader_tokenizer

    _hf_reader_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    _hf_reader_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=t.bfloat16 if t.cuda.is_available() else t.float32,
        device_map="auto" if t.cuda.is_available() else None,
        trust_remote_code=True,
    )
    _hf_reader_model.eval()

    if _hf_reader_tokenizer.pad_token_id is None and _hf_reader_tokenizer.eos_token_id is not None:
        _hf_reader_tokenizer.pad_token = _hf_reader_tokenizer.eos_token

    _hf_reader_model_name = model_name
    return _hf_reader_model, _hf_reader_tokenizer


def build_surrogate_model(model_name: str = DEFAULT_READER_MODEL_NAME) -> Tuple[SurrogateHFAdapter, object, object]:
    hf_model, hf_tokenizer = get_hf_reader(model_name=model_name)
    return SurrogateHFAdapter(hf_model, hf_tokenizer), hf_model, hf_tokenizer
