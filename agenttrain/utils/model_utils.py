from importlib.util import find_spec
from typing import Any, Dict, Optional, Tuple, Union

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None

def get_model(
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    加载模型，如果本地有 liger_kernel 则使用 Liger 优化内核。
    支持从自定义 cache_dir 中读取/下载模型。
    """
    if model_kwargs is None:
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            use_cache=False,
        )
    if is_liger_available():
        print("Using Liger kernel")
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore
        return AutoLigerKernelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )
    else:
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **model_kwargs,
        )

def get_tokenizer(
    model_name: str,
    cache_dir: Optional[str] = None,
) -> Any:
    """
    加载 tokenizer，强制要求带有 chat_template 属性。
    支持从自定义 cache_dir 中读取/下载 tokenizer。
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
    )
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(
            f"Tokenizer for model {model_name} does not have chat_template attribute. "
            "请提供包含 chat_template 的 tokenizer，或者使用带有 `-Instruct` 后缀的模型名。"
        )
    return tokenizer

def get_model_and_tokenizer(
    model_name: str,
    model_kwargs: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    一次性加载模型和 tokenizer，均可指定 cache_dir。
    """
    model = get_model(model_name, model_kwargs=model_kwargs, cache_dir=cache_dir)
    tokenizer = get_tokenizer(model_name, cache_dir=cache_dir)
    return model, tokenizer
