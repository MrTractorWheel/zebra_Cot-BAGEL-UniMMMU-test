# -*- coding: utf-8 -*-
"""
Bagel-Zebra-CoT backend for Uni-MMMU sampling.

Wraps InterleaveInferencer from the Bagel-Zebra-CoT repo
(https://github.com/multimodal-reasoning-lab/Bagel-Zebra-CoT) and exposes the
two primitives the Uni-MMMU sampling protocol needs:

    backend.generate_text(ctx, prompt_suffix=..., max_tokens=...)   -> str
    backend.generate_image(ctx, out_path=..., prompt_suffix=...)    -> str (saved path)

A "context" is a list of ("text", str) / ("image", path-or-PIL) tuples that is
replayed into the model on every call (same stateless protocol as the official
sample_code_example scripts).

Model loading is a faithful port of infz_bf16.py from the Bagel-Zebra-CoT repo,
parameterized for a single-GPU (e.g. 1x RTX 6000 Pro Blackwell 96GB) setup.
"""

import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from PIL import Image

# System prompt used during Zebra-CoT interleaved training/inference (from infz_bf16.py).
INTERLEAVED_SYSTEM_PROMPT = (
    "You are an AI reasoning assistant capable of step-by-step interleaved text "
    "and visual chain of thought. Think step by step and use visual aids to "
    "enhance your problem-solving. Provide your final conclusion clearly in the "
    'format of "Final Answer: "'
)

ContextItem = Tuple[str, object]  # ("text", str) | ("image", str|Path|PIL.Image)


def add_text(ctx: List[ContextItem], text: str) -> None:
    ctx.append(("text", text))


def add_image_path(ctx: List[ContextItem], path, mime: str = "image/png") -> None:
    ctx.append(("image", path))


def _clean_text(raw: str) -> str:
    """Strip chat special tokens from the decoded output."""
    if "<|im_start|>" in raw:
        raw = raw.split("<|im_start|>", 1)[1]
    raw = raw.split("<|im_end|>", 1)[0]
    return raw.strip()


class BagelZebraCoTBackend:
    def __init__(
        self,
        checkpoint_dir: Union[str, Path],
        bagel_repo: Union[str, Path],
        max_mem_gib: int = 90,
        num_timesteps: int = 50,
        text_temperature: float = 0.3,
        do_sample: bool = True,
        seed: int = 42,
        system_prompt: Optional[str] = INTERLEAVED_SYSTEM_PROMPT,
    ):
        self.checkpoint_dir = str(Path(checkpoint_dir).resolve())
        bagel_repo = str(Path(bagel_repo).resolve())
        if bagel_repo not in sys.path:
            sys.path.insert(0, bagel_repo)

        # Imports resolved from the Bagel-Zebra-CoT repo.
        from accelerate import (
            infer_auto_device_map,
            init_empty_weights,
            load_checkpoint_and_dispatch,
        )
        from data.data_utils import add_special_tokens, pil_img2rgb
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            Bagel,
            BagelConfig,
            Qwen2Config,
            Qwen2ForCausalLM,
            SiglipVisionConfig,
            SiglipVisionModel,
        )
        from modeling.qwen2 import Qwen2Tokenizer

        self._pil_img2rgb = pil_img2rgb

        ckpt_dir = self.checkpoint_dir
        checkpoint_path = None
        for cand in ("model_bf16.safetensors", "ema.safetensors", "model.safetensors"):
            p = os.path.join(ckpt_dir, cand)
            if os.path.exists(p):
                checkpoint_path = p
                break
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"No model weights (model_bf16.safetensors / ema.safetensors) found in {ckpt_dir}"
            )

        n_gpu = torch.cuda.device_count()
        print(f"[backend] GPUs available: {n_gpu}")
        for i in range(n_gpu):
            props = torch.cuda.get_device_properties(i)
            print(f"[backend]   GPU {i}: {props.name}, {props.total_memory / 1e9:.1f} GB")

        llm_config = Qwen2Config.from_json_file(os.path.join(ckpt_dir, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(ckpt_dir, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

        vae_model, vae_config = load_ae(local_path=os.path.join(ckpt_dir, "ae.safetensors"))

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act="gelu_pytorch_tanh",
            latent_patch_size=2,
            max_latent_size=64,
        )

        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        tokenizer = Qwen2Tokenizer.from_pretrained(ckpt_dir)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 512, 14)

        print("[backend] Building device map ...")
        device_map = infer_auto_device_map(
            model,
            max_memory={i: f"{max_mem_gib}GiB" for i in range(n_gpu)},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
            dtype=torch.bfloat16,
        )

        same_device_modules = [
            "language_model.model.embed_tokens",
            "time_embedder",
            "latent_pos_embed",
            "vae2llm",
            "llm2vae",
            "connector",
            "vit_pos_embed",
        ]
        if n_gpu == 1:
            first_device = device_map.get(same_device_modules[0], "cuda:0")
            for k in same_device_modules:
                device_map[k] = first_device if k in device_map else "cuda:0"
        else:
            first_device = device_map.get(same_device_modules[0])
            if first_device is not None:
                for k in same_device_modules:
                    if k in device_map:
                        device_map[k] = first_device

        print(f"[backend] Loading checkpoint in bf16: {checkpoint_path}")
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=checkpoint_path,
            device_map=device_map,
            offload_buffers=False,
            dtype=torch.bfloat16,
            force_hooks=True,
        )
        model = model.eval()
        print("[backend] Model loaded.")
        for i in range(n_gpu):
            if torch.cuda.memory_allocated(i) > 0:
                print(
                    f"[backend]   GPU {i}: {torch.cuda.memory_allocated(i) / 1e9:.1f} GB allocated"
                )

        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.system_prompt = system_prompt
        # Generation hyperparameters recommended by the Bagel-Zebra-CoT authors (infz_bf16.py).
        self.gen_hyper = dict(
            do_sample=do_sample,
            text_temperature=text_temperature,
            cfg_text_scale=4.0,
            cfg_img_scale=2.0,
            cfg_interval=[0.0, 1.0],
            timestep_shift=3.0,
            num_timesteps=num_timesteps,
            cfg_renorm_min=0.0,
            cfg_renorm_type="text_channel",
        )

    # ------------------------------------------------------------------
    def _ctx_to_inputs(self, ctx: List[ContextItem], prompt_suffix: str = "") -> List:
        inputs: List[Union[str, Image.Image]] = []
        for kind, payload in ctx:
            if kind == "text":
                inputs.append(str(payload))
            elif kind == "image":
                img = payload if isinstance(payload, Image.Image) else Image.open(str(payload))
                inputs.append(self._pil_img2rgb(img))
            else:
                raise ValueError(f"Unknown context item kind: {kind}")
        if prompt_suffix:
            inputs.append(prompt_suffix)
        return inputs

    @torch.no_grad()
    def generate_text(
        self,
        ctx: List[ContextItem],
        prompt_suffix: str = "",
        max_tokens: int = 800,
    ) -> str:
        inputs = self._ctx_to_inputs(ctx, prompt_suffix)
        out = self.inferencer.interleave_inference(
            inputs,
            understanding_output=True,
            system_prompt=self.system_prompt,
            max_think_token_n=max_tokens,
            **self.gen_hyper,
        )
        for o in out:
            if isinstance(o, str):
                return _clean_text(o)
        return ""

    @torch.no_grad()
    def generate_image(
        self,
        ctx: List[ContextItem],
        out_path: Union[str, Path],
        prompt_suffix: str = "",
    ) -> str:
        inputs = self._ctx_to_inputs(ctx, prompt_suffix)
        out = self.inferencer.interleave_inference(
            inputs,
            understanding_output=False,
            system_prompt=self.system_prompt,
            **self.gen_hyper,
        )
        img = None
        for o in out:
            if isinstance(o, Image.Image):
                img = o
                break
        if img is None:
            raise RuntimeError("Model did not return an image.")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return str(out_path)
