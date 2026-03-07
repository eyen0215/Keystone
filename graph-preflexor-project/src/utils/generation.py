"""
Model loading and inference utilities for Graph-PReFLexOR.
Handles both PRefLexOR library and manual fallback generation.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Try PRefLexOR library (has nice wrappers), fall back to manual generation
try:
    from PRefLexOR import generate_local_model, extract_text
    PREFLEXOR_AVAILABLE = True
except ImportError:
    PREFLEXOR_AVAILABLE = False

THINK_START = "<|thinking|>"
THINK_END = "<|/thinking|>"


def load_model(model_name="lamm-mit/Graph-Preflexor_01062025", dtype="bfloat16"):
    """
    Load the Graph-PReFLexOR model and tokenizer.

    Returns:
        model, tokenizer, device_info (dict)
    """
    print(f"Loading model: {model_name}")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, use_fast=False,
    )

    torch_dtype = getattr(torch, dtype, torch.bfloat16)

    # Try flash attention first, fall back gracefully
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2",
            device_map="auto",
            trust_remote_code=True,
        )
        attn_type = "flash_attention_2"
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        attn_type = "default"

    load_time = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e9

    info = {
        "model_name": model_name,
        "params_B": round(param_count, 1),
        "dtype": dtype,
        "attention": attn_type,
        "load_time_s": round(load_time, 1),
        "preflexor_available": PREFLEXOR_AVAILABLE,
    }

    print(f"  Loaded in {load_time:.1f}s | {param_count:.1f}B params | {attn_type} attention")
    if not PREFLEXOR_AVAILABLE:
        print("  (PRefLexOR library not found — using manual generation fallback)")

    return model, tokenizer, info


def generate_response(model, tokenizer, prompt,
                      system_prompt="You are a materials scientist.",
                      max_new_tokens=2048, temperature=0.3,
                      prepend_thinking=True):
    """
    Generate a response from Graph-PReFLexOR.

    Args:
        model: The loaded model
        tokenizer: The loaded tokenizer
        prompt: User question/prompt
        system_prompt: System context
        max_new_tokens: Max tokens to generate
        temperature: Sampling temperature
        prepend_thinking: Whether to prepend <|thinking|> token to guide
                          the model into graph-reasoning mode

    Returns:
        raw_output (str): Full model output including thinking tokens
    """
    if PREFLEXOR_AVAILABLE:
        output_text, _ = generate_local_model(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            system_prompt=system_prompt,
            prepend_response=THINK_START if prepend_thinking else "",
            num_return_sequences=1,
            repetition_penalty=1.1,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            messages=[],
            do_sample=True,
        )
        return output_text
    else:
        # Manual generation via chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        encoded = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True,
        )

        # apply_chat_template may return a BatchEncoding or a plain tensor
        if hasattr(encoded, "input_ids"):
            input_ids = encoded.input_ids.to(model.device)
        elif isinstance(encoded, torch.Tensor):
            input_ids = encoded.to(model.device)
        else:
            input_ids = torch.tensor([encoded]).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_ids.shape[1]:]
        return tokenizer.decode(new_tokens, skip_special_tokens=False)


def split_thinking_answer(raw_output):
    """
    Split model output into thinking section and answer section.

    Returns:
        thinking (str), answer (str)
    """
    if THINK_START in raw_output and THINK_END in raw_output:
        idx_s = raw_output.index(THINK_START) + len(THINK_START)
        idx_e = raw_output.index(THINK_END)
        thinking = raw_output[idx_s:idx_e].strip()
        answer = raw_output[idx_e + len(THINK_END):].strip()
        return thinking, answer
    elif PREFLEXOR_AVAILABLE:
        thinking = extract_text(raw_output, thinking_start=THINK_START, thinking_end=THINK_END)
        answer = extract_text(raw_output, thinking_start=THINK_END, thinking_end="NONE").strip()
        return thinking, answer
    else:
        return raw_output, ""