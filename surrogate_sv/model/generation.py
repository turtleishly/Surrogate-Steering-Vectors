import torch as t


def generate_sequence(model, prompt, max_tokens: int = 10, insert_bos: bool = False, stream: bool = True):
    """Greedy generation helper preserved from notebook behavior."""
    current_tokens = model.to_tokens(prompt, prepend_bos=insert_bos)

    eot_token_id = model.tokenizer.eos_token_id
    if eot_token_id is None:
        try:
            eot_token_id = model.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        except Exception:
            eot_token_id = None

    for _ in range(max_tokens):
        with t.inference_mode():
            logits = model(current_tokens)

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        current_tokens = t.cat([current_tokens, next_token], dim=-1)

        if stream:
            token_text = model.to_string(next_token[0])
            print(token_text, end="", flush=True)

        if eot_token_id is not None and (next_token == eot_token_id).all():
            break

    if stream:
        print()

    return model.to_string(current_tokens[0])
