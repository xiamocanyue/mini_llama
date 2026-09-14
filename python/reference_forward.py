#!/usr/bin/env python3
"""Independent, standard-library reference forward pass for the tiny model."""

import argparse
import json
import math
import struct
from pathlib import Path


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def add(a, b):
    return f32(a + b)


def mul(a, b):
    return f32(a * b)


def dot(a, b):
    total = 0.0
    for left, right in zip(a, b):
        total = add(total, mul(left, right))
    return total


def matvec(matrix, vector):
    return [dot(row, vector) for row in matrix]


def rmsnorm(values, weight, eps):
    sum_squares = 0.0
    for value in values:
        sum_squares = add(sum_squares, mul(value, value))
    mean = f32(sum_squares / len(values))
    scale = f32(1.0 / f32(math.sqrt(add(mean, eps))))
    return [mul(mul(value, scale), w) for value, w in zip(values, weight)]


def silu(value):
    return f32(value / f32(1.0 + f32(math.exp(-value))))


def softmax(values):
    maximum = max(values)
    exponentials = [f32(math.exp(f32(value - maximum))) for value in values]
    total = 0.0
    for value in exponentials:
        total = add(total, value)
    return [f32(value / total) for value in exponentials]


def reshape_heads(values, heads, head_dim):
    return [values[index * head_dim:(index + 1) * head_dim]
            for index in range(heads)]


def apply_rope(heads, pos, theta):
    head_dim = len(heads[0])
    for head in heads:
        for index in range(0, head_dim, 2):
            exponent = f32(index / head_dim)
            frequency = f32(1.0 / f32(math.pow(theta, exponent)))
            angle = f32(pos * frequency)
            cosine = f32(math.cos(angle))
            sine = f32(math.sin(angle))
            left = head[index]
            right = head[index + 1]
            head[index] = add(mul(left, cosine), -mul(right, sine))
            head[index + 1] = add(mul(left, sine), mul(right, cosine))


def add_vectors(left, right):
    return [add(a, b) for a, b in zip(left, right)]


def load_model(config_path, weights_path):
    manifest = json.loads(Path(config_path).read_text(encoding="utf-8"))
    raw = Path(weights_path).read_bytes()
    weights = {}
    for tensor in manifest["tensors"]:
        if tensor["dtype"] != "float32":
            raise ValueError(f"unsupported reference dtype: {tensor['dtype']}")
        shape = tensor["shape"]
        count = math.prod(shape)
        offset = tensor["offset"]
        size = count * 4
        if offset + size > len(raw):
            raise ValueError(f"weights file ended while reading {tensor['name']}")
        flat = list(struct.unpack_from(f"<{count}f", raw, offset))
        if len(shape) == 1:
            data = flat
        elif len(shape) == 2:
            cols = shape[1]
            data = [flat[row * cols:(row + 1) * cols]
                    for row in range(shape[0])]
        else:
            raise ValueError(f"unsupported reference tensor rank: {tensor['name']}")
        weights[tensor["name"]] = data

    config = manifest["config"]
    config["head_dim"] = config.get("head_dim", config["dim"] // config["n_heads"])
    return ReferenceModel(config, weights)


class ReferenceModel:
    def __init__(self, config, weights):
        self.config = config
        self.weights = weights
        layers = config["n_layers"]
        length = config["max_seq_len"]
        kv_heads = config["n_kv_heads"]
        head_dim = config["head_dim"]
        self.keys = [[[[0.0] * head_dim for _ in range(kv_heads)]
                      for _ in range(length)] for _ in range(layers)]
        self.values = [[[[0.0] * head_dim for _ in range(kv_heads)]
                        for _ in range(length)] for _ in range(layers)]

    def forward_token(self, pos, token_id):
        c = self.config
        w = self.weights
        x = list(w["token_embedding"][token_id])
        head_dim = c["head_dim"]
        group_size = c["n_heads"] // c["n_kv_heads"]

        for layer in range(c["n_layers"]):
            prefix = f"layers.{layer}."
            normalized = rmsnorm(x, w[prefix + "attention_norm"],
                                 c["rms_norm_eps"])
            q = reshape_heads(matvec(w[prefix + "wq"], normalized),
                              c["n_heads"], head_dim)
            k = reshape_heads(matvec(w[prefix + "wk"], normalized),
                              c["n_kv_heads"], head_dim)
            v = reshape_heads(matvec(w[prefix + "wv"], normalized),
                              c["n_kv_heads"], head_dim)
            apply_rope(q, pos, c["rope_theta"])
            apply_rope(k, pos, c["rope_theta"])
            self.keys[layer][pos] = [list(head) for head in k]
            self.values[layer][pos] = [list(head) for head in v]

            attention = []
            scale = f32(1.0 / f32(math.sqrt(head_dim)))
            for q_head, query in enumerate(q):
                kv_head = q_head // group_size
                scores = [mul(dot(query, self.keys[layer][position][kv_head]), scale)
                          for position in range(pos + 1)]
                probabilities = softmax(scores)
                output = []
                for dim in range(head_dim):
                    value = 0.0
                    for position, probability in enumerate(probabilities):
                        value = add(value, mul(
                            probability, self.values[layer][position][kv_head][dim]))
                    output.append(value)
                attention.extend(output)

            x = add_vectors(x, matvec(w[prefix + "wo"], attention))
            normalized = rmsnorm(x, w[prefix + "ffn_norm"],
                                 c["rms_norm_eps"])
            gate = matvec(w[prefix + "w_gate"], normalized)
            up = matvec(w[prefix + "w_up"], normalized)
            activated = [mul(silu(gate_value), up_value)
                         for gate_value, up_value in zip(gate, up)]
            x = add_vectors(x, matvec(w[prefix + "w_down"], activated))

        normalized = rmsnorm(x, w["final_norm"], c["rms_norm_eps"])
        return matvec(w["lm_head"], normalized)


def load_tokenizer(vocab_path):
    entries = json.loads(Path(vocab_path).read_text(encoding="utf-8"))
    token_to_id = {entry["content"]: entry["id"] for entry in entries}
    bos = next(entry["id"] for entry in entries if entry["content"] == "<bos>")
    unk = next(entry["id"] for entry in entries if entry["content"] == "<unk>")
    eos = next(entry["id"] for entry in entries if entry["content"] == "<eos>")
    return token_to_id, bos, eos, unk


def encode_prompt(prompt, vocab_path):
    token_to_id, bos, _, unk = load_tokenizer(vocab_path)
    return [bos] + [token_to_id.get(char, unk) for char in prompt]


def generate_reference(model, tokens, n_predict):
    logits_by_step = []
    for pos, token in enumerate(tokens):
        logits_by_step.append(model.forward_token(pos, token))

    generated = []
    for index in range(n_predict):
        next_token = max(range(len(logits_by_step[-1])),
                         key=logits_by_step[-1].__getitem__)
        generated.append(next_token)
        if index + 1 == n_predict or next_token == 2:
            break
        pos = len(tokens) + index
        logits_by_step.append(model.forward_token(pos, next_token))
    return logits_by_step, generated


def write_reference(output_dir, logits_by_step, generated):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for step, logits in enumerate(logits_by_step):
        (output / f"python_logits_step{step}.bin").write_bytes(
            struct.pack(f"<{len(logits)}f", *logits))
    (output / "python_generation_tokens.txt").write_text(
        " ".join(str(token) for token in generated) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run the independent tiny-model reference")
    parser.add_argument("--config", default="models/tiny/model.json")
    parser.add_argument("--weights", default="models/tiny/model.bin")
    parser.add_argument("--vocab", default="models/tiny/vocab.json")
    parser.add_argument("--prompt", default="hello")
    parser.add_argument("-n", "--n-predict", type=int, default=8)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if args.n_predict < 0:
        parser.error("--n-predict must be non-negative")

    model = load_model(args.config, args.weights)
    tokens = encode_prompt(args.prompt, args.vocab)
    if len(tokens) + args.n_predict > model.config["max_seq_len"]:
        parser.error("prompt and generated tokens exceed max_seq_len")
    logits, generated = generate_reference(model, tokens, args.n_predict)
    write_reference(args.output_dir, logits, generated)
    print(f"reference steps: {len(logits)}")
    print(f"generated tokens: {generated}")


if __name__ == "__main__":
    main()
