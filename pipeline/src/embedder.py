"""ONNX embedder shared by the index builder and the drift check.

Uses the exact weights the browser will load (Xenova/multilingual-e5-small), so
index vectors and query vectors come from the same graph. Mismatched weights
between build and query is a silent failure mode: retrieval degrades but nothing
errors.

e5 models REQUIRE the "query: " / "passage: " prefixes. Omitting them costs
several points of recall -- the model was contrastively trained with them.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

MODEL_DIR = "data/model"
MAX_LEN = 192          # p90 chunk is ~110 chars; 192 tokens covers the long tail


class Embedder:
    def __init__(self, quantized: bool = False, threads: int = 0):
        name = "model_quantized.onnx" if quantized else "model.onnx"
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(f"{MODEL_DIR}/onnx/{name}",
                                         so, providers=["CPUExecutionProvider"])
        self.tok = Tokenizer.from_file(f"{MODEL_DIR}/tokenizer.json")
        self.tok.enable_truncation(MAX_LEN)
        self.tok.enable_padding(length=None)
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], prefix: str) -> np.ndarray:
        enc = self.tok.encode_batch([f"{prefix}{t}" for t in texts])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        out = self.sess.run(None, feed)[0]            # (B, T, H)

        # e5 uses mean pooling over non-padding tokens, then L2 normalisation.
        m = mask[..., None].astype(np.float32)
        vec = (out * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        vec /= np.clip(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9, None)
        return vec.astype(np.float32)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, "passage: ")

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self.encode(texts, "query: ")
