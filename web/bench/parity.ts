// Verify transformers.js (browser/Node) agrees with the Python ONNX builder.
import { pipeline } from "@huggingface/transformers";
import { readFileSync, writeFileSync } from "node:fs";

const texts = JSON.parse(readFileSync("/tmp/parity_texts.json", "utf8")) as string[];
const ex = await pipeline("feature-extraction", "Xenova/multilingual-e5-small", { dtype: "q8" });
const out: number[][] = [];
for (const t of texts) {
  const r = await ex(t, { pooling: "mean", normalize: true });
  out.push(Array.from(r.data as Float32Array));
}
writeFileSync("/tmp/parity_js.json", JSON.stringify(out));
console.log("encoded", out.length, "dim", out[0].length);
