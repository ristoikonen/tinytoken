/**
 * Bun + ONNX spike: real DINOv2 ViT-S/14 (Xenova/dinov2-small) CLS embeddings for ./images.
 * Run: bun run scripts/spike_dinov2.ts [q8|fp32|fp16]
 * Prints load time, per-image latency, RAM, int8 quantisation fidelity and top-3 neighbours per image.
 */
import { pipeline, RawImage, env } from "@huggingface/transformers";
import sharp from "sharp";

const dtype = (Bun.argv[2] ?? "q8") as "q8" | "fp32" | "fp16";
env.cacheDir = "./.cache/transformers";

const mb = () => Math.round(process.memoryUsage.rss() / 1048576);
console.log(`Bun ${Bun.version} | dtype=${dtype} | RSS at start: ${mb()} MB`);

const tLoad = performance.now();
const extractor = await pipeline("image-feature-extraction", "Xenova/dinov2-small", { dtype });
console.log(`Model loaded in ${Math.round(performance.now() - tLoad)} ms | RSS: ${mb()} MB`);

const files: string[] = [];
for await (const f of new Bun.Glob("*.{jpg,jpeg,png,webp,svg}").scan({ cwd: "./images" })) files.push(f);
files.sort();

function l2(v: Float32Array): Float32Array {
  let n = 0;
  for (const x of v) n += x * x;
  n = Math.sqrt(n) || 1;
  return v.map((x) => x / n) as Float32Array;
}
function cosine(a: Float32Array, b: Float32Array): number {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i]! * b[i]!;
  return d;
}
/** int8 quantise (L2-normalised vector, fixed scale from its own max) and return dequantised copy + bytes */
function quantiseInt8(v: Float32Array): { q: Int8Array; scale: number } {
  let max = 0;
  for (const x of v) max = Math.max(max, Math.abs(x));
  const scale = max / 127 || 1;
  const q = new Int8Array(v.length);
  for (let i = 0; i < v.length; i++) q[i] = Math.max(-127, Math.min(127, Math.round(v[i]! / scale)));
  return { q, scale };
}
function cosineInt8(a: Int8Array, b: Int8Array): number {
  let d = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    d += a[i]! * b[i]!;
    na += a[i]! * a[i]!;
    nb += b[i]! * b[i]!;
  }
  return d / (Math.sqrt(na * nb) || 1);
}

const items: { name: string; f32: Float32Array; i8: Int8Array; ms: number }[] = [];
for (const name of files) {
  const buf = await Bun.file(`./images/${name}`).arrayBuffer();
  const { data, info } = await sharp(buf, { failOn: "none" })
    .flatten({ background: { r: 255, g: 255, b: 255 } })
    .resize(512, 512, { fit: "inside", withoutEnlargement: true })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  const img = new RawImage(new Uint8ClampedArray(data), info.width, info.height, 3);

  const t0 = performance.now();
  const out = await extractor(img);
  const ms = performance.now() - t0;

  // last_hidden_state [1, 257, 384] -> CLS = first 384 values
  const raw = out.data as Float32Array;
  const f32 = l2(Float32Array.from(raw.subarray(0, 384)));
  items.push({ name, f32, i8: quantiseInt8(f32).q, ms });
  console.log(`  ${name.padEnd(30)} dims=[${out.dims.join(",")}] ${ms.toFixed(0)} ms`);
}

const avg = items.reduce((s, i) => s + i.ms, 0) / items.length;
console.log(`\nAvg inference: ${avg.toFixed(1)} ms/image (first call includes warm-up) | RSS: ${mb()} MB`);

let maxErr = 0;
for (let i = 0; i < items.length; i++) {
  for (let j = i + 1; j < items.length; j++) {
    maxErr = Math.max(maxErr, Math.abs(cosine(items[i]!.f32, items[j]!.f32) - cosineInt8(items[i]!.i8, items[j]!.i8)));
  }
}
console.log(`int8 quantisation: max |cos_f32 - cos_int8| over all pairs = ${maxErr.toFixed(4)}\n`);

for (const a of items) {
  const nn = items
    .filter((b) => b !== a)
    .map((b) => ({ n: b.name, s: cosine(a.f32, b.f32) }))
    .sort((x, y) => y.s - x.s);
  console.log(`${a.name.padEnd(28)} => ${nn.slice(0, 3).map((x) => `${x.n} (${(x.s * 100).toFixed(1)}%)`).join(" | ")}   [lowest: ${(nn.at(-1)!.s * 100).toFixed(1)}%]`);
}

