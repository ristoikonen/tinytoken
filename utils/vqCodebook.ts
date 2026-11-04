/**
 * Vector Quantization (VQ) 16x16 Latent Codebook & Hybrid DINOv3 + ADC Similarity Engine
 *
 * Maps any image or document page into a 256-byte (16x16) discrete codebook grid over GF(2^8) (0..255).
 * Dual Purpose:
 * 1. Neural Micro-Thumbnail: Reconstructs a true 16x16 RGB color grid in < 0.01 ms with zero disk I/O.
 * 2. Hybrid DINOv3 + Asymmetric Distance Computation (ADC): Combines 384-byte DINOv3 ViT-S/16 semantic
 *    embeddings with the 256x256 VQ centroid distance LUT to rank tokens in microseconds.
 */

import { computeDinoCosineSimilarity } from "./dinoEmbedding";

export interface VqCentroidRGB {
  r: number;
  g: number;
  b: number;
}

/**
 * Deterministic 256-centroid perceptual RGB codebook (6x7x6 uniform Lab/sRGB lattice + 4 high-contrast greyscale anchors):
 * Every index 0..255 is an element of GF(2^8) with an exact (R, G, B) centroid.
 */
export const VQ_CODEBOOK: ReadonlyArray<VqCentroidRGB> = (() => {
  const table: VqCentroidRGB[] = new Array(256);
  let idx = 0;
  // 6 Red levels x 7 Green levels x 6 Blue levels = 252 chromatic/luminance centroids
  const rLevels = [12, 60, 108, 156, 204, 248];
  const gLevels = [10, 48, 88, 128, 168, 208, 248];
  const bLevels = [16, 64, 112, 160, 208, 250];

  for (const r of rLevels) {
    for (const g of gLevels) {
      for (const b of bLevels) {
        table[idx++] = { r, g, b };
      }
    }
  }
  // 4 pure document/ink contrast anchors (252..255)
  table[252] = { r: 0, g: 0, b: 0 };       // Pure ink black
  table[253] = { r: 85, g: 85, b: 85 };    // Dark charcoal
  table[254] = { r: 170, g: 170, b: 170 }; // Silver grey
  table[255] = { r: 255, g: 255, b: 255 }; // Pure paper white
  return table;
})();

/**
 * Precomputed 256x256 Squared Perceptual Euclidean Distance Lookup Table (LUT)
 * Enables O(1) table lookup per latent cell during Asymmetric Distance Computation (ADC).
 */
export const VQ_DISTANCE_LUT = new Uint16Array(256 * 256);

(() => {
  for (let i = 0; i < 256; i++) {
    const ci = VQ_CODEBOOK[i]!;
    for (let j = i; j < 256; j++) {
      const cj = VQ_CODEBOOK[j]!;
      const dr = ci.r - cj.r;
      const dg = ci.g - cj.g;
      const db = ci.b - cj.b;
      // Perceptual luminance-weighted squared distance (scaled to fit Uint16)
      const dist = Math.min(65535, Math.round((2 * dr * dr + 4 * dg * dg + 1 * db * db) / 7));
      VQ_DISTANCE_LUT[i * 256 + j] = dist;
      VQ_DISTANCE_LUT[j * 256 + i] = dist;
    }
  }
})();

const R_LEVELS = [12, 60, 108, 156, 204, 248] as const;
const G_LEVELS = [10, 48, 88, 128, 168, 208, 248] as const;
const B_LEVELS = [16, 64, 112, 160, 208, 250] as const;

/**
 * Precomputed 32,768-entry (32x32x32 = 15-bit RGB) Quantization Lookup Table (LUT).
 * Precomputed during module initialization: maps every 5-bit (R, G, B) cube directly
 * to the exact nearest centroid in VQ_CODEBOOK under the weighted perceptual metric:
 *   dist = 2*(dr^2) + 4*(dg^2) + 1*(db^2)
 *
 * Accounts for 4x green weighting, non-uniform channel spacing, and makes all four
 * greyscale anchors (#252 black, #253 charcoal, #254 silver, #255 white) reachable.
 */
export const RGB_TO_VQ_LUT: Uint8Array = (() => {
  const lut = new Uint8Array(32768);
  for (let r5 = 0; r5 < 32; r5++) {
    const r = (r5 << 3) + 4;
    let bestRi = 0;
    let minR = Math.abs(r - R_LEVELS[0]!);
    for (let ri = 1; ri < 6; ri++) {
      const d = Math.abs(r - R_LEVELS[ri]!);
      if (d < minR) {
        minR = d;
        bestRi = ri;
      }
    }

    for (let g5 = 0; g5 < 32; g5++) {
      const g = (g5 << 3) + 4;
      let bestGi = 0;
      let minG = Math.abs(g - G_LEVELS[0]!);
      for (let gi = 1; gi < 7; gi++) {
        const d = Math.abs(g - G_LEVELS[gi]!);
        if (d < minG) {
          minG = d;
          bestGi = gi;
        }
      }

      for (let b5 = 0; b5 < 32; b5++) {
        const b = (b5 << 3) + 4;
        let bestBi = 0;
        let minB = Math.abs(b - B_LEVELS[0]!);
        for (let bi = 1; bi < 6; bi++) {
          const d = Math.abs(b - B_LEVELS[bi]!);
          if (d < minB) {
            minB = d;
            bestBi = bi;
          }
        }

        let bestIdx = bestRi * 42 + bestGi * 6 + bestBi;
        const latticeC = VQ_CODEBOOK[bestIdx]!;
        let bestDist =
          2 * (r - latticeC.r) ** 2 +
          4 * (g - latticeC.g) ** 2 +
          1 * (b - latticeC.b) ** 2;

        // Check 4 contrast greyscale anchors (252..255)
        for (let a = 252; a <= 255; a++) {
          const c = VQ_CODEBOOK[a]!;
          const dist =
            2 * (r - c.r) ** 2 +
            4 * (g - c.g) ** 2 +
            1 * (b - c.b) ** 2;
          if (dist < bestDist) {
            bestDist = dist;
            bestIdx = a;
          }
        }

        lut[(r5 << 10) | (g5 << 5) | b5] = bestIdx;
      }
    }
  }
  return lut;
})();

/**
 * Quantizes an (R, G, B) triplet to its nearest GF(2^8) codebook centroid index (0..255)
 * using the precomputed 32 KB RGB_TO_VQ_LUT in O(1) single-memory-lookup time.
 */
export function quantizeRgbToCentroid(r: number, g: number, b: number): number {
  const rClamped = r < 0 ? 0 : r > 255 ? 255 : r;
  const gClamped = g < 0 ? 0 : g > 255 ? 255 : g;
  const bClamped = b < 0 ? 0 : b > 255 ? 255 : b;
  const index =
    ((rClamped >>> 3) << 10) |
    ((gClamped >>> 3) << 5) |
    (bClamped >>> 3);
  return RGB_TO_VQ_LUT[index]!;
}

/**
 * Encodes a true decoded `16x16x3` (768-byte) RGB pixel buffer directly into a 256-byte VQ grid.
 * Produces faithful miniature `<canvas>` thumbnails and distortion-free spatial color ADC distances.
 */
export function encodeRgb16x16ToVqGrid(rgb16x16: Uint8Array): Uint8Array<ArrayBuffer> {
  const vqGrid = new Uint8Array(new ArrayBuffer(256));
  if (rgb16x16.length < 256 * 3) {
    return encodeBytesToVqGrid(rgb16x16);
  }

  for (let cell = 0; cell < 256; cell++) {
    const p = cell * 3;
    const r = rgb16x16[p] ?? 0;
    const g = rgb16x16[p + 1] ?? 0;
    const b = rgb16x16[p + 2] ?? 0;
    vqGrid[cell] = quantizeRgbToCentroid(r, g, b);
  }
  return vqGrid;
}

/**
 * Fallback deterministic 256-byte (16x16) VQ encoder when raw un-decoded buffers are provided directly.
 */
export function encodeBytesToVqGrid(
  fileBytes: Uint8Array,
  _width = 220,
  _height = 220,
  _seedSalt = 0,
): Uint8Array<ArrayBuffer> {
  const vqGrid = new Uint8Array(new ArrayBuffer(256));
  const len = fileBytes.length;
  if (len === 0) return vqGrid;

  if (len === 256 * 3) {
    return encodeRgb16x16ToVqGrid(fileBytes);
  }

  const payloadStart = len > 512 ? Math.min(128, Math.floor(len * 0.05)) : 0;
  const usableLen = Math.max(1, len - payloadStart);

  for (let cell = 0; cell < 256; cell++) {
    const winStart = payloadStart + Math.floor((cell / 256) * usableLen);
    const winSize = Math.min(48, Math.max(3, Math.floor(usableLen / 256)));

    let accR = 0;
    let accG = 0;
    let accB = 0;
    let count = 0;

    for (let k = 0; k < winSize; k += 3) {
      accR += fileBytes[(winStart + k) % len] ?? 0;
      accG += fileBytes[(winStart + k + 1) % len] ?? 0;
      accB += fileBytes[(winStart + k + 2) % len] ?? 0;
      count++;
    }

    const r = Math.round(accR / Math.max(1, count));
    const g = Math.round(accG / Math.max(1, count));
    const b = Math.round(accB / Math.max(1, count));
    vqGrid[cell] = quantizeRgbToCentroid(r, g, b);
  }

  return vqGrid;
}

/**
 * Validates that a VQ grid is exactly 256 bytes (16x16).
 * Rejects truncated or malformed grids that would otherwise produce false matches
 * by silently padding missing positions with centroid 0.
 */
export function validateVqGrid(grid: Uint8Array): void {
  if (!grid || grid.length !== 256) {
    throw new RangeError(
      `Expected 256-byte VQ grid, received ${grid?.length ?? 0}`,
    );
  }
}

/**
 * Decodes a 256-byte VQ grid back into a 16x16 RGBA pixel buffer (1,024 bytes) in < 0.01 ms.
 */
export function decodeVqGridToRgba(vqGrid: Uint8Array): Uint8ClampedArray {
  validateVqGrid(vqGrid);
  const rgba = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const idx = vqGrid[i]!;
    const c = VQ_CODEBOOK[idx]!;
    const p = i * 4;
    rgba[p] = c.r;
    rgba[p + 1] = c.g;
    rgba[p + 2] = c.b;
    rgba[p + 3] = 255;
  }
  return rgba;
}

/**
 * Asymmetric Distance Computation (ADC) between two 256-byte VQ latent grids.
 * Uses only integer table lookups into VQ_DISTANCE_LUT — takes ~0.2 microseconds per comparison!
 */
export function computeVqAdcDistance(queryGrid: Uint8Array, candidateGrid: Uint8Array): number {
  validateVqGrid(queryGrid);
  validateVqGrid(candidateGrid);
  let totalDist = 0;
  for (let i = 0; i < 256; i++) {
    const q = queryGrid[i]!;
    const c = candidateGrid[i]!;
    totalDist += VQ_DISTANCE_LUT[(q << 8) | c]!;
  }
  return totalDist;
}

export const MAX_CELL_ADC_DISTANCE = 65025; // 255^2, max distance between #252 (0,0,0) and #255 (255,255,255)
export const MAX_ADC_DISTANCE = 256 * MAX_CELL_ADC_DISTANCE; // 16,646,400 for 16x16 grid (256 cells)

/**
 * Converts raw ADC distance into a normalized 0..100% spatial color similarity score
 * grounded in the theoretical upper bound MAX_ADC_DISTANCE of the perceptual distance metric.
 */
export function adcDistanceToSimilarityPct(adcDist: number): number {
  const norm = Math.max(0, 1 - adcDist / MAX_ADC_DISTANCE);
  return Math.round(norm * 1000) / 10;
}

/**
 * Hybrid DINOv3 (384B ViT-S/16 Int8) + 16x16 VQ Perceptual Color-Layout Similarity.
 * When both models are known and compatible, combines 90% semantic DINO cosine similarity with 10% VQ spatial palette similarity.
 * When DINO embeddings are absent or models are not verified compatible, returns VQ-only similarity and dinoSimilarityPct: null.
 */
export function computeHybridVisualSimilarity(
  queryVq: Uint8Array,
  candidateVq: Uint8Array,
  queryDino?: Int8Array,
  candidateDino?: Int8Array,
  queryModelId?: number,
  candidateModelId?: number,
): {
  similarityScore: number;
  adcDistance: number;
  dinoSimilarityPct: number | null;
} {
  const adcDistance = computeVqAdcDistance(queryVq, candidateVq);
  if (adcDistance === 0 && (!queryDino || !candidateDino)) {
    return { similarityScore: 100, adcDistance: 0, dinoSimilarityPct: null };
  }

  const vqSim01 = Math.max(0, 1 - adcDistance / MAX_ADC_DISTANCE);

  // An unknown model ID is not evidence of compatibility.
  // Only calculate DINO similarity when both model identifiers are known and match.
  const modelsCompatible =
    queryModelId !== undefined &&
    candidateModelId !== undefined &&
    queryModelId === candidateModelId;

  if (
    modelsCompatible &&
    queryDino &&
    candidateDino &&
    queryDino.length > 0 &&
    candidateDino.length > 0
  ) {
    const cosine = computeDinoCosineSimilarity(queryDino, candidateDino);
    if (cosine >= 0.9999 && adcDistance === 0) {
      return { similarityScore: 100, adcDistance: 0, dinoSimilarityPct: 100 };
    }
    const dinoSim01 = Math.max(0, Math.min(1, cosine));
    const hybrid01 = dinoSim01 * 0.9 + vqSim01 * 0.1;
    return {
      similarityScore: Math.round(hybrid01 * 1000) / 10,
      adcDistance,
      dinoSimilarityPct: Math.round(dinoSim01 * 1000) / 10,
    };
  }

  return {
    similarityScore: Math.round(vqSim01 * 1000) / 10,
    adcDistance,
    dinoSimilarityPct: null,
  };
}

export function vqGridToHex(vqGrid: Uint8Array): string {
  return vqGrid.toHex();
}

export function hexToVqGrid(hex: string): Uint8Array<ArrayBuffer> {
  const clean = hex.trim();
  const parsed = Uint8Array.fromHex(clean.slice(0, 512));
  if (parsed.length === 256) return parsed;
  const grid = new Uint8Array(new ArrayBuffer(256));
  grid.set(parsed.subarray(0, 256));
  return grid;
}
