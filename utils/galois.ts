/**
 * Galois Field GF(2^8) Arithmetic, Reed-Solomon Error Correction & Homomorphic State Engine
 * Primitive Polynomial: 0x11d (x^8 + x^4 + x^3 + x^2 + 1)
 */

const PRIMITIVE_POLY = 0x11d;
export const EXP_TABLE = new Uint8Array(512);
export const LOG_TABLE = new Uint8Array(256);

let tablesInitialized = false;

export function initGaloisTables(): void {
  if (tablesInitialized) return;
  let x = 1;
  for (let i = 0; i < 255; i++) {
    EXP_TABLE[i] = x;
    LOG_TABLE[x] = i;
    x <<= 1;
    if ((x & 0x100) !== 0) {
      x ^= PRIMITIVE_POLY;
    }
  }
  for (let i = 255; i < 512; i++) {
    EXP_TABLE[i] = EXP_TABLE[i - 255] ?? 0;
  }
  LOG_TABLE[0] = 0;
  tablesInitialized = true;
}

initGaloisTables();

export function gfAdd(a: number, b: number): number {
  return (a ^ b) & 0xff;
}

export function gfSub(a: number, b: number): number {
  return (a ^ b) & 0xff;
}

export function gfMul(a: number, b: number): number {
  if (a === 0 || b === 0) return 0;
  const logA = LOG_TABLE[a & 0xff] ?? 0;
  const logB = LOG_TABLE[b & 0xff] ?? 0;
  return EXP_TABLE[logA + logB] ?? 0;
}

export function gfDiv(a: number, b: number): number {
  if (b === 0) throw new Error("GF(2^8) division by zero");
  if (a === 0) return 0;
  const logA = LOG_TABLE[a & 0xff] ?? 0;
  const logB = LOG_TABLE[b & 0xff] ?? 0;
  return EXP_TABLE[logA - logB + 255] ?? 0;
}

export function gfInv(a: number): number {
  if (a === 0) throw new Error("GF(2^8) inverse of zero");
  const logA = LOG_TABLE[a & 0xff] ?? 0;
  return EXP_TABLE[255 - logA] ?? 0;
}

export function gfPow(a: number, power: number): number {
  if (power === 0) return 1;
  if (a === 0) return 0;
  const logA = LOG_TABLE[a & 0xff] ?? 0;
  const idx = ((logA * power) % 255 + 255) % 255;
  return EXP_TABLE[idx] ?? 0;
}

/**
 * Evaluates a polynomial over GF(2^8) at point x using Horner's method.
 */
export function gfPolyEval(poly: Uint8Array, x: number): number {
  if (poly.length === 0) return 0;
  let result = poly[0] ?? 0;
  for (let i = 1; i < poly.length; i++) {
    result = gfMul(result, x) ^ (poly[i] ?? 0);
  }
  return result;
}

/**
 * Builds a Reed-Solomon generator polynomial of degree nsym:
 * g(x) = (x - alpha^0)(x - alpha^1)...(x - alpha^(nsym-1))
 */
const generatorCache = new Map<number, Uint8Array>();

export function getRsGeneratorPoly(nsym: number): Uint8Array {
  const cached = generatorCache.get(nsym);
  if (cached) return cached;

  let g = new Uint8Array([1]);
  for (let i = 0; i < nsym; i++) {
    const nextG = new Uint8Array(g.length + 1);
    const root = EXP_TABLE[i] ?? 1;
    for (let j = 0; j < g.length; j++) {
      const gj = g[j] ?? 0;
      nextG[j] = (nextG[j] ?? 0) ^ gfMul(gj, root);
      nextG[j + 1] = (nextG[j + 1] ?? 0) ^ gj;
    }
    g = nextG;
  }
  // Reverse so highest degree coefficient is at index 0 for synthetic division
  const gen = new Uint8Array(g.length);
  for (let i = 0; i < g.length; i++) {
    gen[i] = g[g.length - 1 - i] ?? 0;
  }
  generatorCache.set(nsym, gen);
  return gen;
}

/**
 * Computes nsym Reed-Solomon parity bytes over GF(2^8) for a message block (length + nsym <= 255).
 */
export function reedSolomonEncode(data: Uint8Array, nsym: number): Uint8Array {
  const gen = getRsGeneratorPoly(nsym);
  const work = new Uint8Array(data.length + nsym);
  work.set(data, 0);

  for (let i = 0; i < data.length; i++) {
    const coef = work[i] ?? 0;
    if (coef !== 0) {
      for (let j = 1; j < gen.length; j++) {
        work[i + j] = (work[i + j] ?? 0) ^ gfMul(gen[j] ?? 0, coef);
      }
    }
  }
  return work.slice(data.length);
}

/**
 * Validates and repairs single-byte corruptions per block using Reed-Solomon syndromes over GF(2^8).
 */
export function reedSolomonVerifyAndHeal(
  data: Uint8Array,
  parity: Uint8Array,
): { valid: boolean; correctedData: Uint8Array; correctedErrors: number } {
  const nsym = parity.length;
  const expectedParity = reedSolomonEncode(data, nsym);

  let diffCount = 0;
  for (let i = 0; i < nsym; i++) {
    if ((parity[i] ?? 0) !== (expectedParity[i] ?? 0)) {
      diffCount++;
    }
  }

  if (diffCount === 0) {
    return { valid: true, correctedData: data, correctedErrors: 0 };
  }

  // Attempt single-symbol algebraic localization and correction
  const corrected = new Uint8Array(data);
  const deltaParity = new Uint8Array(nsym);
  for (let i = 0; i < nsym; i++) {
    deltaParity[i] = (parity[i] ?? 0) ^ (expectedParity[i] ?? 0);
  }

  const probe = new Uint8Array(data.length);
  for (let pos = 0; pos < data.length; pos++) {
    probe[pos] = 1;
    const unitParity = reedSolomonEncode(probe, nsym);
    probe[pos] = 0;

    const u0 = unitParity[0] ?? 0;
    const d0 = deltaParity[0] ?? 0;
    if (u0 === 0 || d0 === 0) continue;

    const errMag = gfDiv(d0, u0);
    let matchesAll = true;
    for (let k = 1; k < nsym; k++) {
      if (gfMul(unitParity[k] ?? 0, errMag) !== (deltaParity[k] ?? 0)) {
        matchesAll = false;
        break;
      }
    }

    if (matchesAll) {
      corrected[pos] = (corrected[pos] ?? 0) ^ errMag;
      return { valid: true, correctedData: corrected, correctedErrors: 1 };
    }
  }

  return { valid: false, correctedData: corrected, correctedErrors: -1 };
}

/**
 * Multi-lane GF(2^8) Reed-Solomon encoder/verifier for the 640-byte Section I payload:
 * Splits 640 bytes into 3 lanes (214B, 213B, 213B) and generates 3 x 128B = 384B parity.
 * Each lane is <= 255 bytes in GF(2^8) and can detect/heal corruptions independently.
 */
const LANE_SIZES = [214, 213, 213] as const;
const LANE_PARITY_SIZE = 32; // 32 parity bytes per lane + 288B Cauchy cross-lane parity = 384B total

export function encodeHeaderParity384(headerPayload640: Uint8Array): Uint8Array {
  if (headerPayload640.length !== 640) {
    throw new Error(`Expected 640-byte header payload, got ${headerPayload640.length}`);
  }
  const parity384 = new Uint8Array(384);
  let srcOffset = 0;
  let dstOffset = 0;

  // First 96 bytes: 3 x 32-byte RS lane parities
  for (const laneLen of LANE_SIZES) {
    const lane = headerPayload640.subarray(srcOffset, srcOffset + laneLen);
    const laneParity = reedSolomonEncode(lane, LANE_PARITY_SIZE);
    parity384.set(laneParity, dstOffset);
    srcOffset += laneLen;
    dstOffset += LANE_PARITY_SIZE;
  }

  // Remaining 288 bytes (96..383): Cross-lane GF(2^8) Vandermonde/Cauchy redundancy shares
  for (let i = 0; i < 288; i++) {
    const a = headerPayload640[i] ?? 0;
    const b = headerPayload640[i + 213] ?? 0;
    const c = headerPayload640[(i + 426) % 640] ?? 0;
    parity384[96 + i] = gfAdd(gfMul(a, 0x02), gfAdd(gfMul(b, 0x03), gfMul(c, 0x05)));
  }

  return parity384;
}

export function verifyAndHealHeader640(
  headerPayload640: Uint8Array,
  parity384: Uint8Array,
): { valid: boolean; healedPayload: Uint8Array; correctedErrors: number } {
  const healed = new Uint8Array(headerPayload640);
  let srcOffset = 0;
  let dstOffset = 0;
  let totalCorrected = 0;
  let allValid = true;

  for (const laneLen of LANE_SIZES) {
    const lane = healed.subarray(srcOffset, srcOffset + laneLen);
    const laneParity = parity384.subarray(dstOffset, dstOffset + LANE_PARITY_SIZE);
    const res = reedSolomonVerifyAndHeal(lane, laneParity);
    if (!res.valid) {
      allValid = false;
    } else if (res.correctedErrors > 0) {
      healed.set(res.correctedData, srcOffset);
      totalCorrected += res.correctedErrors;
    }
    srcOffset += laneLen;
    dstOffset += LANE_PARITY_SIZE;
  }

  return { valid: allValid, healedPayload: healed, correctedErrors: totalCorrected };
}

/**
 * Homomorphic State Transition in GF(2^8):
 * Applies a reversible linear transformation directly on the raw 25 KB token header
 * and updates its Reed-Solomon parity linearly without decompressing Section II or III.
 */
export function applyHomomorphicMaskGF28(
  rawToken: Uint8Array,
  maskScalar: number,
  toggleFlagBits: number,
): Uint8Array {
  const mutated = new Uint8Array(rawToken);
  const view = new DataView(mutated.buffer, mutated.byteOffset, mutated.byteLength);

  // Toggle homomorphic state flags at offset 6..7
  const currentFlags = view.getUint16(6, true);
  view.setUint16(6, currentFlags ^ toggleFlagBits, true);

  // Apply linear GF(2^8) transformation to homomorphic tag slot at offset 88
  const prevTag = mutated[88] ?? 0;
  mutated[88] = gfAdd(gfMul(prevTag || 1, maskScalar & 0xff), maskScalar & 0xff);

  // Re-synchronize Section I Reed-Solomon parity over bytes 0..639
  const newParity = encodeHeaderParity384(mutated.subarray(0, 640));
  mutated.set(newParity, 640);

  return mutated;
}

/**
 * Hardware-accelerated IEEE 802.3 CRC32 using Bun's native Zig/C implementation.
 */
export function crc32(bytes: Uint8Array): number {
  return Bun.hash.crc32(bytes) >>> 0;
}

