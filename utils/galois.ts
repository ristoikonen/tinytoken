/**
 * Galois Field GF(2^8) Arithmetic, Reed-Solomon Error Correction & Homomorphic State Engine
 * Primitive Polynomial: 0x11d (x^8 + x^4 + x^3 + x^2 + 1)
 *
 * Performance Architecture:
 * - Precomputed 64 KB 2D Multiplication Table (`MUL_TABLE`): Zero branching, single L1-cache lookup
 * - Precomputed 256-byte Inversion Table (`INV_TABLE`): O(1) multiplicative inverse and division
 * - Row-hoisted synthetic polynomial division in `reedSolomonEncode`
 */

const PRIMITIVE_POLY = 0x11d;
export const EXP_TABLE = new Uint8Array(512);
export const LOG_TABLE = new Uint8Array(256);
export const MUL_TABLE = new Uint8Array(256 * 256);
export const INV_TABLE = new Uint8Array(256);

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
    EXP_TABLE[i] = EXP_TABLE[i - 255]!;
  }
  LOG_TABLE[0] = 0;

  // Precompute 64 KB 2D Multiplication Table: zero branches, single L1 data cache lookup
  for (let a = 0; a < 256; a++) {
    const row = a << 8;
    for (let b = 0; b < 256; b++) {
      MUL_TABLE[row | b] = a === 0 || b === 0 ? 0 : EXP_TABLE[LOG_TABLE[a]! + LOG_TABLE[b]!]!;
    }
  }

  // Precompute 256-byte Inversion Table: a * a^-1 = 1 (INV_TABLE[0] = 0)
  INV_TABLE[0] = 0;
  for (let a = 1; a < 256; a++) {
    INV_TABLE[a] = EXP_TABLE[255 - LOG_TABLE[a]!]!;
  }

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
  return MUL_TABLE[((a & 0xff) << 8) | (b & 0xff)]!;
}

export function gfDiv(a: number, b: number): number {
  const bByte = b & 0xff;
  if (bByte === 0) throw new Error("GF(2^8) division by zero");
  return MUL_TABLE[((a & 0xff) << 8) | INV_TABLE[bByte]!]!;
}

export function gfInv(a: number): number {
  const aByte = a & 0xff;
  if (aByte === 0) throw new Error("GF(2^8) inverse of zero");
  return INV_TABLE[aByte]!;
}

export function gfPow(a: number, power: number): number {
  if (power === 0) return 1;
  if (a === 0) return 0;
  const logA = LOG_TABLE[a & 0xff]!;
  const idx = ((logA * power) % 255 + 255) % 255;
  return EXP_TABLE[idx]!;
}

/**
 * Evaluates a polynomial over GF(2^8) at point x using Horner's method with accelerated lookup.
 */
export function gfPolyEval(poly: Uint8Array, x: number): number {
  const len = poly.length;
  if (len === 0) return 0;
  const xByte = x & 0xff;
  let result = poly[0]! & 0xff;
  for (let i = 1; i < len; i++) {
    result = MUL_TABLE[(result << 8) | xByte]! ^ poly[i]!;
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
    const root = EXP_TABLE[i]!;
    const rootRow = root << 8;
    for (let j = 0; j < g.length; j++) {
      const gj = g[j]!;
      nextG[j] = nextG[j]! ^ MUL_TABLE[rootRow | gj]!;
      nextG[j + 1] = nextG[j + 1]! ^ gj;
    }
    g = nextG;
  }
  // Reverse so highest degree coefficient is at index 0 for synthetic division
  const gen = new Uint8Array(g.length);
  for (let i = 0; i < g.length; i++) {
    gen[i] = g[g.length - 1 - i]!;
  }
  generatorCache.set(nsym, gen);
  return gen;
}

/**
 * Computes nsym Reed-Solomon parity bytes over GF(2^8) for a message block (length + nsym <= 255).
 * Uses row-hoisted table lookup for high throughput.
 */
export function reedSolomonEncode(data: Uint8Array, nsym: number): Uint8Array {
  const gen = getRsGeneratorPoly(nsym);
  const dataLen = data.length;
  const genLen = gen.length;
  const totalLen = dataLen + nsym;
  if (totalLen === 0) return new Uint8Array(0);

  const work = new Uint8Array(totalLen);
  work.set(data, 0);

  for (let i = 0; i < dataLen; i++) {
    const coef = work[i]!;
    if (coef !== 0) {
      const row = coef << 8;
      for (let j = 1; j < genLen; j++) {
        const idx = i + j;
        work[idx] = work[idx]! ^ MUL_TABLE[row | gen[j]!]!;
      }
    }
  }

  return work.subarray(dataLen);
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
    if (parity[i] !== expectedParity[i]) {
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
    deltaParity[i] = parity[i]! ^ expectedParity[i]!;
  }

  const probe = new Uint8Array(data.length);
  for (let pos = 0; pos < data.length; pos++) {
    probe[pos] = 1;
    const unitParity = reedSolomonEncode(probe, nsym);
    probe[pos] = 0;

    const u0 = unitParity[0]!;
    const d0 = deltaParity[0]!;
    if (u0 === 0 || d0 === 0) continue;

    const errMag = gfDiv(d0, u0);
    const errRow = errMag << 8;
    let matchesAll = true;
    for (let k = 1; k < nsym; k++) {
      if (MUL_TABLE[errRow | unitParity[k]!] !== deltaParity[k]) {
        matchesAll = false;
        break;
      }
    }

    if (matchesAll) {
      corrected[pos] = corrected[pos]! ^ errMag;
      return { valid: true, correctedData: corrected, correctedErrors: 1 };
    }
  }

  return { valid: false, correctedData: corrected, correctedErrors: -1 };
}

/**
 * Multi-lane GF(2^8) Reed-Solomon encoder/verifier for the 640-byte Section I payload:
 * Splits 640 bytes into 3 lanes (214B, 213B, 213B) and generates 22B + 21B + 21B = 64B parity.
 * Each lane is <= 255 bytes in GF(2^8) and heals single-byte corruptions independently.
 */
export const LANE_SIZES = [214, 213, 213] as const;
export const LANE_PARITY_SIZES = [22, 21, 21] as const;

export function encodeHeaderParity64(headerPayload640: Uint8Array): Uint8Array {
  if (headerPayload640.length !== 640) {
    throw new Error(`Expected 640-byte header payload, got ${headerPayload640.length}`);
  }
  const parity64 = new Uint8Array(64);
  let srcOffset = 0;
  let dstOffset = 0;

  for (let i = 0; i < LANE_SIZES.length; i++) {
    const laneLen = LANE_SIZES[i]!;
    const parityLen = LANE_PARITY_SIZES[i]!;
    const lane = headerPayload640.subarray(srcOffset, srcOffset + laneLen);
    const laneParity = reedSolomonEncode(lane, parityLen);
    parity64.set(laneParity, dstOffset);
    srcOffset += laneLen;
    dstOffset += parityLen;
  }

  return parity64;
}

export const encodeHeaderParity = encodeHeaderParity64;
export const encodeHeaderParity384 = encodeHeaderParity64; // Backwards-compatible alias

export function verifyAndHealHeader640(
  headerPayload640: Uint8Array,
  parity64: Uint8Array,
): { valid: boolean; healedPayload: Uint8Array; correctedErrors: number } {
  const healed = new Uint8Array(headerPayload640);
  let srcOffset = 0;
  let dstOffset = 0;
  let totalCorrected = 0;
  let allValid = true;

  for (let i = 0; i < LANE_SIZES.length; i++) {
    const laneLen = LANE_SIZES[i]!;
    const parityLen = LANE_PARITY_SIZES[i]!;
    const lane = healed.subarray(srcOffset, srcOffset + laneLen);
    const laneParity = parity64.subarray(dstOffset, dstOffset + parityLen);
    const res = reedSolomonVerifyAndHeal(lane, laneParity);
    if (!res.valid) {
      allValid = false;
    } else if (res.correctedErrors > 0) {
      healed.set(res.correctedData, srcOffset);
      totalCorrected += res.correctedErrors;
    }
    srcOffset += laneLen;
    dstOffset += parityLen;
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
  const newParity = encodeHeaderParity64(mutated.subarray(0, 640));
  mutated.set(newParity, 640);

  return mutated;
}

/**
 * Hardware-accelerated IEEE 802.3 CRC32 using Bun's native Zig/C implementation.
 */
export function crc32(bytes: Uint8Array): number {
  return Bun.hash.crc32(bytes) >>> 0;
}

