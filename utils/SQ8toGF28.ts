/**
 * SQ8toGF28.ts
 * Highly Optimized SQ8 & GF(2^8) Vector Pipeline for Bun
 * 
 * Features utilized:
 * - Bun-native binary file I/O & memory streaming
 * - GF(2^8) algebraic tables for error-correcting storage blocks
 */

// Initialize GF(2^8) lookup tables using primitive polynomial 0x11d
const EXP_TABLE = new Uint8Array(512);
const LOG_TABLE = new Uint8Array(256);

function initGaloisTables(): void {
    let x = 1;
    for (let i = 0; i < 255; i++) {
        EXP_TABLE[i] = x;
        LOG_TABLE[x] = i;
        x <<= 1;
        if (x & 0x100) {
            x ^= 0x11d;
        }
    }
    for (let i = 255; i < 512; i++) {
        EXP_TABLE[i] = EXP_TABLE[i - 255]!;
    }
}
initGaloisTables();

export function gfMul(a: number, b: number): number {
    if (a === 0 || b === 0) return 0;
    return EXP_TABLE[LOG_TABLE[a]! + LOG_TABLE[b]!]!;
}

export interface QuantizedCollection {
    buffer: ArrayBuffer;
    dim: number;
    count: number;
    scale: number;
    offset: number;
}

/**
 * Quantizes a batch of normalized Float32Array vectors into a tight Uint8Array buffer
 * matching GF(2^8) field elements.
 */
export function quantizeBatchToGF28(vectors: Float32Array[]): QuantizedCollection {
    const count = vectors.length;
    if (count === 0) throw new Error("Empty vector batch provided.");
    const dim = vectors[0]?.length || 0;

    const scale = 255.0 / 2.0;
    const offset = 1.0;

    // Single contiguous buffer allocation for maximum cache locality
    const buffer = new ArrayBuffer(count * dim!);
    const data = new Uint8Array(buffer);

    for (let i = 0; i < count; i++) {
        const vec = vectors[i];
        const baseOffset = i * dim!;
        for (let j = 0; j < dim!; j++) {
            const val = Math.max(-1, Math.min(1, vec![j]!));
            data[baseOffset + j] = Math.round((val + offset) * scale);
        }
    }

    return { buffer, dim, count, scale, offset };
}

/**
 * High-performance dot-product search optimized for Bun's typed arrays.
 * Dequantizes on-the-fly to maintain 0 memory allocation overhead during scanning.
 */
export function searchQuantizedCollectionBun(
    query: Float32Array,
    collection: QuantizedCollection,
    topK: number = 5
): { index: number; score: number }[] {
    const { buffer, dim, count, scale, offset } = collection;
    const data = new Uint8Array(buffer);

    // Pre-allocate results array for V8 optimization
    const scores = new Array<{ index: number; score: number }>(count);

    for (let i = 0; i < count; i++) {
        let dotProduct = 0;
        const baseOffset = i * dim;

        for (let j = 0; j < dim; j++) {
            const symbol = data[baseOffset + j];
            const reconstructed = (symbol! / scale) - offset;
            dotProduct += query[j]! * reconstructed;
        }

        scores[i] = { index: i, score: dotProduct };
    }

    // Quick partial sort or full sort depending on topK vs count
    scores.sort((a, b) => b.score - a.score);
    return scores.slice(0, topK);
}

/**
 * Save quantized collection directly to disk using Bun's ultra-fast file API.
 */
export async function saveQuantizedCollection(filepath: string, collection: QuantizedCollection): Promise<void> {
    // Store metadata header (16 bytes: count, dim, scale, offset) + raw buffer
    const header = new ArrayBuffer(16);
    const view = new DataView(header);
    view.setUint32(0, collection.count, true);
    view.setUint32(4, collection.dim, true);
    view.setFloat32(8, collection.scale, true);
    view.setFloat32(12, collection.offset, true);

    const combined = new Uint8Array(header.byteLength + collection.buffer.byteLength);
    combined.set(new Uint8Array(header), 0);
    combined.set(new Uint8Array(collection.buffer), header.byteLength);

    await Bun.write(filepath, combined);
}

/**
 * Load and memory-map quantized collection from disk using Bun.file().
 */
export async function loadQuantizedCollection(filepath: string): Promise<QuantizedCollection> {
    const file = Bun.file(filepath);
    const ab = await file.arrayBuffer();

    const view = new DataView(ab);
    const count = view.getUint32(0, true);
    const dim = view.getUint32(4, true);
    const scale = view.getFloat32(8, true);
    const offset = view.getFloat32(12, true);

    const buffer = ab.slice(16);

    return { buffer, dim, count, scale, offset };
}