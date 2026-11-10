//import { join } from "node:path";

const EMBEDDING_DIM = 1536;
const BYTES_PER_FLOAT = 4;
const RECORD_SIZE = (EMBEDDING_DIM * BYTES_PER_FLOAT) + 4; // 4-byte cluster header + float vector

// tok_00001_HectorDoor.png.token
// tok_00002_HectorWelcome.png.token
// tok_00007_TonkaFam.png
// tok_00008_TonkaSparta.png
// tok_00009_TonkasFamPOWER_Ad_MID.png
const v1 = await readVector("tok_00007_TonkaFam.png.token", 0);
const v2 = await readVector("tok_00009_TonkasFamPOWER_Ad_MID.png.token", 0);
const v3 = await readVector("tok_00008_TonkaSparta.png.token", 0);
const sim1 = cosineSimilaritySafe(v1, v2);
const sim2 = cosineSimilaritySafe(v1, v3);
console.log(`Cosine Similarity 1: ${sim1}`);
console.log(`Cosine Similarity 2: ${sim2}`);


const [v1x, v2x, v3x] = await Promise.all([
    readVector("tok_00007_TonkaFam.png.token"),
    readVector("tok_00009_TonkasFamPOWER_Ad_MID.png.token"),
    readVector("tok_00008_TonkaSparta.png.token")
]);

const sim1x = cosineSimilaritySafe(v1x, v2x);
const sim2x = cosineSimilaritySafe(v1x, v3x);

console.log(sim1x, sim2x);

/**
 * Reads a VQ embedding vector from a binary token file.
 * @param filePath Absolute or relative path to the token file
 * @param recordIndex The index of the record to read (defaults to 0)
 */
async function readVector(
    filePath: string,
    recordIndex = 0
): Promise<Float32Array> {

    if (!Number.isSafeInteger(recordIndex) || recordIndex < 0) {
        throw new RangeError("Invalid record index");
    }

    const file = Bun.file(filePath);

    const start = recordIndex * RECORD_SIZE;
    const end = start + RECORD_SIZE;

    if (!Number.isSafeInteger(end) || end > file.size) {
        throw new RangeError(
            `Record ${recordIndex} exceeds file size (${file.size} bytes)`
        );
    }

    const buffer = await file.slice(start, end).arrayBuffer();

    if (buffer.byteLength !== RECORD_SIZE) {
        throw new Error("Incomplete vector record");
    }

    const view = new DataView(buffer);

    // Assuming unsigned 32-bit cluster IDs.
    const clusterId = view.getUint32(0, true);

    // Zero-copy view into the same ArrayBuffer.
    const vector = new Float32Array(
        buffer,
        4,
        EMBEDDING_DIM
    );

    console.log(`Cluster ID: ${clusterId}`);

    return vector;
}

function cosineSimilaritySafe(a: Float32Array, b: Float32Array): number {
    if (a.length !== b.length) {
        throw new Error("Vectors must have identical dimensions");
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        const valA = a[i]!;
        const valB = b[i]!;

        // Skip if either value is NaN or invalid
        if (Number.isNaN(valA) || Number.isNaN(valB)) continue;

        dotProduct += valA * valB;
        normA += valA * valA;
        normB += valB * valB;
    }

    const denominator = Math.sqrt(normA) * Math.sqrt(normB);
    if (denominator === 0 || Number.isNaN(denominator)) {
        return 0; // Fallback to 0 similarity if norms are zero or invalid
    }

    return dotProduct / denominator;
}

// Example usage with your VQ embedding vectors:
// const sim = cosineSimilarity(vectorA, vectorB);
// console.log(`Cosine Similarity: ${sim.toFixed(4)}`);

function l2NormalizeInPlace(vector: Float32Array): Float32Array {
    let sumSquares = 0;

    // 1. Calculate sum of squared elements
    for (const val of vector) {
        sumSquares += val * val;
    }

    const norm = Math.sqrt(sumSquares);

    // Prevent division by zero for zero-vectors
    if (norm === 0) return vector;

    // 2. Divide each element by the L2 norm
    for (let i = 0; i < vector.length; i++) {
        vector[i] = (vector[i]!) / norm;
    }

    return vector;
}

function dotProductSimilarity(normA: Float32Array, normB: Float32Array): number {
    if (normA.length !== normB.length) {
        throw new Error("Vectors must have identical dimensions");
    }

    let dot = 0;
    for (let i = 0; i < normA.length; i++) {
        dot += (normA[i]!) * (normB[i]!);
    }

    return dot; // This equals the cosine similarity!
}