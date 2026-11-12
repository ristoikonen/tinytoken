
/**
 * Real BGE-small-en-v1.5 ONNX
 * (`Xenova/bge-small-en-v1.5`, 384-Dim Int8 CLS)
 * Semantic Text Embedding Engine
 * + Synchronous lexical hashing fallback.
 *
 * - Primary (`EmbeddingModelId.BGE_SMALL_EN_V15_Q8 = 0x04`):
 *   Runs BAAI's BERT-based BGE-small-en-v1.5 sentence embedding
 *   model via `@huggingface/transformers` + ONNX Runtime.
 *
 *   Uses q8 model weights cached in `./.cache/transformers`,
 *   extracts the 384-dimensional CLS representation,
 *   L2-normalizes it and scalar-quantizes to Int8Array(384).
 *
 * - Fallback (`EmbeddingModelId.BGE_TEXT_FALLBACK = 0x03`):
 *   Synchronous 384-byte hashed lexical descriptor for
 *   offline/mock tests. Not a neural semantic embedding.
 *
 * Model spaces are distinct from DINOv3 and from each other.
 */

import { env, pipeline } from "@huggingface/transformers";
import { EmbeddingModelId } from "../models/token";
import {
    quantizeFloat32ToInt8,
    computeDinoCosineSimilarity,
} from "./dinoEmbedding"; // Adjust to your actual file path

export const BGE_EMBEDDING_DIMS = 384;
export const BGE_MAX_TOKENS = 512;
export const BGE_MODEL =
    "Xenova/bge-small-en-v1.5";


env.cacheDir = "./.cache/transformers";

export interface BgeEmbeddingResult {
    embedding: Int8Array<ArrayBuffer>;
    modelId: EmbeddingModelId;
}

// Singleton ONNX feature-extraction pipeline promise.
// eslint-disable-next-line @typescript-eslint/no-explicit-any
//let bgeExtractorPromise: Promise<any> | null = null;
let bgeExtractorPromise: ReturnType<typeof pipeline<"feature-extraction">> | null = null;

async function getBgeExtractor() {
    if (!bgeExtractorPromise) {
        bgeExtractorPromise = pipeline(
            "feature-extraction",
            BGE_MODEL,
            {
                dtype: "q8",
                device: "cpu",
            },
        ).catch((err) => {
            // Allow subsequent attempts after a loading failure.
            bgeExtractorPromise = null;
            throw err;
        });
    }

    return bgeExtractorPromise;
}

/**
 * Runs real BGE-small-en-v1.5 inference on UTF-8 text.
 *
 * Returns a 384-byte Int8 CLS semantic embedding.
 *
 * Input is truncated to the model's maximum sequence length.
 * For long documents, chunk and aggregate separately.
 */
export async function extractRealBgeEmbedding(
    text: string,
): Promise<BgeEmbeddingResult> {

    if (!text.trim()) {
        throw new Error(
            "BGE embedding requires non-empty text.",
        );
    }

    try {
        const extractor = await getBgeExtractor();

        const output = await extractor(text, {
            pooling: "cls",
            normalize: true,
        });

        const rawData = output.data as Float32Array;

        if (rawData.length !== BGE_EMBEDDING_DIMS) {
            throw new Error(
                `Invalid BGE output dimensions: ${rawData.length}`,
            );
        }

        const clsFloat32 = Float32Array.from(rawData);

        return {
            embedding: quantizeFloat32ToInt8(clsFloat32),
            modelId: EmbeddingModelId.BGE_SMALL_EN_V15_Q8,
        };

    } catch (err) {
        console.warn(
            "BGE ONNX fallback triggered:",
            err,
        );

        return {
            embedding: extractBgeTextFallback(text),
            modelId: EmbeddingModelId.BGE_TEXT_FALLBACK,
        };
    }
}

export const extractRealBertEmbedding =
    extractRealBgeEmbedding;

/**
 * Synchronous deterministic lexical fallback.
 *
 * Uses signed feature hashing over normalized words
 * and adjacent word pairs.
 *
 * Produces a 384-byte descriptor, but it is NOT
 * equivalent to a trained BGE semantic embedding.
 */
export function extractBgeTextFallback(
    text: string,
): Int8Array<ArrayBuffer> {

    const vec = new Float32Array(BGE_EMBEDDING_DIMS);

    const words =
        text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? [];

    if (words.length === 0) {
        return new Int8Array(
            new ArrayBuffer(BGE_EMBEDDING_DIMS),
        );
    }

    function hashString(value: string): number {
        let hash = 2166136261;

        for (let i = 0; i < value.length; i++) {
            hash ^= value.charCodeAt(i);
            hash = Math.imul(hash, 16777619);
        }

        return hash >>> 0;
    }

    function addFeature(
        feature: string,
        weight: number,
    ): void {

        const hash = hashString(feature);
        const index = hash % BGE_EMBEDDING_DIMS;
        const sign = (hash & 0x80000000) ? -1 : 1;

        vec[index] =
            (vec[index] ?? 0) + sign * weight;
    }

    // Unigram features
    for (const word of words) {
        addFeature(`w:${word}`, 1);
    }

    // Bigram features retain some local word order.
    for (let i = 1; i < words.length; i++) {
        addFeature(
            `b:${words[i - 1]}:${words[i]}`,
            0.5,
        );
    }

    return quantizeFloat32ToInt8(vec);
}

/**
 * Cosine similarity between two 384-byte BGE embeddings.
 *
 * Only compare vectors with compatible model IDs
 * and identical preprocessing/quantization versions.
 */
export function computeBgeCosineSimilarity(
    a: Int8Array,
    b: Int8Array,
): number {

    if (
        a.length !== BGE_EMBEDDING_DIMS ||
        b.length !== BGE_EMBEDDING_DIMS
    ) {
        throw new Error(
            "BGE cosine similarity requires two 384-byte vectors.",
        );
    }

    return computeDinoCosineSimilarity(a, b);
}

/**
 * Hex encoding using Bun's native Uint8Array.toHex().
 */
export function bgeEmbeddingToHex(
    embedding: Int8Array,
): string {

    if (embedding.length !== BGE_EMBEDDING_DIMS) {
        throw new Error("Invalid BGE embedding length.");
    }

    return new Uint8Array(
        embedding.buffer,
        embedding.byteOffset,
        embedding.byteLength,
    ).toHex();
}

/**
 * Decode exactly 384 bytes from hex.
 */
export function hexToBgeEmbedding(
    hex: string,
): Int8Array<ArrayBuffer> {

    const clean = hex.trim();

    if (clean.length !== BGE_EMBEDDING_DIMS * 2) {
        throw new Error(
            `Expected ${BGE_EMBEDDING_DIMS * 2} hex characters.`,
        );
    }

    const bytes = Uint8Array.fromHex(clean);

    if (bytes.length !== BGE_EMBEDDING_DIMS) {
        throw new Error("Invalid BGE embedding.");
    }

    const result = new Int8Array(
        new ArrayBuffer(BGE_EMBEDDING_DIMS),
    );

    result.set(
        new Int8Array(
            bytes.buffer,
            bytes.byteOffset,
            bytes.byteLength,
        ),
    );

    return result;
}
