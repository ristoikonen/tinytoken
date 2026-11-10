import { EmbeddingModelId } from "../models/token";
import { extractTokenMetadata, inspectBinaryTokenHeader } from "../utils/tokenCodec";
import type { TokenMetadataPayload } from "../models/token";

// tok_00001_HectorDoor.png.token
// tok_00002_HectorWelcome.png.token
// tok_00007_TonkaFam.png
// tok_00008_TonkaSparta.png
// tok_00009_TonkasFamPOWER_Ad_MID.png
const [v1, v2, v3] = await Promise.all([
    readEmbedding("tok_00007_TonkaFam.png.token"),
    readEmbedding("tok_00009_TonkasFamPOWER_Ad_MID.png.token"),
    readEmbedding("tok_00008_TonkaSparta.png.token")
]);

if (
    v1.modelId !== v2.modelId ||
    v1.modelId !== v3.modelId ||
    v1.modelId === EmbeddingModelId.NONE
) {
    throw new Error("Tokens must contain embeddings from the same known model");
}

console.log(`Cosine Similarity 1: ${cosineSimilaritySafe(v1.embedding, v2.embedding).toFixed(4)}`);
console.log(`Cosine Similarity 2: ${cosineSimilaritySafe(v1.embedding, v3.embedding).toFixed(4)}`);


/**
 * Reads the signed-int8 semantic embedding stored in a token's metadata.
 */
async function readEmbedding(
    filePath: string,
): Promise<{ embedding: Int8Array; modelId: EmbeddingModelId }> {
    const bytes = new Uint8Array(await Bun.file(filePath).arrayBuffer());
    const header = inspectBinaryTokenHeader(bytes);
    const metadata: TokenMetadataPayload = extractTokenMetadata(bytes);
    const hex = metadata.dinoEmbeddingHex;

    if (!hex || !/^[\da-f]{768}$/i.test(hex)) {
        throw new Error(`${filePath} does not contain a valid 384-byte embedding`);
    }

    const embeddingBytes = Uint8Array.fromHex(hex);
    return {
        embedding: new Int8Array(
            embeddingBytes.buffer,
            embeddingBytes.byteOffset,
            embeddingBytes.byteLength
        ),
        modelId: header.embeddingModelId,
    };
}

function cosineSimilaritySafe(a: Int8Array, b: Int8Array): number {
    if (a.length !== b.length) {
        throw new Error("Vectors must have identical dimensions");
    }

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
        const valA = a[i]!;
        const valB = b[i]!;

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