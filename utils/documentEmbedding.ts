
import { getDocumentProcessor } from "./documentProcessor";
import { type ExtractedDocumentInfo } from "./indexer";
import { extractRealBgeEmbedding } from "./bgeEmbedding";
import { extname } from "node:path";
import { EmbeddingModelId } from "../models/token";

export interface EmbeddedDocument {
    document: ExtractedDocumentInfo;
    textEmbedding: Int8Array<ArrayBuffer> | null;
    textEmbeddingModelId: EmbeddingModelId | null;
}

export async function embedDocument(
    bytes: Uint8Array,
    filename: string,
): Promise<EmbeddedDocument> {
    const processor = getDocumentProcessor(
        extname(filename).toLowerCase()
    );

    const document = await processor.extract(bytes, filename);
    const text = document.extractedText.trim();

    if (!text || document.isOversized) {
        return {
            document,
            textEmbedding: null,
            textEmbeddingModelId: null,
        };
    }

    const bge = await extractRealBgeEmbedding(text);

    return {
        document,
        textEmbedding: bge.embedding,
        textEmbeddingModelId: bge.modelId,
    };
}

export async function embedDocumentOld(
    bytes: Uint8Array,
    filename: string,
) {
    const processor = getDocumentProcessor(
        extname(filename).toLowerCase()
    );

    const document = await processor.extract(bytes, filename);

    const text = document.extractedText.trim();

    if (!text || document.isOversized) {
        return {
            document,
            textEmbedding: null,
        };
    }

    const bge = await extractRealBgeEmbedding(text);

    return {
        document,
        textEmbedding: bge.embedding,
        textEmbeddingModelId: bge.modelId,
    };
}
