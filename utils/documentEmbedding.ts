
import { getDocumentProcessor } from "./documentProcessor";
import { type ExtractedDocumentInfo } from "./indexer";
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
  const processor = getDocumentProcessor(extname(filename).toLowerCase());
  const document = await processor.extract(bytes, filename);
  const embResult = await processor.embedText?.(bytes, filename);

  return {
    document,
    textEmbedding: embResult?.embedding ?? null,
    textEmbeddingModelId: embResult?.modelId ?? null,
  };
}
