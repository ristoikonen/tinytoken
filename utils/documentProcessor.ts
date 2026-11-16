/**
 * Format-Aware Document Processor Engine (utils/documentProcessor.ts)
 *
 * Provides dedicated processors for:
 * 1. PDF documents (via `@hyzyla/pdfium` with failure fallback)
 * 2. DOCX documents (via native `Bun.inflateSync` with 20 MB zip-bomb ceiling guard)
 * 3. Plain Text / Markdown / Code documents (via UTF-8 stream with visual page card rendering)
 * 4. Fallback binary document handler
 */

import {
  AllowedFileTypes,
  DEFAULT_DOC_PAGE_HEIGHT,
  DEFAULT_DOC_PAGE_WIDTH,
  EmbeddingModelId,
  PreviewFormat,
  getClipTextLimit,
  getMaxDecompressedDocxBytes,
  getMaxDocxMb,
  getMaxPagesPerDoc,
} from "../models/token";
import { extensionToMimeType, rasterizeAssetToWebpAndTensors } from "./imagePreview";
import { type PageBundle, renderPdfPagesWithPdfium } from "./pdfEngine";
import { extractRealBgeEmbedding } from "./bgeEmbedding";


/**
 * Maximum decompressed size ceiling for DOCX `word/document.xml` to prevent zip-bomb attacks.
 * Configurable via APP_MAX_DOCX_MB in .env (defaults to 20 MB).
 */
export const MAXIMUM_DECOMPRESSED_DOCX = getMaxDecompressedDocxBytes();
export const maximum_decompressed_DOCX = MAXIMUM_DECOMPRESSED_DOCX;

export type { PageBundle };

export interface ExtractedDocumentInfo {
  totalPages: number;
  width: number;
  height: number;
  extractedText: string;
  isOversized?: boolean;
}


export interface TextEmbeddingResult {
  embedding: Int8Array<ArrayBuffer>;
  modelId: EmbeddingModelId;
}


/**
 * Core interface for format-aware document parsing & page rendering.
 */
export interface DocumentProcessor {
  supports(extension: string): boolean;

  extract(
    bytes: Uint8Array,
    filename: string,
  ): Promise<ExtractedDocumentInfo>;

  renderPages(
    bytes: Uint8Array,
    maxPages?: number,
    filename?: string,
  ): Promise<PageBundle[]>;

  embedText?(
    bytes: Uint8Array,
    filename: string,
  ): Promise<TextEmbeddingResult | null>;
}

export async function embedDocumentText(
  processor: DocumentProcessor,
  bytes: Uint8Array,
  filename: string,
): Promise<TextEmbeddingResult | null> {
  const info = await processor.extract(bytes, filename);
  const text = info.extractedText.trim();
  if (!text || info.isOversized) return null;
  const bge = await extractRealBgeEmbedding(text);
  return {
    embedding: bge.embedding,
    modelId: bge.modelId,
  };
}

/**
 * PDF Document Processor: handles multi-page rendering and Unicode text extraction via PDFium.
 */
export class PdfDocumentProcessor implements DocumentProcessor {
  public supports(extension: string): boolean {
    const ext = extension.startsWith(".") ? extension.toLowerCase() : `.${extension.toLowerCase()}`;
    return ext === ".pdf";
  }

  public async extract(bytes: Uint8Array, filename: string): Promise<ExtractedDocumentInfo> {
    try {
      const bundles = await renderPdfPagesWithPdfium(bytes, 1);
      if (bundles.length > 0) {
        const first = bundles[0]!;
        return {
          totalPages: first.totalPages,
          width: first.width,
          height: first.height,
          extractedText: first.extractedText || `PDF Document: ${filename}`,
        };
      }
    } catch {
      // PDFium threw (e.g. malformed or corrupted PDF)
    }

    return {
      totalPages: 1,
      width: DEFAULT_DOC_PAGE_WIDTH,
      height: DEFAULT_DOC_PAGE_HEIGHT,
      extractedText: `PDF Document: ${filename}`,
    };
  }

  public async renderPages(bytes: Uint8Array, maxPages?: number): Promise<PageBundle[]> {
    return renderPdfPagesWithPdfium(bytes, maxPages ?? getMaxPagesPerDoc());
  }

  public async embedText(bytes: Uint8Array, filename: string): Promise<TextEmbeddingResult | null> {
    return embedDocumentText(this, bytes, filename);
  }
}

/**
 * DOCX Document Processor: extracts `word/document.xml` with a 20 MB uncompressed size guard against zip-bombs.
 */
export class DocxDocumentProcessor implements DocumentProcessor {
  public supports(extension: string): boolean {
    const ext = extension.startsWith(".") ? extension.toLowerCase() : `.${extension.toLowerCase()}`;
    return ext === ".docx";
  }

  public async extract(bytes: Uint8Array, filename: string): Promise<ExtractedDocumentInfo> {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    let ptr = 0;
    let docxText = "";
    let pageEstimate = 1;
    let isOversized = false;

    while (ptr + 30 < bytes.length && view.getUint32(ptr, true) === 0x04034b50) {
      const method = view.getUint16(ptr + 8, true);
      const compSize = view.getUint32(ptr + 18, true);
      const uncompSize = view.getUint32(ptr + 22, true);
      const nameLen = view.getUint16(ptr + 26, true);
      const extraLen = view.getUint16(ptr + 28, true);
      const entryName = new TextDecoder().decode(bytes.subarray(ptr + 30, ptr + 30 + nameLen));
      const dataStart = ptr + 30 + nameLen + extraLen;

      if (entryName === "word/document.xml" && dataStart + compSize <= bytes.length) {
        const maxDocxBytes = getMaxDecompressedDocxBytes();
        const maxDocxMb = getMaxDocxMb();
        // Zip-bomb guard: check claimed uncompressed size
        if (uncompSize > maxDocxBytes) {
          isOversized = true;
          docxText = `DOCX uncompressed size over max size (${(uncompSize / (1024 * 1024)).toFixed(1)} MB > ${maxDocxMb} MB) | DOCX uncompressed size over max size`;
          break;
        }

        try {
          const slice = bytes.subarray(dataStart, dataStart + compSize);
          const xmlBytes = method === 8 ? Bun.inflateSync(new Uint8Array(slice)) : slice;
          const xmlStr = new TextDecoder().decode(xmlBytes);
          pageEstimate = Math.max(1, (xmlStr.match(/<w:lastRenderedPageBreak\b/g) || []).length + 1);
          docxText = xmlStr
            .replace(/<[^>]+>/g, " ")
            .replace(/[^\x20-\x7E]+/g, " ")
            .replace(/\s+/g, " ")
            .trim()
            .slice(0, getClipTextLimit());
        } catch {
          // Fallback below
        }
        break;
      }
      if (compSize === 0) break;
      ptr = dataStart + compSize;
    }

    return {
      totalPages: pageEstimate,
      width: DEFAULT_DOC_PAGE_WIDTH,
      height: DEFAULT_DOC_PAGE_HEIGHT,
      extractedText: docxText || `DOCX Document asset: ${filename}`,
      isOversized,
    };
  }

  public async renderPages(
    bytes: Uint8Array,
    _maxPages?: number,
    filenameParam?: string,
  ): Promise<PageBundle[]> {
    const filename = filenameParam || "document.docx";
    const info = await this.extract(bytes, filename);
    const bundle = await rasterizeAssetToWebpAndTensors(
      bytes,
      ".docx",
      filename,
      1,
      info.totalPages,
      {
        width: info.width,
        height: info.height,
        excerpt: info.extractedText,
      },
    );

    const bge = info.extractedText.trim()
      ? await extractRealBgeEmbedding(info.extractedText)
      : null;

    return [
      {
        page: 1,
        totalPages: info.totalPages,
        width: info.width,
        height: info.height,
        extractedText: info.extractedText,
        previewBytes: bundle.previewBytes,
        previewFormat: bundle.previewFormat,
        vqIndices: bundle.vqIndices,
        dinoEmbedding: bundle.dinoEmbedding,
        bgeEmbedding: bge?.embedding,
        embeddingModelId: bundle.embeddingModelId,
        bgeModelId: bge?.modelId,
        timing: bundle.timing,
      },
    ];
  }

  public async embedText(bytes: Uint8Array, filename: string): Promise<TextEmbeddingResult | null> {
    return embedDocumentText(this, bytes, filename);
  }
}

/**
 * Text Document Processor: renders clean visual document page previews for plain-text, markdown, and config files.
 */
export class TextDocumentProcessor implements DocumentProcessor {
  private static readonly TEXT_EXTENSIONS = AllowedFileTypes.TEXT_DOCUMENT_SET;

  public supports(extension: string): boolean {
    return AllowedFileTypes.isTextDocument(extension);
  }

  public async extract(bytes: Uint8Array, filename: string): Promise<ExtractedDocumentInfo> {
    const clipLimit = getClipTextLimit();
    const maxSample = Math.min(bytes.length, Math.max(8192, clipLimit * 4));
    const sample = bytes.subarray(0, maxSample);
    const decoded = new TextDecoder("utf-8", { fatal: false }).decode(sample);
    const lines = decoded.split(/\r?\n/);
    const totalLinesEstimate = Math.max(
      1,
      Math.round((lines.length * bytes.length) / Math.max(1, sample.length)),
    );
    const totalPages = Math.max(1, Math.ceil(totalLinesEstimate / 45));

    const cleanSnippet = decoded
      .replace(/[^\x20-\x7E\n\r\t]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, clipLimit);

    return {
      totalPages,
      width: DEFAULT_DOC_PAGE_WIDTH,
      height: DEFAULT_DOC_PAGE_HEIGHT,
      extractedText: cleanSnippet || `Text Document: ${filename}`,
    };
  }

  public async renderPages(
    bytes: Uint8Array,
    _maxPages?: number,
    filenameParam?: string,
  ): Promise<PageBundle[]> {
    const filename = filenameParam || "document.txt";
    const info = await this.extract(bytes, filename);
    const bundle = await rasterizeAssetToWebpAndTensors(
      bytes,
      ".txt",
      filename,
      1,
      info.totalPages,
      {
        width: info.width,
        height: info.height,
        excerpt: info.extractedText,
      },
    );

    const bge = info.extractedText.trim()
      ? await extractRealBgeEmbedding(info.extractedText)
      : null;

    return [
      {
        page: 1,
        totalPages: info.totalPages,
        width: info.width,
        height: info.height,
        extractedText: info.extractedText,
        previewBytes: bundle.previewBytes,
        previewFormat: bundle.previewFormat,
        vqIndices: bundle.vqIndices,
        dinoEmbedding: bundle.dinoEmbedding,
        bgeEmbedding: bge?.embedding,
        embeddingModelId: bundle.embeddingModelId,
        bgeModelId: bge?.modelId,
        timing: bundle.timing,
      },
    ];
  }

  public async embedText(bytes: Uint8Array, filename: string): Promise<TextEmbeddingResult | null> {
    return embedDocumentText(this, bytes, filename);
  }
}

/**
 * Fallback Document Processor: handles legacy or unknown binary document formats (e.g. `.doc`).
 */
export class FallbackDocumentProcessor implements DocumentProcessor {
  public supports(): boolean {
    return true;
  }

  public async extract(bytes: Uint8Array, filename: string): Promise<ExtractedDocumentInfo> {
    const clipLimit = getClipTextLimit();
    const maxSample = Math.min(bytes.length, Math.max(4096, clipLimit * 4));
    const textPreview = new TextDecoder("utf-8", { fatal: false })
      .decode(bytes.subarray(0, maxSample))
      .replace(/[^\x20-\x7E]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, clipLimit);

    return {
      totalPages: 1,
      width: DEFAULT_DOC_PAGE_WIDTH,
      height: DEFAULT_DOC_PAGE_HEIGHT,
      extractedText: textPreview || `Document asset: ${filename}`,
    };
  }

  public async renderPages(
    bytes: Uint8Array,
    _maxPages?: number,
    filenameParam?: string,
  ): Promise<PageBundle[]> {
    const filename = filenameParam || "document.doc";
    const info = await this.extract(bytes, filename);
    const bundle = await rasterizeAssetToWebpAndTensors(
      bytes,
      ".doc",
      filename,
      1,
      info.totalPages,
      {
        width: info.width,
        height: info.height,
        excerpt: info.extractedText,
      },
    );

    const bge = info.extractedText.trim()
      ? await extractRealBgeEmbedding(info.extractedText)
      : null;

    return [
      {
        page: 1,
        totalPages: 1,
        width: info.width,
        height: info.height,
        extractedText: info.extractedText,
        previewBytes: bundle.previewBytes,
        previewFormat: bundle.previewFormat,
        vqIndices: bundle.vqIndices,
        dinoEmbedding: bundle.dinoEmbedding,
        bgeEmbedding: bge?.embedding,
        embeddingModelId: bundle.embeddingModelId,
        bgeModelId: bge?.modelId,
        timing: bundle.timing,
      },
    ];
  }

  public async embedText(bytes: Uint8Array, filename: string): Promise<TextEmbeddingResult | null> {
    return embedDocumentText(this, bytes, filename);
  }
}

/**
 * Registered list of active document processors in evaluation priority order.
 */
export const DOCUMENT_PROCESSORS: DocumentProcessor[] = [
  new PdfDocumentProcessor(),
  new DocxDocumentProcessor(),
  new TextDocumentProcessor(),
  new FallbackDocumentProcessor(),
];

/**
 * Resolves the appropriate DocumentProcessor for a given file extension.
 */
export function getDocumentProcessor(extension: string): DocumentProcessor {
  return (
    DOCUMENT_PROCESSORS.find((p) => p.supports(extension)) ??
    DOCUMENT_PROCESSORS[DOCUMENT_PROCESSORS.length - 1]!
  );
}

