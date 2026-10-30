/**
 * Native Bun + PDFium WASM (`@hyzyla/pdfium`) Real PDF Page Renderer & Structured Text Extractor
 * (utils/pdfEngine.ts)
 *
 * Replaces regex operator scraping and synthetic SVG cover cards with Google Chrome's PDFium WASM engine:
 * - Extracts clean, Unicode-mapped text per page via `page.getText()` (zero kerning garbage).
 * - Renders actual PDF pages at 1.5x scale into raw RGBA bitmaps and pipes them through `sharp` to produce:
 *   1. Portrait-aspect 320px-wide WebP page previews (`~5 - 14 KB`, Section III)
 *   2. True 16x16 RGB VQ codebook grids (`256 B`, Section I)
 *   3. Real `Xenova/dinov2-small` 384-byte `[CLS]` visual embeddings (`Section II`)
 */

import { PDFiumLibrary } from "@hyzyla/pdfium";
import sharp from "sharp";
import { EmbeddingModelId, PreviewFormat } from "../models/token";
import { extractRealDinoV2Embedding } from "./dinoEmbedding";
import { encodeRgb16x16ToVqGrid } from "./vqCodebook";

export interface RenderedPdfPageBundle {
  page: number; // 1-indexed page number
  totalPages: number;
  width: number;
  height: number;
  extractedText: string;
  previewBytes: Uint8Array<ArrayBuffer>;
  previewFormat: PreviewFormat;
  vqIndices: Uint8Array<ArrayBuffer>;
  dinoEmbedding: Int8Array<ArrayBuffer>;
  embeddingModelId: EmbeddingModelId;
}

let pdfiumPromise: Promise<PDFiumLibrary> | null = null;

async function getPdfiumLibrary(): Promise<PDFiumLibrary> {
  if (!pdfiumPromise) {
    pdfiumPromise = PDFiumLibrary.init();
  }
  return pdfiumPromise;
}

/**
 * Selects which 0-indexed pages of a PDF to tokenize so large books (e.g. 448 pages)
 * don't block startup while still generating real multi-page tokens.
 */
export function selectPdfPageIndices(totalPages: number, maxPagesPerDoc = 3): number[] {
  const count = Math.max(1, totalPages);
  const cap = Math.max(1, Math.min(count, maxPagesPerDoc));
  if (count <= cap) {
    return Array.from({ length: count }, (_, i) => i);
  }
  // Always include page 0 (cover/title) and page 1 (TOC/abstract), then stride evenly
  const indices: number[] = [0];
  if (cap >= 2) indices.push(1);
  for (let k = 2; k < cap; k++) {
    const idx = Math.min(count - 1, Math.floor((k * count) / cap));
    if (!indices.includes(idx)) {
      indices.push(idx);
    }
  }
  return indices;
}

/**
 * Renders selected pages of a PDF using `@hyzyla/pdfium` + `sharp` + `Xenova/dinov2-small`.
 */
export async function renderPdfPagesWithPdfium(
  pdfBytes: Uint8Array,
  maxPagesPerDoc = 3,
  maxPreviewBytes = 16384,
): Promise<RenderedPdfPageBundle[]> {
  const library = await getPdfiumLibrary();
  const doc = await library.loadDocument(pdfBytes);

  try {
    const totalPages = Math.max(1, doc.getPageCount());
    const pageIndices = selectPdfPageIndices(totalPages, maxPagesPerDoc);
    const bundles: RenderedPdfPageBundle[] = [];

    for (const pageIdx of pageIndices) {
      const page = doc.getPage(pageIdx);
      const rawText = page.getText() ?? "";
      const cleanText = rawText
        .replace(/\r\n/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 2200);

      // Render PDF page to raw RGBA bitmap via PDFium WASM
      let bitmapWidth = 612;
      let bitmapHeight = 792;
      const rendered = await page.render({
        scale: 1.5,
        render: async (options) => {
          bitmapWidth = options.width;
          bitmapHeight = options.height;
          return options.data;
        },
      });

      const rawRgba = rendered.data;
      const rawInput = {
        raw: {
          width: bitmapWidth,
          height: bitmapHeight,
          channels: 4 as const,
        },
      };

      const [webpBuf, rgb16Buf, dinoRgbResult] = await Promise.all([
        sharp(rawRgba, rawInput)
          .flatten({ background: { r: 255, g: 255, b: 255 } })
          .resize(320, 452, { fit: "inside", withoutEnlargement: true })
          .webp({ quality: 78, smartSubsample: true, effort: 4 })
          .toBuffer(),
        sharp(rawRgba, rawInput)
          .flatten({ background: { r: 255, g: 255, b: 255 } })
          .resize(16, 16, { fit: "fill", kernel: "lanczos3" })
          .removeAlpha()
          .raw()
          .toBuffer(),
        sharp(rawRgba, rawInput)
          .flatten({ background: { r: 255, g: 255, b: 255 } })
          .resize(384, 384, { fit: "inside", withoutEnlargement: true })
          .removeAlpha()
          .raw()
          .toBuffer({ resolveWithObject: true }),
      ]);

      let finalWebp = webpBuf;
      if (finalWebp.byteLength > maxPreviewBytes) {
        finalWebp = await sharp(rawRgba, rawInput)
          .flatten({ background: { r: 255, g: 255, b: 255 } })
          .resize(240, 340, { fit: "inside", withoutEnlargement: true })
          .webp({ quality: 68, smartSubsample: true, effort: 4 })
          .toBuffer();
      }

      const previewBytes = new Uint8Array(new ArrayBuffer(finalWebp.byteLength));
      previewBytes.set(new Uint8Array(finalWebp.buffer, finalWebp.byteOffset, finalWebp.byteLength));

      const vqIndices = encodeRgb16x16ToVqGrid(
        new Uint8Array(rgb16Buf.buffer, rgb16Buf.byteOffset, rgb16Buf.byteLength),
      );

      const dinoRes = await extractRealDinoV2Embedding(
        new Uint8Array(
          dinoRgbResult.data.buffer,
          dinoRgbResult.data.byteOffset,
          dinoRgbResult.data.byteLength,
        ),
        dinoRgbResult.info.width,
        dinoRgbResult.info.height,
      );

      bundles.push({
        page: pageIdx + 1,
        totalPages,
        width: bitmapWidth,
        height: bitmapHeight,
        extractedText: cleanText,
        previewBytes,
        previewFormat: PreviewFormat.WEBP,
        vqIndices,
        dinoEmbedding: dinoRes.embedding,
        embeddingModelId: dinoRes.modelId,
      });
    }

    return bundles;
  } finally {
    doc.destroy();
  }
}

