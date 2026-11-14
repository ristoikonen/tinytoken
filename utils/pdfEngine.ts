/**
 * Native Bun + PDFium WASM (`@hyzyla/pdfium`) Real PDF Page Renderer & Structured Text Extractor
 * (utils/pdfEngine.ts)
 *
 * Replaces regex operator scraping and synthetic SVG cover cards with Google Chrome's PDFium WASM engine:
 * - Extracts clean, Unicode-mapped text per page via `page.getText()` (zero kerning garbage).
 * - Renders actual PDF pages at 1.5x scale into raw RGBA bitmaps and pipes them through `sharp` to produce:
 *   1. Portrait-aspect 320px-wide WebP page previews (`~5 - 14 KB`, Section III)
 *   2. True 16x16 RGB VQ codebook grids (`256 B`, Section I)
 *   3. Real Meta DINOv3 ViT-S/16 384-byte `[CLS]` visual embeddings (`Section II`)
 */

import { PDFiumLibrary } from "@hyzyla/pdfium";
import sharp from "sharp";
import { EmbeddingModelId, PreviewFormat, getClipTextLimit } from "../models/token";
import { extractRealDinoEmbedding } from "./dinoEmbedding";
import { extractRealBgeEmbedding } from "./bgeEmbedding";
import { boxDownsampleRgbTo16x16 } from "./imagePreview";
import { encodeRgb16x16ToVqGrid } from "./vqCodebook";

const MAX_PAGES = 3;
const MAX_PREVIEW_BYTES = 16384;

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
  bgeEmbedding?: Int8Array<ArrayBuffer>;
  embeddingModelId: EmbeddingModelId;
  bgeModelId?: EmbeddingModelId;
  timing?: {
    sharpNs: number;
    vqNs: number;
    dinoNs: number;
  };
}

export type PageBundle = RenderedPdfPageBundle;

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
 * Renders selected pages of a PDF using `@hyzyla/pdfium` + `sharp` + `dinoEmbedding`.
 */
export async function renderPdfPagesWithPdfium(
  pdfBytes: Uint8Array,
  maxPagesPerDoc = MAX_PAGES,
  maxPreviewBytes = MAX_PREVIEW_BYTES,
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
        .slice(0, getClipTextLimit());

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

      const tSharp0 = Bun.nanoseconds();

      // 1. Single decode: resize and flatten PDF page to 384x384 raw RGB buffer with cubic resampling
      const dinoRgbResult = await sharp(rawRgba, rawInput)
        .flatten({ background: { r: 255, g: 255, b: 255 } })
        .resize(384, 384, { fit: "inside", withoutEnlargement: true, kernel: "cubic" })
        .removeAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });

      // 2. WebP preview encoded directly from the 384x384 raw RGB buffer (effort: 2, smartSubsample: false)
      let finalWebp = await sharp(dinoRgbResult.data, {
        raw: {
          width: dinoRgbResult.info.width,
          height: dinoRgbResult.info.height,
          channels: 3,
        },
      })
        .resize(320, 452, { fit: "inside", withoutEnlargement: true })
        .webp({ quality: 78, smartSubsample: false, effort: 2 })
        .toBuffer();

      if (finalWebp.byteLength > maxPreviewBytes) {
        finalWebp = await sharp(dinoRgbResult.data, {
          raw: {
            width: dinoRgbResult.info.width,
            height: dinoRgbResult.info.height,
            channels: 3,
          },
        })
          .resize(240, 340, { fit: "inside", withoutEnlargement: true })
          .webp({ quality: 68, smartSubsample: false, effort: 2 })
          .toBuffer();
      }
      const sharpNs = Bun.nanoseconds() - tSharp0;

      const previewBytes = new Uint8Array(new ArrayBuffer(finalWebp.byteLength));
      previewBytes.set(new Uint8Array(finalWebp.buffer, finalWebp.byteOffset, finalWebp.byteLength));

      // 3. 16x16 extraction via pure Bun typed-array 24x24 box average (no Sharp enc/dec)
      const tVq0 = Bun.nanoseconds();
      const rgb16Buf = boxDownsampleRgbTo16x16(
        dinoRgbResult.data,
        dinoRgbResult.info.width,
        dinoRgbResult.info.height,
      );
      const vqIndices = encodeRgb16x16ToVqGrid(rgb16Buf);
      const vqNs = Bun.nanoseconds() - tVq0;

      const tDino0 = Bun.nanoseconds();
      const dinoRes = await extractRealDinoEmbedding(
        new Uint8Array(
          dinoRgbResult.data.buffer,
          dinoRgbResult.data.byteOffset,
          dinoRgbResult.data.byteLength,
        ),
        dinoRgbResult.info.width,
        dinoRgbResult.info.height,
      );
      const dinoNs = Bun.nanoseconds() - tDino0;

      const bge = cleanText.trim() ? await extractRealBgeEmbedding(cleanText) : null;

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
        bgeEmbedding: bge?.embedding,
        embeddingModelId: dinoRes.modelId,
        bgeModelId: bge?.modelId,
        timing: {
          sharpNs,
          vqNs,
          dinoNs,
        },
      });
    }

    return bundles;
  } finally {
    doc.destroy();
  }
}

