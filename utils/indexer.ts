/**
 * Native Bun Directory Scanner & Custom 25 KB Tokenizer (utils/indexer.ts)
 *
 * Implements `Win11TokenPipeline` using:
 * - Bun native I/O (`Bun.Glob`, `Bun.file`, `Bun.write`, `Bun.inflateSync`)
 * - Real Meta/Xenova DINOv2 ViT-S/14 ONNX (`q8`) 384-byte `[CLS]` embeddings (`EmbeddingModelId.DINOV2_VIT_S14_Q8 = 0x02`)
 * - Google Chrome PDFium WASM (`@hyzyla/pdfium`) for real multi-page PDF rasterization & clean Unicode text extraction
 * - `sharp` for 320px `smartSubsample` WebP previews and true 16x16 RGB VQ codebook grids
 */

import {
  type TokenMetadataPayload,
  type ZipTokenIndexEntry,
} from "../models/token";
import { dinoEmbeddingToHex } from "./dinoEmbedding";
import { rasterizeAssetToWebpAndTensors } from "./imagePreview";
import { renderPdfPagesWithPdfium } from "./pdfEngine";
import { decodeBinaryToken, encodeBinaryToken } from "./tokenCodec";
import { writeNativeTokenZipArchive } from "./zipEngine";

export interface IndexerConfig {
  sourceDirectory: string;
  documentDirectory?: string;
  outputZipPath: string;
  allowedExtensions: string[];
  maxTokenSizeKb: number;
  maxPagesPerDocument?: number;
}

export type ArchiveTokenItem = {
  entryName: string;
  tokenBytes: Uint8Array<ArrayBuffer>;
  meta: Omit<
    ZipTokenIndexEntry,
    | "localHeaderOffset"
    | "dataOffset"
    | "compressedSize"
    | "uncompressedSize"
    | "compressionMethod"
    | "crc32"
  >;
};

export interface ExtractedDocumentInfo {
  totalPages: number;
  width: number;
  height: number;
  extractedText: string;
}

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".svg"]);
const DOCUMENT_EXTENSIONS = new Set([".pdf", ".docx", ".doc", ".txt", ".md"]);

export class Win11TokenPipeline {
  constructor(private config: IndexerConfig) {}

  public setSourceDirectory(dir: string): void {
    this.config.sourceDirectory = dir;
  }

  public getSourceDirectory(): string {
    return this.config.sourceDirectory;
  }

  /**
   * Scans a directory using native `Bun.Glob` (zero node:fs imports) and filters by allowedExtensions.
   */
  public async scanDirectory(
    dir: string = this.config.sourceDirectory,
    filterSet?: ReadonlySet<string>,
  ): Promise<string[]> {
    const matchedFiles: string[] = [];
    const extSet =
      filterSet ?? new Set(this.config.allowedExtensions.map((e) => e.toLowerCase()));

    try {
      const glob = new Bun.Glob("**/*");
      for await (const relPath of glob.scan({ cwd: dir, onlyFiles: true })) {
        const normalizedRel = relPath.replace(/\\/g, "/");
        const dotIdx = normalizedRel.lastIndexOf(".");
        if (dotIdx !== -1) {
          const ext = normalizedRel.slice(dotIdx).toLowerCase();
          if (extSet.has(ext)) {
            matchedFiles.push(`${dir.replace(/\\/g, "/").replace(/\/$/, "")}/${normalizedRel}`);
          }
        }
      }
    } catch (err) {
      console.error(`Error scanning directory ${dir}:`, err);
    }

    matchedFiles.sort();
    return matchedFiles;
  }

  /**
   * Scans `documentDirectory` (`./documents`) for existing PDF/DOCX documents.
   */
  public async scanDocumentDirectory(): Promise<string[]> {
    const docDir = this.config.documentDirectory ?? "./documents";
    const allowedSet = new Set(
      this.config.allowedExtensions
        .map((e) => e.toLowerCase())
        .filter((e) => DOCUMENT_EXTENSIONS.has(e)),
    );
    return this.scanDirectory(docDir, allowedSet.size > 0 ? allowedSet : DOCUMENT_EXTENSIONS);
  }

  /**
   * Extracts DOCX (`word/document.xml`) or plain-text document info using native `Bun.inflateSync`.
   */
  public extractDocumentInfo(
    rawBytes: Uint8Array,
    filename: string,
    ext: string,
  ): ExtractedDocumentInfo {
    if (ext === ".docx") {
      const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
      let ptr = 0;
      let docxText = "";
      let pageEstimate = 1;
      while (ptr + 30 < rawBytes.length && view.getUint32(ptr, true) === 0x04034b50) {
        const method = view.getUint16(ptr + 8, true);
        const compSize = view.getUint32(ptr + 18, true);
        const nameLen = view.getUint16(ptr + 26, true);
        const extraLen = view.getUint16(ptr + 28, true);
        const entryName = new TextDecoder().decode(rawBytes.subarray(ptr + 30, ptr + 30 + nameLen));
        const dataStart = ptr + 30 + nameLen + extraLen;
        if (entryName === "word/document.xml" && dataStart + compSize <= rawBytes.length) {
          try {
            const slice = rawBytes.subarray(dataStart, dataStart + compSize);
            const xmlBytes = method === 8 ? Bun.inflateSync(new Uint8Array(slice)) : slice;
            const xmlStr = new TextDecoder().decode(xmlBytes);
            pageEstimate = Math.max(1, (xmlStr.match(/<w:lastRenderedPageBreak\b/g) || []).length + 1);
            docxText = xmlStr
              .replace(/<[^>]+>/g, " ")
              .replace(/[^\x20-\x7E]+/g, " ")
              .replace(/\s+/g, " ")
              .trim()
              .slice(0, 1200);
          } catch {
            // fallback below
          }
          break;
        }
        if (compSize === 0) break;
        ptr = dataStart + compSize;
      }
      return {
        totalPages: pageEstimate,
        width: 1240,
        height: 1754,
        extractedText: docxText || `DOCX Document asset: ${filename}`,
      };
    }

    const textPreview = new TextDecoder("utf-8", { fatal: false })
      .decode(rawBytes.subarray(0, 4096))
      .replace(/[^\x20-\x7E]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 800);
    return {
      totalPages: 1,
      width: 1240,
      height: 1754,
      extractedText: textPreview || `Document asset: ${filename}`,
    };
  }

  /**
   * Tokenizes a single image or SVG file from `sourceDirectory` (`./images`), generating:
   * - Lean 320x320 WebP raster preview (Section III)
   * - True decoded 16x16 RGB VQ codebook grid (Section I, 256B)
   * - Real 384B Int8 `Xenova/dinov2-small` ViT-S/14 CLS semantic embedding (Section II, modelId=0x02)
   */
  public async tokenizeImageFile(
    fullPath: string,
    tokenCounter: number,
  ): Promise<ArchiveTokenItem | null> {
    const bunFile = Bun.file(fullPath);
    if (!(await bunFile.exists())) return null;

    const rawBytes = new Uint8Array(await bunFile.arrayBuffer());
    const filename = fullPath.split("/").pop() ?? "asset.bin";
    const dotIdx = filename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : "";

    const id = `tok_${String(tokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const entryName = `${id}.token`;

    const bundle = await rasterizeAssetToWebpAndTensors(rawBytes, ext, filename, 1, 1);
    const dinoEmbeddingHex = dinoEmbeddingToHex(bundle.dinoEmbedding);

    const createdAt = new Date(bunFile.lastModified || Date.now()).toISOString();
    const svgTextNote = bundle.svgExtractedText ? ` | SVG Text: ${bundle.svgExtractedText}` : "";
    const ocrText =
      `Image Asset: ${filename} | Dimensions: ${bundle.width}x${bundle.height} (${bundle.colorDepth}-bit ${ext.replace(".", "").toUpperCase()} -> 320x320 WebP ${bundle.previewBytes.length}B)` +
      `${svgTextNote} | DINOv2-ViT-S14-384B + 16x16 VQ + 256B SHARK/SNARK. Source path: ${fullPath}`;

    const metadata: TokenMetadataPayload = {
      id,
      filename,
      sourcePath: `/api/image/${encodeURIComponent(filename)}`,
      mimeType: bunFile.type || (ext === ".svg" ? "image/svg+xml" : "image/jpeg"),
      page: 1,
      totalPages: 1,
      width: bundle.width,
      height: bundle.height,
      colorDepth: bundle.colorDepth,
      createdAt,
      ocrText,
      dinoEmbeddingHex,
      tags: ["local-image", ext.replace(".", ""), "dinov2-vit-s14", "webp-preview", "verified"],
    };

    const tokenBytes = await encodeBinaryToken({
      originalFileSize: rawBytes.length,
      width: bundle.width,
      height: bundle.height,
      page: 1,
      previewFormat: bundle.previewFormat,
      shardBlockId: Math.floor((tokenCounter - 1) / 64),
      shardIndex: (tokenCounter - 1) % 64,
      embeddingModelId: bundle.embeddingModelId,
      vqIndices: bundle.vqIndices,
      metadata,
      previewBytes: bundle.previewBytes,
    });

    const decoded = await decodeBinaryToken(tokenBytes);
    return {
      entryName,
      tokenBytes,
      meta: {
        id,
        entryName,
        filename,
        sourcePath: metadata.sourcePath,
        mimeType: metadata.mimeType,
        page: 1,
        totalPages: 1,
        width: bundle.width,
        height: bundle.height,
        originalFileSize: rawBytes.length,
        previewFormat: bundle.previewFormat,
        flags: decoded.flags,
        ocrText,
        createdAt,
        merkleRootHex: decoded.merkleRootHex,
        snarkVerified: decoded.snarkAttestation.valid,
        vqIndices: bundle.vqIndices,
        dinoEmbedding: bundle.dinoEmbedding,
        embeddingModelId: bundle.embeddingModelId,
      },
    };
  }

  /**
   * Tokenizes a document file from `documentDirectory` (`./documents`):
   * - For `.pdf` files: uses `@hyzyla/pdfium` (`renderPdfPagesWithPdfium`) to render real portrait
   *   page bitmaps to 320x452 WebP, extract clean Unicode text via `page.getText()`, and emit
   *   multi-page tokens (`P.1..P.N`).
   * - For `.docx` files: extracts `word/document.xml` text and renders a clean cover preview.
   */
  public async tokenizeDocumentFile(
    fullPath: string,
    startTokenCounter: number,
  ): Promise<ArchiveTokenItem[]> {
    const bunFile = Bun.file(fullPath);
    if (!(await bunFile.exists())) return [];

    const rawBytes = new Uint8Array(await bunFile.arrayBuffer());
    const filename = fullPath.split("/").pop() ?? "document.pdf";
    const dotIdx = filename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : ".pdf";
    const createdAt = new Date(bunFile.lastModified || Date.now()).toISOString();
    const maxPagesPerDoc = this.config.maxPagesPerDocument ?? 3;

    if (ext === ".pdf") {
      try {
        const pageBundles = await renderPdfPagesWithPdfium(rawBytes, maxPagesPerDoc);
        const items: ArchiveTokenItem[] = [];
        let counter = startTokenCounter;

        for (const pg of pageBundles) {
          const seq = counter++;
          const safeFile = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
          const id = `tok_${String(seq).padStart(5, "0")}_p${pg.page}_${safeFile}`;
          const entryName = `${id}.token`;
          const dinoEmbeddingHex = dinoEmbeddingToHex(pg.dinoEmbedding);

          const snippetPart = pg.extractedText ? ` | Page Text: ${pg.extractedText}` : "";
          const ocrText =
            `PDF Page ${pg.page}/${pg.totalPages}: ${filename} (${(rawBytes.length / 1024).toFixed(1)} KB, ${pg.width}x${pg.height}px)` +
            `${snippetPart}`;

          const metadata: TokenMetadataPayload = {
            id,
            filename,
            sourcePath: `/api/document/${encodeURIComponent(filename)}#page=${pg.page}`,
            mimeType: "application/pdf",
            page: pg.page,
            totalPages: pg.totalPages,
            width: pg.width,
            height: pg.height,
            colorDepth: 24,
            createdAt,
            ocrText,
            dinoEmbeddingHex,
            tags: [
              "local-document",
              "pdf",
              `page-${pg.page}-of-${pg.totalPages}`,
              "pdfium-rendered",
              "dinov2-vit-s14",
              "verified",
            ],
          };

          const tokenBytes = await encodeBinaryToken({
            originalFileSize: rawBytes.length,
            width: pg.width,
            height: pg.height,
            page: pg.page,
            previewFormat: pg.previewFormat,
            shardBlockId: Math.floor((seq - 1) / 64),
            shardIndex: (seq - 1) % 64,
            embeddingModelId: pg.embeddingModelId,
            vqIndices: pg.vqIndices,
            metadata,
            previewBytes: pg.previewBytes,
          });

          const decoded = await decodeBinaryToken(tokenBytes);
          items.push({
            entryName,
            tokenBytes,
            meta: {
              id,
              entryName,
              filename,
              sourcePath: metadata.sourcePath,
              mimeType: metadata.mimeType,
              page: pg.page,
              totalPages: pg.totalPages,
              width: pg.width,
              height: pg.height,
              originalFileSize: rawBytes.length,
              previewFormat: pg.previewFormat,
              flags: decoded.flags,
              ocrText,
              createdAt,
              merkleRootHex: decoded.merkleRootHex,
              snarkVerified: decoded.snarkAttestation.valid,
              vqIndices: pg.vqIndices,
              dinoEmbedding: pg.dinoEmbedding,
              embeddingModelId: pg.embeddingModelId,
            },
          });
        }
        return items;
      } catch (err) {
        console.warn(`PDFium fallback for ${filename}:`, err);
      }
    }

    const docInfo = this.extractDocumentInfo(rawBytes, filename, ext);
    const id = `tok_${String(startTokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const entryName = `${id}.token`;

    const bundle = await rasterizeAssetToWebpAndTensors(
      rawBytes,
      ext,
      filename,
      1,
      docInfo.totalPages,
      {
        width: docInfo.width,
        height: docInfo.height,
        excerpt: docInfo.extractedText,
      },
    );
    const dinoEmbeddingHex = dinoEmbeddingToHex(bundle.dinoEmbedding);

    const snippetPart = docInfo.extractedText ? ` | Excerpt: ${docInfo.extractedText}` : "";
    const ocrText =
      `Document: ${filename} (${docInfo.totalPages} pages, ${(rawBytes.length / 1024).toFixed(1)} KB, ${docInfo.width}x${docInfo.height}px @ 150 DPI)` +
      `${snippetPart} | Source path: ${fullPath}`;

    const metadata: TokenMetadataPayload = {
      id,
      filename,
      sourcePath: `/api/document/${encodeURIComponent(filename)}`,
      mimeType: bunFile.type || (ext === ".pdf" ? "application/pdf" : "application/octet-stream"),
      page: 1,
      totalPages: docInfo.totalPages,
      width: docInfo.width,
      height: docInfo.height,
      colorDepth: 24,
      createdAt,
      ocrText,
      dinoEmbeddingHex,
      tags: ["local-document", ext.replace(".", ""), `${docInfo.totalPages}-pages`, "dinov2-vit-s14", "verified"],
    };

    const tokenBytes = await encodeBinaryToken({
      originalFileSize: rawBytes.length,
      width: docInfo.width,
      height: docInfo.height,
      page: 1,
      previewFormat: bundle.previewFormat,
      shardBlockId: Math.floor((startTokenCounter - 1) / 64),
      shardIndex: (startTokenCounter - 1) % 64,
      embeddingModelId: bundle.embeddingModelId,
      vqIndices: bundle.vqIndices,
      metadata,
      previewBytes: bundle.previewBytes,
    });

    const decoded = await decodeBinaryToken(tokenBytes);
    return [
      {
        entryName,
        tokenBytes,
        meta: {
          id,
          entryName,
          filename,
          sourcePath: metadata.sourcePath,
          mimeType: metadata.mimeType,
          page: 1,
          totalPages: docInfo.totalPages,
          width: docInfo.width,
          height: docInfo.height,
          originalFileSize: rawBytes.length,
          previewFormat: bundle.previewFormat,
          flags: decoded.flags,
          ocrText,
          createdAt,
          merkleRootHex: decoded.merkleRootHex,
          snarkVerified: decoded.snarkAttestation.valid,
          vqIndices: bundle.vqIndices,
          dinoEmbedding: bundle.dinoEmbedding,
          embeddingModelId: bundle.embeddingModelId,
        },
      },
    ];
  }

  /**
   * Tokenizes all real files in `sourceDirectory` (`./images`) and `documentDirectory` (`./documents`)
   * and writes the packed ZIP container to `outputZipPath`.
   */
  public async buildTokenArchive(overrideSourceDir?: string): Promise<{
    totalTokens: number;
    imageTokens: number;
    documentTokens: number;
    archiveBytes: number;
    elapsedMs: number;
  }> {
    if (overrideSourceDir) {
      this.config.sourceDirectory = overrideSourceDir;
    }
    const t0 = performance.now();
    const sourceDir = this.config.sourceDirectory;
    const imageFilter = new Set(
      this.config.allowedExtensions
        .map((e) => e.toLowerCase())
        .filter((e) => IMAGE_EXTENSIONS.has(e)),
    );
    const imagePaths = await this.scanDirectory(
      sourceDir,
      imageFilter.size > 0 ? imageFilter : IMAGE_EXTENSIONS,
    );
    const documentPaths = await this.scanDocumentDirectory();

    const archiveItems: ArchiveTokenItem[] = [];
    let tokenCounter = 0;
    let imageTokens = 0;
    let documentTokens = 0;

    for (const fullPath of imagePaths) {
      const item = await this.tokenizeImageFile(fullPath, ++tokenCounter);
      if (item) {
        archiveItems.push(item);
        imageTokens++;
      }
    }

    for (const fullPath of documentPaths) {
      const items = await this.tokenizeDocumentFile(fullPath, tokenCounter + 1);
      for (const item of items) {
        archiveItems.push(item);
        tokenCounter++;
        documentTokens++;
      }
    }

    const { totalBytes, tokenCount } = await writeNativeTokenZipArchive(
      this.config.outputZipPath,
      archiveItems,
    );

    return {
      totalTokens: tokenCount,
      imageTokens,
      documentTokens,
      archiveBytes: totalBytes,
      elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
    };
  }
}
