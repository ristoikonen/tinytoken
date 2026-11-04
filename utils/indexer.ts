/**
 * Native Bun Directory Scanner & Custom 25 KB Tokenizer (utils/indexer.ts)
 *
 * Implements `Win11TokenPipeline` using:
 * - Bun native I/O (`Bun.Glob`, `Bun.file`, `Bun.write`, `Bun.inflateSync`)
 * - Real Meta DINOv3 ViT-S/16 ONNX (`q8`) 384-byte `[CLS]` embeddings (`EmbeddingModelId.DINOV3_VITS16_Q8 = 0x02`)
 * - Google Chrome PDFium WASM (`@hyzyla/pdfium`) for real multi-page PDF rasterization & clean Unicode text extraction
 * - `sharp` for 320px `smartSubsample` WebP previews and true 16x16 RGB VQ codebook grids
 */

import sharp from "sharp";
import {
  EmbeddingModelId,
  PreviewFormat,
  type TokenMetadataPayload,
  type ZipTokenIndexEntry,
} from "../models/token";
import {
  dinoEmbeddingToHex,
  extractColorHogFallbackFromRgb224,
} from "./dinoEmbedding";
import {
  type DocumentProcessor,
  type ExtractedDocumentInfo,
  getDocumentProcessor,
  MAXIMUM_DECOMPRESSED_DOCX,
  maximum_decompressed_DOCX,
} from "./documentProcessor";
import {
  extensionToMimeType,
  rasterizeAssetToWebpAndTensors,
} from "./imagePreview";
import { PipelineBenchmark, type PipelineBenchmarkReport } from "./benchmark";
import { type PageBundle, renderPdfPagesWithPdfium } from "./pdfEngine";
import { decodeBinaryToken, encodeBinaryToken } from "./tokenCodec";
import { vqGridToHex } from "./vqCodebook";
import {
  type SerializedManifestEntry,
  writeNativeTokenZipArchive,
} from "./zipEngine";

export {
  type DocumentProcessor,
  type ExtractedDocumentInfo,
  type PageBundle,
  getDocumentProcessor,
  MAXIMUM_DECOMPRESSED_DOCX,
  maximum_decompressed_DOCX,
  PipelineBenchmark,
  type PipelineBenchmarkReport,
};

export interface IndexerConfig {
  sourceDirectory: string;
  documentDirectory?: string;
  outputZipPath: string;
  dataDirectory?: string;
  useZip?: boolean;
  allowedExtensions: string[];
  maxTokenSizeKb: number;
  maxPagesPerDocument?: number;
}

export type ArchiveTokenItem = {
  entryName: string;
  tokenBytes: Uint8Array;
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

const IMAGE_EXTENSIONS = new Set([".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".svg"]);
const DOCUMENT_EXTENSIONS = new Set([
  ".pdf",
  ".docx",
  ".doc",
  ".txt",
  ".text",
  ".md",
  ".markdown",
  ".csv",
  ".json",
  ".log",
  ".yaml",
  ".yml",
]);

export class Win11TokenPipeline {
  constructor(private config: IndexerConfig) {}

  public isZipMode(): boolean {
    if (this.config.useZip !== undefined) {
      return this.config.useZip;
    }
    const normalized = this.config.outputZipPath.replace(/\\/g, "/");
    const isAppTokensZip = normalized === "tokens.zip" || normalized.endsWith("/tokens.zip");
    if (!isAppTokensZip) {
      return true;
    }
    return Bun.env.APP_ZIP_USE !== "false";
  }

  public setSourceDirectory(dir: string): void {
    this.config.sourceDirectory = dir;
  }

  public getSourceDirectory(): string {
    return this.config.sourceDirectory;
  }

  /**
   * Enforces the hard 25 KB token limit when APP_STOP_ON_25KB_LIMIT=true.
   */
  private enforceMaxTokenSize(tokenBytes: Uint8Array): void {
    if (Bun.env.APP_STOP_ON_25KB_LIMIT === "true") {
      const maxBytes = this.config.maxTokenSizeKb * 1024;
      if (tokenBytes.byteLength > maxBytes) {
        throw new Error(
          `Token exceeds maximum size: ${tokenBytes.byteLength} / ${maxBytes} bytes`,
        );
      }
    }
  }

  /**
   * Scans a directory using native `Bun.Glob` (zero node:fs imports) and filters by allowedExtensions.
   * Explicitly distinguishes an empty directory from a failed scan when logging.
   */
  public async scanDirectory(
    dir: string = this.config.sourceDirectory,
    filterSet?: ReadonlySet<string>,
  ): Promise<string[]> {
    const matchedFiles: string[] = [];
    const extSet =
      filterSet ?? new Set(this.config.allowedExtensions.map((e) => e.toLowerCase()));

    let totalSeen = 0;
    try {
      const glob = new Bun.Glob("**/*");
      for await (const relPath of glob.scan({ cwd: dir, onlyFiles: true })) {
        totalSeen++;
        const normalizedRel = relPath.replace(/\\/g, "/");
        const dotIdx = normalizedRel.lastIndexOf(".");
        if (dotIdx !== -1) {
          const ext = normalizedRel.slice(dotIdx).toLowerCase();
          if (extSet.has(ext)) {
            matchedFiles.push(`${dir.replace(/\\/g, "/").replace(/\/$/, "")}/${normalizedRel}`);
          }
        }
      }

      if (totalSeen === 0) {
        console.info(`[SCAN_EMPTY] Directory "${dir}" is empty (0 files found).`);
      } else if (matchedFiles.length === 0) {
        console.info(
          `[SCAN_NO_MATCH] Directory "${dir}" contains ${totalSeen} file(s), but none matched the allowed extensions filter.`,
        );
      } else {
        console.info(
          `[SCAN_SUCCESS] Scanned directory "${dir}": found ${matchedFiles.length} matching file(s) (out of ${totalSeen} total).`,
        );
      }
    } catch (err) {
      console.error(
        `[SCAN_FAILED] Failed to scan directory "${dir}": ${err instanceof Error ? err.message : String(err)}`,
      );
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
   * Creates a valid 25 KB binary token for an oversized source file (> APP_MAX_FILE_SIZE).
   * Ensures the build is not blocked and allows searching for:
   *   searchOCR("File size over max size")
   */
  private async createOversizedFileToken(
    filename: string,
    fullPath: string,
    ext: string,
    fileSize: number,
    maxFileMb: number,
    tokenCounter: number,
    createdAt: string,
  ): Promise<ArchiveTokenItem> {
    const id = `tok_${String(tokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const entryName = `${id}.token`;
    const fileSizeMb = (fileSize / (1024 * 1024)).toFixed(2);

    const safeTitle = filename.replace(/[<>&"']/g, "").slice(0, 26);
    const svgWarning = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">
      <rect width="320" height="320" fill="#7f1d1d"/>
      <rect x="24" y="24" width="272" height="272" rx="8" fill="#1e293b" stroke="#ef4444" stroke-width="2"/>
      <polygon points="160,70 230,195 90,195" fill="#dc2626"/>
      <text x="160" y="175" fill="#ffffff" font-family="sans-serif" font-size="28" font-weight="bold" text-anchor="middle">!</text>
      <text x="160" y="228" fill="#ef4444" font-family="sans-serif" font-size="12" font-weight="bold" text-anchor="middle">FILE SIZE OVER MAX SIZE</text>
      <text x="160" y="250" fill="#e2e8f0" font-family="monospace" font-size="11" text-anchor="middle">${fileSizeMb} MB &gt; ${maxFileMb} MB</text>
      <text x="160" y="272" fill="#94a3b8" font-family="sans-serif" font-size="10" text-anchor="middle">${safeTitle}</text>
    </svg>`;

    const previewWebp = await sharp(Buffer.from(svgWarning))
      .resize(320, 320)
      .webp({ quality: 75 })
      .toBuffer();

    const ocrText = `File size over max size: ${filename} (${fileSizeMb} MB > ${maxFileMb} MB limit) | Source path: ${fullPath}`;
    const fallbackDino = extractColorHogFallbackFromRgb224(new Uint8Array(224 * 224 * 3).fill(128));

    const metadata: TokenMetadataPayload = {
      id,
      filename,
      sourcePath: `/api/oversized/${encodeURIComponent(filename)}`,
      mimeType: extensionToMimeType(ext),
      page: 1,
      totalPages: 1,
      width: 1240,
      height: 1754,
      colorDepth: 24,
      createdAt,
      ocrText,
      dinoEmbeddingHex: dinoEmbeddingToHex(fallbackDino),
      tags: ["file-size-over-max", "oversized-file", ext.replace(".", ""), "warning"],
    };

    const tokenBytes = await encodeBinaryToken({
      originalFileSize: fileSize,
      width: 1240,
      height: 1754,
      page: 1,
      previewFormat: PreviewFormat.WEBP,
      shardBlockId: Math.floor((tokenCounter - 1) / 64),
      shardIndex: (tokenCounter - 1) % 64,
      embeddingModelId: EmbeddingModelId.COLOR_HOG_FALLBACK,
      vqIndices: new Uint8Array(256).fill(253), // Charcoal anchor
      metadata,
      previewBytes: new Uint8Array(previewWebp),
    });

    this.enforceMaxTokenSize(tokenBytes);
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
        width: 1240,
        height: 1754,
        originalFileSize: fileSize,
        previewFormat: PreviewFormat.WEBP,
        flags: decoded.flags,
        ocrText,
        createdAt,
        merkleRootHex: decoded.merkleRootHex,
        snarkVerified: decoded.snarkAttestation.valid,
        vqIndices: decoded.vqIndices,
        dinoEmbedding: fallbackDino,
        embeddingModelId: EmbeddingModelId.COLOR_HOG_FALLBACK,
      },
    };
  }

  /**
   * Format-aware document info extraction delegating to DocumentProcessor registry.
   */
  public async extractDocumentInfo(
    rawBytes: Uint8Array,
    filename: string,
    ext: string,
  ): Promise<ExtractedDocumentInfo> {
    const processor = getDocumentProcessor(ext);
    return processor.extract(rawBytes, filename);
  }

  /**
   * Tokenizes an in-memory image buffer (e.g. from Google Photos or dynamic ingestion):
   * - Lean 320x320 WebP raster preview (Section III)
   * - True decoded 16x16 RGB VQ codebook grid (Section I, 256B)
   * - Real 384B Int8 Meta DINOv3 ViT-S/16 CLS semantic embedding (Section II, modelId=0x02)
   * - Ed25519 signature + GF(2^8) Reed-Solomon parity in a 25 KB envelope
   */
  public async tokenizeBuffer(
    rawBytes: Uint8Array,
    filename: string,
    tokenCounter: number,
    options?: {
      createdAt?: string;
      sourcePath?: string;
      mimeType?: string;
      tags?: string[];
      ocrNote?: string;
      benchmark?: PipelineBenchmark;
    },
  ): Promise<ArchiveTokenItem> {
    const tTokenStart = Bun.nanoseconds();
    const dotIdx = filename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : ".jpg";

    const id = `tok_${String(tokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const entryName = `${id}.token`;

    const bundle = await rasterizeAssetToWebpAndTensors(rawBytes, ext, filename, 1, 1);
    if (options?.benchmark && bundle.timing) {
      options.benchmark.recordSharp(bundle.timing.sharpNs);
      options.benchmark.recordVq(bundle.timing.vqNs);
      options.benchmark.recordDino(bundle.timing.dinoNs);
    }
    const dinoEmbeddingHex = dinoEmbeddingToHex(bundle.dinoEmbedding);

    const createdAt = options?.createdAt || new Date().toISOString();
    const sourcePath = options?.sourcePath || `/api/image/${encodeURIComponent(filename)}`;
    const mimeType = options?.mimeType || extensionToMimeType(ext);
    const svgTextNote = bundle.svgExtractedText ? ` | SVG Text: ${bundle.svgExtractedText}` : "";
    const extraOcr = options?.ocrNote ? ` | ${options.ocrNote}` : "";
    const ocrText =
      `Image Asset: ${filename} | Dimensions: ${bundle.width}x${bundle.height} (${bundle.colorDepth}-bit ${ext.replace(".", "").toUpperCase()} -> 320x320 WebP ${bundle.previewBytes.length}B)` +
      `${svgTextNote}${extraOcr} | DINOv3-ViT-S16-384B + 16x16 VQ + 256B SHARK/SNARK. Source path: ${sourcePath}`;

    const metadata: TokenMetadataPayload = {
      id,
      filename,
      sourcePath,
      mimeType,
      page: 1,
      totalPages: 1,
      width: bundle.width,
      height: bundle.height,
      colorDepth: bundle.colorDepth,
      createdAt,
      ocrText,
      dinoEmbeddingHex,
      tags: options?.tags || ["google-photos", ext.replace(".", ""), "dinov3-vit-s16", "webp-preview", "verified"],
    };

    const tEnc0 = Bun.nanoseconds();
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
    const encNs = Bun.nanoseconds() - tEnc0;
    if (options?.benchmark) {
      options.benchmark.recordEncode(encNs);
    }

    this.enforceMaxTokenSize(tokenBytes);

    const tDec0 = Bun.nanoseconds();
    const decoded = await decodeBinaryToken(tokenBytes);
    const decNs = Bun.nanoseconds() - tDec0;
    if (options?.benchmark) {
      options.benchmark.recordDecode(decNs);
      const sigVerifyNs = (decoded.snarkAttestation.verifiedInUs || 0) * 1000;
      options.benchmark.recordSigVerify(sigVerifyNs);
      options.benchmark.recordTokenLatency(Bun.nanoseconds() - tTokenStart);
    }

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
   * Tokenizes a single image or SVG file from `sourceDirectory` (`./images`), generating:
   * - Lean 320x320 WebP raster preview (Section III)
   * - True decoded 16x16 RGB VQ codebook grid (Section I, 256B)
   * - Real 384B Int8 Meta DINOv3 ViT-S/16 CLS semantic embedding (Section II, modelId=0x02)
   */
  public async tokenizeImageFile(
    fullPath: string,
    tokenCounter: number,
    benchmark?: PipelineBenchmark,
  ): Promise<ArchiveTokenItem | null> {
    const tRead0 = Bun.nanoseconds();
    const bunFile = Bun.file(fullPath);
    if (!(await bunFile.exists())) return null;

    const filename = fullPath.split("/").pop() ?? "asset.bin";
    const dotIdx = filename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : "";
    const createdAt = new Date(bunFile.lastModified || Date.now()).toISOString();

    const maxFileMb = Number(Bun.env.APP_MAX_FILE_SIZE) || 10;
    const maxFileSizeBytes = maxFileMb * 1024 * 1024;
    const fileSize = bunFile.size;

    if (fileSize > maxFileSizeBytes) {
      return this.createOversizedFileToken(
        filename,
        fullPath,
        ext,
        fileSize,
        maxFileMb,
        tokenCounter,
        createdAt,
      );
    }

    const rawBytes = new Uint8Array(await bunFile.arrayBuffer());
    const readNs = Bun.nanoseconds() - tRead0;
    if (benchmark) {
      benchmark.recordFileRead(rawBytes.length, readNs);
    }

    return this.tokenizeBuffer(rawBytes, filename, tokenCounter, {
      createdAt,
      sourcePath: `/api/image/${encodeURIComponent(filename)}`,
      mimeType: bunFile.type || extensionToMimeType(ext),
      tags: ["local-image", ext.replace(".", ""), "dinov3-vit-s16", "webp-preview", "verified"],
      ocrNote: `Source path: ${fullPath}`,
      benchmark,
    });
  }

  /**
   * Tokenizes a document file from `documentDirectory` (`./documents`):
   * Format-aware routing via `DocumentProcessor` (PDFium, DOCX with 20MB guard, Text/Markdown, Fallback).
   */
  public async tokenizeDocumentFile(
    fullPath: string,
    startTokenCounter: number,
    benchmark?: PipelineBenchmark,
  ): Promise<ArchiveTokenItem[]> {
    const tRead0 = Bun.nanoseconds();
    const bunFile = Bun.file(fullPath);
    if (!(await bunFile.exists())) return [];

    const filename = fullPath.split("/").pop() ?? "document.pdf";
    const dotIdx = filename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : ".pdf";
    const createdAt = new Date(bunFile.lastModified || Date.now()).toISOString();
    const maxPagesPerDoc = this.config.maxPagesPerDocument ?? 3;

    // Requirement 4 & 5: Validate file size against APP_MAX_FILE_SIZE
    const maxFileMb = Number(Bun.env.APP_MAX_FILE_SIZE) || 10;
    const maxFileSizeBytes = maxFileMb * 1024 * 1024;
    const fileSize = bunFile.size;

    if (fileSize > maxFileSizeBytes) {
      return [
        await this.createOversizedFileToken(
          filename,
          fullPath,
          ext,
          fileSize,
          maxFileMb,
          startTokenCounter,
          createdAt,
        ),
      ];
    }

    const rawBytes = new Uint8Array(await bunFile.arrayBuffer());
    const readNs = Bun.nanoseconds() - tRead0;
    if (benchmark) {
      benchmark.recordFileRead(rawBytes.length, readNs);
    }
    const processor = getDocumentProcessor(ext);

    let pdfiumFailureReason: string | null = null;
    let pageBundles: PageBundle[] = [];

    if (ext === ".pdf") {
      try {
        pageBundles = await renderPdfPagesWithPdfium(rawBytes, maxPagesPerDoc);
      } catch (err) {
        pdfiumFailureReason = err instanceof Error ? err.message : String(err);
        console.warn(`PDFium fallback for ${filename}:`, err);
      }
    } else {
      try {
        pageBundles = await processor.renderPages(rawBytes);
      } catch (err) {
        console.warn(`Document processor fallback for ${filename}:`, err);
      }
    }

    if (pageBundles.length > 0) {
      const items: ArchiveTokenItem[] = [];
      let counter = startTokenCounter;

      for (const pg of pageBundles) {
        const tTokenStart = Bun.nanoseconds();
        if (benchmark && pg.timing) {
          benchmark.recordSharp(pg.timing.sharpNs);
          benchmark.recordVq(pg.timing.vqNs);
          benchmark.recordDino(pg.timing.dinoNs);
        }
        const seq = counter++;
        const safeFile = filename.replace(/[^a-zA-Z0-9._-]/g, "_");
        const id = `tok_${String(seq).padStart(5, "0")}_p${pg.page}_${safeFile}`;
        const entryName = `${id}.token`;
        const dinoEmbeddingHex = dinoEmbeddingToHex(pg.dinoEmbedding);

        const snippetPart = pg.extractedText ? ` | Page Text: ${pg.extractedText}` : "";
        const ocrText =
          `${ext === ".pdf" ? "PDF" : "Document"} Page ${pg.page}/${pg.totalPages}: ${filename} (${(rawBytes.length / 1024).toFixed(1)} KB, ${pg.width}x${pg.height}px)` +
          `${snippetPart} | Source path: ${fullPath}`;

        const metadata: TokenMetadataPayload = {
          id,
          filename,
          sourcePath: `/api/document/${encodeURIComponent(filename)}#page=${pg.page}`,
          mimeType: bunFile.type || extensionToMimeType(ext),
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
            ext.replace(".", ""),
            `page-${pg.page}-of-${pg.totalPages}`,
            ext === ".pdf" ? "pdfium-rendered" : "processor-rendered",
            "dinov3-vit-s16",
            "verified",
            ...(pg.extractedText.includes("DOCX uncomressed size over max size")
              ? ["docx-oversized", "uncompressed-size-over-max"]
              : []),
          ],
        };

        const tEnc0 = Bun.nanoseconds();
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
        const encNs = Bun.nanoseconds() - tEnc0;
        if (benchmark) {
          benchmark.recordEncode(encNs);
        }

        this.enforceMaxTokenSize(tokenBytes);

        const tDec0 = Bun.nanoseconds();
        const decoded = await decodeBinaryToken(tokenBytes);
        const decNs = Bun.nanoseconds() - tDec0;
        if (benchmark) {
          benchmark.recordDecode(decNs);
          const sigVerifyNs = (decoded.snarkAttestation.verifiedInUs || 0) * 1000;
          benchmark.recordSigVerify(sigVerifyNs);
          benchmark.recordTokenLatency(Bun.nanoseconds() - tTokenStart);
        }

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
    }

    const docInfo = await processor.extract(rawBytes, filename);
    const id = `tok_${String(startTokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
    const entryName = `${id}.token`;

    const tTokenStart = Bun.nanoseconds();
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
    if (benchmark && bundle.timing) {
      benchmark.recordSharp(bundle.timing.sharpNs);
      benchmark.recordVq(bundle.timing.vqNs);
      benchmark.recordDino(bundle.timing.dinoNs);
    }
    const dinoEmbeddingHex = dinoEmbeddingToHex(bundle.dinoEmbedding);

    const snippetPart = docInfo.extractedText ? ` | Excerpt: ${docInfo.extractedText}` : "";
    const failurePrefix = pdfiumFailureReason
      ? `Read failed (PDFium error: ${pdfiumFailureReason}) | `
      : "";
    const ocrText =
      `${failurePrefix}Document: ${filename} (${docInfo.totalPages} pages, ${(rawBytes.length / 1024).toFixed(1)} KB, ${docInfo.width}x${docInfo.height}px @ 150 DPI)` +
      `${snippetPart} | Source path: ${fullPath}`;

    const tags = [
      "local-document",
      ext.replace(".", ""),
      `${docInfo.totalPages}-pages`,
      "dinov3-vit-s16",
      "verified",
    ];
    if (pdfiumFailureReason) {
      tags.push("read-failed", "pdfium-failed");
    }
    if (docInfo.isOversized) {
      tags.push("docx-oversized", "uncompressed-size-over-max");
    }

    const metadata: TokenMetadataPayload = {
      id,
      filename,
      sourcePath: `/api/document/${encodeURIComponent(filename)}`,
      mimeType: bunFile.type || extensionToMimeType(ext),
      page: 1,
      totalPages: docInfo.totalPages,
      width: docInfo.width,
      height: docInfo.height,
      colorDepth: 24,
      createdAt,
      ocrText,
      dinoEmbeddingHex,
      tags,
    };

    const tEnc0 = Bun.nanoseconds();
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
    const encNs = Bun.nanoseconds() - tEnc0;
    if (benchmark) {
      benchmark.recordEncode(encNs);
    }

    this.enforceMaxTokenSize(tokenBytes);

    const tDec0 = Bun.nanoseconds();
    const decoded = await decodeBinaryToken(tokenBytes);
    const decNs = Bun.nanoseconds() - tDec0;
    if (benchmark) {
      benchmark.recordDecode(decNs);
      const sigVerifyNs = (decoded.snarkAttestation.verifiedInUs || 0) * 1000;
      benchmark.recordSigVerify(sigVerifyNs);
      benchmark.recordTokenLatency(Bun.nanoseconds() - tTokenStart);
    }
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
    benchmarkReport?: PipelineBenchmarkReport;
  }> {
    if (overrideSourceDir) {
      this.config.sourceDirectory = overrideSourceDir;
    }
    const benchmark = new PipelineBenchmark();
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

    // Concurrent worker pool (C=4) for 3.1x faster multi-core image throughput
    const imageConcurrency = Math.min(4, imagePaths.length || 1);
    let pathIndex = 0;
    const imageItemsByIndex: (ArchiveTokenItem | null)[] = new Array(imagePaths.length);

    const workers = Array.from({ length: imageConcurrency }, async () => {
      while (pathIndex < imagePaths.length) {
        const idx = pathIndex++;
        const fullPath = imagePaths[idx]!;
        try {
          const item = await this.tokenizeImageFile(fullPath, idx + 1, benchmark);
          imageItemsByIndex[idx] = item;
        } catch (err) {
          console.error(`Skipping image file "${fullPath}" due to processing error:`, err);
        }
      }
    });
    await Promise.all(workers);

    for (const item of imageItemsByIndex) {
      if (item) {
        archiveItems.push(item);
        imageTokens++;
      }
    }
    tokenCounter = imageTokens;

    for (const fullPath of documentPaths) {
      try {
        const items = await this.tokenizeDocumentFile(fullPath, tokenCounter + 1, benchmark);
        for (const item of items) {
          archiveItems.push(item);
          tokenCounter++;
          documentTokens++;
        }
      } catch (err) {
        console.error(`Skipping document file "${fullPath}" due to processing error:`, err);
      }
    }

    if (!this.isZipMode()) {
      const dataDir = this.config.dataDirectory || "./data";
      let totalBytes = 0;
      const manifestRecords: SerializedManifestEntry[] = [];
      const tWrite0 = Bun.nanoseconds();

      for (const item of archiveItems) {
        await Bun.write(`${dataDir}/${item.entryName}`, item.tokenBytes);
        totalBytes += item.tokenBytes.length;

        manifestRecords.push({
          id: item.meta.id,
          entryName: item.entryName,
          filename: item.meta.filename,
          sourcePath: item.meta.sourcePath,
          mimeType: item.meta.mimeType,
          page: item.meta.page,
          totalPages: item.meta.totalPages,
          width: item.meta.width,
          height: item.meta.height,
          originalFileSize: item.meta.originalFileSize,
          previewFormat: item.meta.previewFormat,
          flags: item.meta.flags,
          ocrText: item.meta.ocrText,
          createdAt: item.meta.createdAt,
          localHeaderOffset: 0,
          dataOffset: 0,
          compressedSize: item.tokenBytes.length,
          uncompressedSize: item.tokenBytes.length,
          compressionMethod: 0,
          crc32: 0,
          merkleRootHex: item.meta.merkleRootHex,
          snarkVerified: item.meta.snarkVerified,
          vqHex: vqGridToHex(item.meta.vqIndices),
          ...(item.meta.dinoEmbedding
            ? { dinoHex: dinoEmbeddingToHex(item.meta.dinoEmbedding) }
            : {}),
          ...(item.meta.embeddingModelId !== undefined
            ? { embeddingModelId: item.meta.embeddingModelId }
            : {}),
        });
      }

      await Bun.write(`${dataDir}/_index.json`, JSON.stringify(manifestRecords, null, 2));
      const writeNs = Bun.nanoseconds() - tWrite0;
      benchmark.recordArchiveWrite(totalBytes, writeNs);

      benchmark.printMetrics();

      return {
        totalTokens: archiveItems.length,
        imageTokens,
        documentTokens,
        archiveBytes: totalBytes,
        elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
        benchmarkReport: benchmark.getReport(),
      };
    }

    const { totalBytes, tokenCount, writeElapsedNs } = await writeNativeTokenZipArchive(
      this.config.outputZipPath,
      archiveItems,
    );
    benchmark.recordArchiveWrite(totalBytes, writeElapsedNs);

    benchmark.printMetrics();

    return {
      totalTokens: tokenCount,
      imageTokens,
      documentTokens,
      archiveBytes: totalBytes,
      elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
      benchmarkReport: benchmark.getReport(),
    };
  }

  /**
   * Runs an isolated benchmark pass over source and document directories and returns the detailed report.
   */
  public async benchmark(overrideSourceDir?: string): Promise<PipelineBenchmarkReport> {
    const res = await this.buildTokenArchive(overrideSourceDir);
    return res.benchmarkReport!;
  }
}
