/**
 * Native Bun Directory Scanner & Custom 25 KB Tokenizer (utils/indexer.ts)
 *
 * Implements `Win11TokenPipeline` using 100% Bun native I/O (`Bun.Glob`, `Bun.file`, `Bun.write`, `Bun.inflateSync`)
 * + `sharp` WebP rasterization and DINOv2 ViT-14 384B semantic embeddings.
 * Scans both `sourceDirectory` (`./images`) and `documentDirectory` (`./documents`), generates 25 KB `GF28` V2 tokens
 * with 256B SNARK/SHARK provenance proofs, 256B VQ 16x16 neural grids, 384B DINOv2 embeddings,
 * 384B GF(2^8) RS parity, and lean WebP raster previews, packing them into `./data/tokens.zip`.
 */

import {
  type TokenMetadataPayload,
  type ZipTokenIndexEntry,
} from "../models/token";
import { dinoEmbeddingToHex } from "./dinoEmbedding";
import { rasterizeAssetToWebpAndTensors } from "./imagePreview";
import { decodeBinaryToken, encodeBinaryToken } from "./tokenCodec";
import { writeNativeTokenZipArchive } from "./zipEngine";

export interface IndexerConfig {
  sourceDirectory: string;
  documentDirectory?: string;
  outputZipPath: string;
  allowedExtensions: string[];
  maxTokenSizeKb: number;
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
   * Extracts PDF page count, MediaBox dimensions (scaled to 150 DPI), and readable text snippets
   * from PDF binary streams using native `Bun.inflateSync`.
   */
  public extractDocumentInfo(
    rawBytes: Uint8Array,
    filename: string,
    ext: string,
  ): ExtractedDocumentInfo {
    if (ext === ".docx") {
      // Native Bun ZIP local-header scan for `word/document.xml`
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
              .slice(0, 450);
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

    if (ext !== ".pdf") {
      const textPreview = new TextDecoder("utf-8", { fatal: false })
        .decode(rawBytes.subarray(0, 4096))
        .replace(/[^\x20-\x7E]+/g, " ")
        .replace(/\s+/g, " ")
        .trim()
        .slice(0, 600);
      return {
        totalPages: 1,
        width: 1240,
        height: 1754,
        extractedText: textPreview || `Document asset: ${filename}`,
      };
    }

    const headLen = Math.min(rawBytes.length, 512 * 1024);
    const headText = new TextDecoder("latin1").decode(rawBytes.subarray(0, headLen));
    const fullText =
      rawBytes.length <= 2 * 1024 * 1024
        ? new TextDecoder("latin1").decode(rawBytes)
        : headText +
          "\n" +
          new TextDecoder("latin1").decode(
            rawBytes.subarray(Math.max(headLen, rawBytes.length - 512 * 1024)),
          );

    const pageMatches = fullText.match(/\/Type\s*\/Page\b/g);
    let totalPages = pageMatches ? pageMatches.length : 1;
    const countMatch = fullText.match(/\/Type\s*\/Pages\b[\s\S]{0,200}?\/Count\s+(\d+)/);
    if (countMatch?.[1]) {
      const parsedCount = Number.parseInt(countMatch[1], 10);
      if (parsedCount > totalPages) totalPages = parsedCount;
    }

    let width = 1240;
    let height = 1754;
    const mediaBox = headText.match(
      /\/MediaBox\s*\[\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\]/,
    );
    if (mediaBox) {
      const wPt = Number.parseFloat(mediaBox[3] ?? "595.32") - Number.parseFloat(mediaBox[1] ?? "0");
      const hPt = Number.parseFloat(mediaBox[4] ?? "841.92") - Number.parseFloat(mediaBox[2] ?? "0");
      if (wPt > 0 && hPt > 0) {
        width = Math.round((wPt / 72) * 150);
        height = Math.round((hPt / 72) * 150);
      }
    }

    const snippets: string[] = [];
    const titleMatch = headText.match(/\/Title\s*\(([^()\\]{3,120})\)/);
    if (titleMatch?.[1]) snippets.push(titleMatch[1]);

    const streamRegex = /stream\r?\n/g;
    let match: RegExpExecArray | null = null;
    let streamsInspected = 0;

    while ((match = streamRegex.exec(headText)) !== null && streamsInspected < 25) {
      const start = match.index + match[0].length;
      const end = headText.indexOf("endstream", start);
      if (end === -1 || end - start > 128 * 1024) continue;

      const slice = rawBytes.subarray(start, end);
      if (slice.length > 6 && slice[0] === 0x78) {
        try {
          const inflated = Bun.inflateSync(new Uint8Array(slice.subarray(2)));
          const str = new TextDecoder("latin1").decode(inflated);
          const textLiterals = str.match(/\(([\x20-\x7E]{2,120})\)/g);
          if (textLiterals) {
            const cleaned = textLiterals
              .map((lit) => lit.slice(1, -1))
              .filter((lit) => /[a-zA-Z]{2,}/.test(lit) && !/^en-[A-Z]{2}$/.test(lit))
              .join(" ");
            if (cleaned.length > 0) {
              snippets.push(cleaned);
              streamsInspected++;
            }
          }
        } catch {
          // Non-deflate or encrypted stream; skip
        }
      }
      if (snippets.join(" ").length >= 450) break;
    }

    const cleanSnippet = snippets
      .join(" ")
      .replace(/\\[nrt()\\]/g, " ")
      .replace(/[^\x20-\x7E]+/g, " ")
      .replace(/\s+/g, " ")
      .trim()
      .slice(0, 450);

    return {
      totalPages: Math.max(1, totalPages),
      width,
      height,
      extractedText: cleanSnippet,
    };
  }

  /**
   * Tokenizes a single image or SVG file from `sourceDirectory` (`./images`), generating:
   * - Lean 220x220 WebP raster preview (Section III)
   * - True decoded 16x16 RGB VQ codebook grid (Section I, 256B)
   * - 384B Int8 DINOv2 ViT-14 semantic embedding (Section II)
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
      `Image Asset: ${filename} | Dimensions: ${bundle.width}x${bundle.height} (${bundle.colorDepth}-bit ${ext.replace(".", "").toUpperCase()} -> 220x220 WebP ${bundle.previewBytes.length}B)` +
      `${svgTextNote} | DINOv2-384B + 16x16 VQ + 256B SHARK/SNARK. Source path: ${fullPath}`;

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
      tags: ["local-image", ext.replace(".", ""), "dinov2-384b", "webp-preview", "verified"],
    };

    const tokenBytes = encodeBinaryToken({
      originalFileSize: rawBytes.length,
      width: bundle.width,
      height: bundle.height,
      page: 1,
      previewFormat: bundle.previewFormat,
      shardBlockId: Math.floor((tokenCounter - 1) / 64),
      shardIndex: (tokenCounter - 1) % 64,
      vqIndices: bundle.vqIndices,
      metadata,
      previewBytes: bundle.previewBytes,
    });

    const decoded = decodeBinaryToken(tokenBytes);
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
      },
    };
  }

  /**
   * Tokenizes a single document file from `documentDirectory` (`./documents`) with WebP cover preview,
   * 16x16 VQ grid, and 384B DINOv2 embedding.
   */
  public async tokenizeDocumentFile(
    fullPath: string,
    tokenCounter: number,
  ): Promise<ArchiveTokenItem | null> {
    const bunFile = Bun.file(fullPath);
    if (!(await bunFile.exists())) return null;

    const rawBytes = new Uint8Array(await bunFile.arrayBuffer());
    const filename = fullPath.split("/").pop() ?? "document.pdf";
    const dotIdx = filename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : ".pdf";

    const docInfo = this.extractDocumentInfo(rawBytes, filename, ext);
    const id = `tok_${String(tokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
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

    const createdAt = new Date(bunFile.lastModified || Date.now()).toISOString();
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
      tags: ["local-document", ext.replace(".", ""), `${docInfo.totalPages}-pages`, "dinov2-384b", "verified"],
    };

    const tokenBytes = encodeBinaryToken({
      originalFileSize: rawBytes.length,
      width: docInfo.width,
      height: docInfo.height,
      page: 1,
      previewFormat: bundle.previewFormat,
      shardBlockId: Math.floor((tokenCounter - 1) / 64),
      shardIndex: (tokenCounter - 1) % 64,
      vqIndices: bundle.vqIndices,
      metadata,
      previewBytes: bundle.previewBytes,
    });

    const decoded = decodeBinaryToken(tokenBytes);
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
      },
    };
  }

  /**
   * Tokenizes all real files in `sourceDirectory` (`./images`) and `documentDirectory` (`./documents`)
   * and writes the packed ZIP container to `outputZipPath`.
   */
  public async buildTokenArchive(): Promise<{
    totalTokens: number;
    imageTokens: number;
    documentTokens: number;
    archiveBytes: number;
    elapsedMs: number;
  }> {
    const t0 = performance.now();
    const imageFilter = new Set(
      this.config.allowedExtensions
        .map((e) => e.toLowerCase())
        .filter((e) => IMAGE_EXTENSIONS.has(e)),
    );
    const imagePaths = await this.scanDirectory(
      this.config.sourceDirectory,
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
      const item = await this.tokenizeDocumentFile(fullPath, ++tokenCounter);
      if (item) {
        archiveItems.push(item);
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
