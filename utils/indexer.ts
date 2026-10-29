/**
 * Native Bun Directory Scanner & Custom 25 KB Tokenizer (utils/indexer.ts)
 *
 * Implements `Win11TokenPipeline` using 100% Bun native I/O (`Bun.Glob`, `Bun.file`, `Bun.write`).
 * Scans `./images` (and any configured document folders), generates 25 KB `GF28` V2 tokens
 * with 256B SNARK/SHARK provenance proofs, 256B VQ 16x16 neural grids, and 384B GF(2^8) RS parity,
 * and packs them into `./data/tokens.zip`.
 */

import {
  PreviewFormat,
  type TokenMetadataPayload,
  type ZipTokenIndexEntry,
} from "../models/token";
import {
  buildCompactPreviewPayload,
  previewFormatToMime,
  sniffImageHeader,
} from "./imagePreview";
import { decodeBinaryToken, encodeBinaryToken } from "./tokenCodec";
import { encodeBytesToVqGrid } from "./vqCodebook";
import { writeNativeTokenZipArchive } from "./zipEngine";

export interface IndexerConfig {
  sourceDirectory: string;
  outputZipPath: string;
  allowedExtensions: string[];
  maxTokenSizeKb: number;
  includeSampleMultiPageDocs?: boolean;
}

export class Win11TokenPipeline {
  constructor(private config: IndexerConfig) {}

  /**
   * Scans sourceDirectory using native `Bun.Glob` (no node:fs imports).
   */
  public async scanDirectory(dir: string = this.config.sourceDirectory): Promise<string[]> {
    const matchedFiles: string[] = [];
    const extSet = new Set(this.config.allowedExtensions.map((e) => e.toLowerCase()));

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
   * Tokenizes all matched files (plus optional multi-page specification reference tokens)
   * and writes the packed ZIP container to `outputZipPath`.
   */
  public async buildTokenArchive(extraSyntheticCount = 12): Promise<{
    totalTokens: number;
    archiveBytes: number;
    elapsedMs: number;
  }> {
    const t0 = performance.now();
    const filePaths = await this.scanDirectory();

    const archiveItems: Array<{
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
    }> = [];

    let tokenCounter = 0;

    // 1. Tokenize real files in `./images` (e.g. d.jpg and any user-added images/docs)
    for (const fullPath of filePaths) {
      const bunFile = Bun.file(fullPath);
      if (!(await bunFile.exists())) continue;

      const rawBytes = new Uint8Array(await bunFile.arrayBuffer());
      const filename = fullPath.split("/").pop() ?? "asset.bin";
      const dotIdx = filename.lastIndexOf(".");
      const ext = dotIdx !== -1 ? filename.slice(dotIdx).toLowerCase() : "";
      const headerInfo = sniffImageHeader(rawBytes, ext);

      const id = `tok_${String(++tokenCounter).padStart(5, "0")}_${filename.replace(/[^a-zA-Z0-9._-]/g, "_")}`;
      const entryName = `${id}.token`;
      const vqIndices = encodeBytesToVqGrid(rawBytes, headerInfo.width, headerInfo.height, tokenCounter);

      const { previewBytes, previewFormat } = buildCompactPreviewPayload(
        rawBytes,
        headerInfo,
        vqIndices,
        filename,
        1,
      );

      const createdAt = new Date(bunFile.lastModified || Date.now()).toISOString();
      const ocrText =
        `Asset: ${filename} | Dimensions: ${headerInfo.width}x${headerInfo.height} (${headerInfo.colorDepth}-bit ${ext.replace(".", "").toUpperCase()}) | ` +
        `Indexed via TinyToken GF(2^8) Reed-Solomon + 16x16 VQ Neural Thumbnail + 256B SHARK/SNARK Attestation. Source path: ${fullPath}`;

      const metadata: TokenMetadataPayload = {
        id,
        filename,
        sourcePath: `/api/image/${encodeURIComponent(filename)}`,
        mimeType: bunFile.type || headerInfo.mimeType,
        page: 1,
        totalPages: 1,
        width: headerInfo.width,
        height: headerInfo.height,
        colorDepth: headerInfo.colorDepth,
        createdAt,
        ocrText,
        tags: ["local-image", ext.replace(".", ""), "verified"],
      };

      const tokenBytes = encodeBinaryToken({
        originalFileSize: rawBytes.length,
        width: headerInfo.width,
        height: headerInfo.height,
        page: 1,
        previewFormat,
        shardBlockId: Math.floor((tokenCounter - 1) / 64),
        shardIndex: (tokenCounter - 1) % 64,
        vqIndices,
        metadata,
        previewBytes,
      });

      const decoded = decodeBinaryToken(tokenBytes);
      archiveItems.push({
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
          width: headerInfo.width,
          height: headerInfo.height,
          originalFileSize: rawBytes.length,
          previewFormat,
          flags: decoded.flags,
          ocrText,
          createdAt,
          merkleRootHex: decoded.merkleRootHex,
          snarkVerified: decoded.snarkAttestation.valid,
          vqIndices,
        },
      });
    }

    // 2. Generate structured multi-page document & visual test tokens so search, VQ ADC similarity,
    //    and multi-page filtering work out-of-the-box alongside `./images/d.jpg`
    const sampleAssets = [
      {
        filename: "GF28_Token_Specification_V2.pdf",
        page: 1,
        totalPages: 8,
        width: 1654,
        height: 2339,
        size: 482910,
        ocr: "Compressed Token Index & Zip Search Engine Specification V2. Hybrid 25 KB token allocation: Section I 1KB Header & GF(2^8) Reed-Solomon Parity, Section II 4KB Metadata & OCR, Section III 20KB 220x220 AVIF Preview.",
        paletteSeed: [14, 165, 233],
      },
      {
        filename: "GF28_Token_Specification_V2.pdf",
        page: 2,
        totalPages: 8,
        width: 1654,
        height: 2339,
        size: 482910,
        ocr: "Section 3: Win11TokenPipeline & CompressedTokenSearchEngine. In-memory central directory indexing in under 500 ms for 21,000 structured assets (16,000 document page tokens + 5,000 standalone images).",
        paletteSeed: [16, 185, 129],
      },
      {
        filename: "VQ_SHARK_Neural_Attestation.pdf",
        page: 1,
        totalPages: 4,
        width: 1654,
        height: 2339,
        size: 319400,
        ocr: "ZK-SNARK & SHARK Asset Attestation stub (256 bytes in Section I) binding AVIF preview downscaling, OCR text layer, and 16x16 Vector Quantized (VQ-VAE) neural thumbnail codebook indices.",
        paletteSeed: [168, 85, 247],
      },
      {
        filename: "VQ_SHARK_Neural_Attestation.pdf",
        page: 2,
        totalPages: 4,
        width: 1654,
        height: 2339,
        size: 319400,
        ocr: "Asymmetric Distance Computation (ADC) in GF(2^8): Precalculated 256x256 centroid distance lookup table scores 21,000 VQ latent grids in microseconds without image decompression.",
        paletteSeed: [244, 63, 94],
      },
      {
        filename: "Risukko_Oksa_GF28_Matrix8x8.png",
        page: 1,
        totalPages: 1,
        width: 1920,
        height: 1080,
        size: 612400,
        ocr: "Risukko & Oksa 2D Bijective Finite Field Generator over primitive polynomial 0x11d (x^8 + x^4 + x^3 + x^2 + 1). Strategy #3 self-healing color checksums and 8x8 Gauss-Jordan matrix inversion.",
        paletteSeed: [247, 133, 34],
      },
      {
        filename: "GaloisOil_PBIN_Telemetry_Chart.webp",
        page: 1,
        totalPages: 1,
        width: 2560,
        height: 1440,
        size: 294120,
        ocr: "GaloisOil PBIN Packed Binary Time-Series Store: WTI Crude RWTC, Brent RBRTE, Henry Hub RNGWHHD. Berlekamp-Massey LFSR synthesis and Golden Ratio phi boundaries.",
        paletteSeed: [56, 189, 248],
      },
    ];

    const totalExtra = Math.max(0, extraSyntheticCount);
    for (let i = 0; i < totalExtra; i++) {
      const tmpl = sampleAssets[i % sampleAssets.length]!;
      const pageNum = i < sampleAssets.length ? tmpl.page : (i % 8) + 1;
      const docName =
        i < sampleAssets.length
          ? tmpl.filename
          : `Archive_Doc_${String(Math.floor(i / 8) + 1).padStart(4, "0")}.pdf`;

      const id = `tok_${String(++tokenCounter).padStart(5, "0")}_p${pageNum}`;
      const entryName = `${id}.token`;

      // Synthesize deterministic payload seed for VQ latent codebook
      const synthSeed = new Uint8Array(1024);
      for (let b = 0; b < synthSeed.length; b++) {
        synthSeed[b] = ((tmpl.paletteSeed[b % 3] ?? 128) + b * 13 + i * 29) & 0xff;
      }
      const vqIndices = encodeBytesToVqGrid(synthSeed, tmpl.width, tmpl.height, i + 1);

      const headerInfo = {
        width: tmpl.width,
        height: tmpl.height,
        colorDepth: 24,
        format: PreviewFormat.SVG,
        mimeType: "image/svg+xml",
      };

      const { previewBytes, previewFormat } = buildCompactPreviewPayload(
        new Uint8Array(0),
        headerInfo,
        vqIndices,
        docName,
        pageNum,
      );

      const createdAt = new Date(Date.now() - i * 3600_000).toISOString();
      const ocrText = `[Page ${pageNum}/${tmpl.totalPages}] ${docName} — ${tmpl.ocr}`;

      const metadata: TokenMetadataPayload = {
        id,
        filename: docName,
        sourcePath: `/api/token/${encodeURIComponent(id)}/preview`,
        mimeType: previewFormatToMime(previewFormat),
        page: pageNum,
        totalPages: tmpl.totalPages,
        width: tmpl.width,
        height: tmpl.height,
        colorDepth: 24,
        createdAt,
        ocrText,
        tags: ["spec-v2", "vq-16x16", "shark-attested"],
      };

      const tokenBytes = encodeBinaryToken({
        originalFileSize: tmpl.size,
        width: tmpl.width,
        height: tmpl.height,
        page: pageNum,
        previewFormat,
        shardBlockId: Math.floor((tokenCounter - 1) / 64),
        shardIndex: (tokenCounter - 1) % 64,
        vqIndices,
        metadata,
        previewBytes,
      });

      const decoded = decodeBinaryToken(tokenBytes);
      archiveItems.push({
        entryName,
        tokenBytes,
        meta: {
          id,
          entryName,
          filename: docName,
          sourcePath: metadata.sourcePath,
          mimeType: metadata.mimeType,
          page: pageNum,
          totalPages: tmpl.totalPages,
          width: tmpl.width,
          height: tmpl.height,
          originalFileSize: tmpl.size,
          previewFormat,
          flags: decoded.flags,
          ocrText,
          createdAt,
          merkleRootHex: decoded.merkleRootHex,
          snarkVerified: decoded.snarkAttestation.valid,
          vqIndices,
        },
      });
    }

    const { totalBytes, tokenCount } = await writeNativeTokenZipArchive(
      this.config.outputZipPath,
      archiveItems,
    );

    return {
      totalTokens: tokenCount,
      archiveBytes: totalBytes,
      elapsedMs: Math.round((performance.now() - t0) * 100) / 100,
    };
  }
}

