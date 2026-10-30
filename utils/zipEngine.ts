/**
 * 100% Native Bun ZIP Container Writer & Sub-Millisecond In-Memory Search Engine (utils/zipEngine.ts)
 *
 * Strictly uses Bun native I/O (Bun.file, Bun.write, DataView, Bun.deflateSync, Bun.inflateSync)
 * with ZERO Node.js fs/stream/yauzl dependencies.
 *
 * Features:
 * - Writes standard PKZIP archives containing 25 KB .token files + a trailing `_index.json` fast manifest
 *   so startup loads all 21,000 token metadata records, OCR layers, and 256B VQ grids in < 50 ms.
 * - On-demand single-token extraction via `Bun.file(zipPath).slice(dataOffset, dataOffset + compressedSize)`
 *   in 0.2 - 2.0 ms without unpacking the rest of the archive.
 * - Supports both OCR/filename substring filtering and Asymmetric Distance Computation (ADC) VQ similarity search.
 */

import {
  EmbeddingModelId,
  PreviewFormat,
  type SearchHitDTO,
  type ZipTokenIndexEntry,
} from "../models/token";
import { dinoEmbeddingToHex, hexToDinoEmbedding } from "./dinoEmbedding";
import { crc32 } from "./galois";
import { previewFormatToLabel } from "./imagePreview";
import { decodeBinaryToken } from "./tokenCodec";
import {
  adcDistanceToSimilarityPct,
  computeHybridVisualSimilarity,
  hexToVqGrid,
  vqGridToHex,
} from "./vqCodebook";

const ZIP_LOCAL_FILE_HEADER_SIG = 0x04034b50;
const ZIP_CENTRAL_DIR_HEADER_SIG = 0x02014b50;
const ZIP_EOCD_SIG = 0x06054b50;
const FAST_MANIFEST_ENTRY_NAME = "_index.json";

export interface RawZipEntryInput {
  entryName: string;
  data: Uint8Array;
  compress?: boolean; // false = STORE (0), true = DEFLATE (8)
}

interface SerializedManifestEntry {
  id: string;
  entryName: string;
  filename: string;
  sourcePath: string;
  mimeType: string;
  page: number;
  totalPages: number;
  width: number;
  height: number;
  originalFileSize: number;
  previewFormat: PreviewFormat;
  flags: number;
  ocrText: string;
  createdAt: string;
  localHeaderOffset: number;
  dataOffset: number;
  compressedSize: number;
  uncompressedSize: number;
  compressionMethod: number;
  crc32: number;
  merkleRootHex: string;
  snarkVerified: boolean;
  vqHex: string;
  dinoHex?: string;
  embeddingModelId?: EmbeddingModelId;
}

/**
 * Writes a standard random-access ZIP archive directly to disk using Bun.write().
 * Computes exact payload `dataOffset` values so each token can be sliced with zero header-parsing overhead.
 */
export async function writeNativeTokenZipArchive(
  zipFilePath: string,
  tokenEntries: Array<{
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
  }>,
): Promise<{ totalBytes: number; tokenCount: number }> {
  const localChunks: Uint8Array[] = [];
  const centralChunks: Uint8Array[] = [];
  const manifestRecords: SerializedManifestEntry[] = [];

  let currentOffset = 0;

  const appendEntry = (name: string, rawPayload: Uint8Array, useDeflate: boolean) => {
    const nameBytes = new TextEncoder().encode(name);
    const uncompressedSize = rawPayload.length;
    const entryCrc = crc32(rawPayload);

    let payloadToWrite: Uint8Array = rawPayload;
    let compressionMethod = 0; // 0 = STORE
    if (useDeflate && rawPayload.length > 128) {
      const deflated = Bun.deflateSync(new Uint8Array(rawPayload));
      if (deflated.length < rawPayload.length) {
        payloadToWrite = deflated;
        compressionMethod = 8; // 8 = DEFLATE
      }
    }
    const compressedSize = payloadToWrite.length;
    const localHeaderOffset = currentOffset;
    const dataOffset = localHeaderOffset + 30 + nameBytes.length;

    // 1. Local File Header (30 bytes + name)
    const localHeader = new Uint8Array(30 + nameBytes.length);
    const lView = new DataView(localHeader.buffer);
    lView.setUint32(0, ZIP_LOCAL_FILE_HEADER_SIG, true);
    lView.setUint16(4, 20, true); // Version needed (2.0)
    lView.setUint16(6, 0, true);  // Flags
    lView.setUint16(8, compressionMethod, true);
    lView.setUint16(10, 0, true); // Mod time
    lView.setUint16(12, 0, true); // Mod date
    lView.setUint32(14, entryCrc, true);
    lView.setUint32(18, compressedSize, true);
    lView.setUint32(22, uncompressedSize, true);
    lView.setUint16(26, nameBytes.length, true);
    lView.setUint16(28, 0, true); // Extra len
    localHeader.set(nameBytes, 30);

    localChunks.push(localHeader, payloadToWrite);
    currentOffset += localHeader.length + compressedSize;

    // 2. Central Directory Header (46 bytes + name)
    const centralHeader = new Uint8Array(46 + nameBytes.length);
    const cView = new DataView(centralHeader.buffer);
    cView.setUint32(0, ZIP_CENTRAL_DIR_HEADER_SIG, true);
    cView.setUint16(4, 20, true);  // Version made by
    cView.setUint16(6, 20, true);  // Version needed
    cView.setUint16(8, 0, true);   // Flags
    cView.setUint16(10, compressionMethod, true);
    cView.setUint16(12, 0, true);  // Mod time
    cView.setUint16(14, 0, true);  // Mod date
    cView.setUint32(16, entryCrc, true);
    cView.setUint32(20, compressedSize, true);
    cView.setUint32(24, uncompressedSize, true);
    cView.setUint16(28, nameBytes.length, true);
    cView.setUint16(30, 0, true);  // Extra len
    cView.setUint16(32, 0, true);  // Comment len
    cView.setUint16(34, 0, true);  // Disk number
    cView.setUint16(36, 0, true);  // Internal attrs
    cView.setUint32(38, 0, true);  // External attrs
    cView.setUint32(42, localHeaderOffset, true);
    centralHeader.set(nameBytes, 46);

    centralChunks.push(centralHeader);

    return {
      localHeaderOffset,
      dataOffset,
      compressedSize,
      uncompressedSize,
      compressionMethod,
      crc32: entryCrc,
    };
  };

  // Write all 25 KB token entries (STORE mode by default for zero-decompress random-access slicing)
  for (const item of tokenEntries) {
    const offsets = appendEntry(item.entryName, item.tokenBytes, false);
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
      localHeaderOffset: offsets.localHeaderOffset,
      dataOffset: offsets.dataOffset,
      compressedSize: offsets.compressedSize,
      uncompressedSize: offsets.uncompressedSize,
      compressionMethod: offsets.compressionMethod,
      crc32: offsets.crc32,
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

  // Append compressed `_index.json` fast manifest entry as the final entry before Central Directory
  const manifestBytes = new TextEncoder().encode(JSON.stringify(manifestRecords));
  appendEntry(FAST_MANIFEST_ENTRY_NAME, manifestBytes, true);

  const centralDirOffset = currentOffset;
  let centralDirSize = 0;
  for (const c of centralChunks) {
    centralDirSize += c.length;
  }

  // End of Central Directory Record (22 bytes)
  const totalEntriesCount = tokenEntries.length + 1;
  const eocd = new Uint8Array(22);
  const eView = new DataView(eocd.buffer);
  eView.setUint32(0, ZIP_EOCD_SIG, true);
  eView.setUint16(4, 0, true);
  eView.setUint16(6, 0, true);
  eView.setUint16(8, totalEntriesCount > 65535 ? 65535 : totalEntriesCount, true);
  eView.setUint16(10, totalEntriesCount > 65535 ? 65535 : totalEntriesCount, true);
  eView.setUint32(12, centralDirSize, true);
  eView.setUint32(16, centralDirOffset, true);
  eView.setUint16(20, 0, true);

  const totalSize = centralDirOffset + centralDirSize + 22;
  const finalArchive = new Uint8Array(totalSize);
  let pos = 0;
  for (const chunk of localChunks) {
    finalArchive.set(chunk, pos);
    pos += chunk.length;
  }
  for (const chunk of centralChunks) {
    finalArchive.set(chunk, pos);
    pos += chunk.length;
  }
  finalArchive.set(eocd, pos);

  await Bun.write(zipFilePath, finalArchive);
  return { totalBytes: totalSize, tokenCount: tokenEntries.length };
}

/**
 * In-Memory Compressed Token & VQ Search Engine
 */
export class CompressedTokenSearchEngine {
  private metadataIndex: Map<string, ZipTokenIndexEntry> = new Map();
  private metadataStore: ZipTokenIndexEntry[] = [];
  private zipFilePath: string;
  public lastBootIndexMs = 0;
  public archiveSizeBytes = 0;

  constructor(zipFilePath: string) {
    this.zipFilePath = zipFilePath;
  }

  public getZipFilePath(): string {
    return this.zipFilePath;
  }

  public getCount(): number {
    return this.metadataStore.length;
  }

  public getAllEntries(): ReadonlyArray<ZipTokenIndexEntry> {
    return this.metadataStore;
  }

  public getEntry(idOrEntryName: string): ZipTokenIndexEntry | undefined {
    return this.metadataIndex.get(idOrEntryName);
  }

  /**
   * Reads the ZIP Central Directory and `_index.json` manifest into RAM in < 50 ms using Bun.file().slice().
   */
  public async buildIndex(): Promise<void> {
    const startTime = performance.now();
    this.metadataIndex.clear();
    this.metadataStore = [];

    const file = Bun.file(this.zipFilePath);
    if (!(await file.exists())) {
      this.archiveSizeBytes = 0;
      this.lastBootIndexMs = 0;
      return;
    }

    this.archiveSizeBytes = file.size;
    if (this.archiveSizeBytes < 22) return;

    // 1. Slice the last 64 KB of the ZIP to find the End of Central Directory (EOCD) record
    const tailLen = Math.min(this.archiveSizeBytes, 65557);
    const tailOffset = this.archiveSizeBytes - tailLen;
    const tailBuf = new Uint8Array(await file.slice(tailOffset, this.archiveSizeBytes).arrayBuffer());
    const tailView = new DataView(tailBuf.buffer, tailBuf.byteOffset, tailBuf.byteLength);

    let eocdPos = -1;
    for (let i = tailBuf.length - 22; i >= 0; i--) {
      if (tailView.getUint32(i, true) === ZIP_EOCD_SIG) {
        eocdPos = i;
        break;
      }
    }
    if (eocdPos === -1) {
      throw new Error("Invalid ZIP container: End of Central Directory signature not found.");
    }

    const centralDirSize = tailView.getUint32(eocdPos + 12, true);
    const centralDirOffset = tailView.getUint32(eocdPos + 16, true);

    // 2. Slice the Central Directory to locate `_index.json` (or individual `.token` entries)
    const cdBuf = new Uint8Array(
      await file.slice(centralDirOffset, centralDirOffset + centralDirSize).arrayBuffer(),
    );
    const cdView = new DataView(cdBuf.buffer, cdBuf.byteOffset, cdBuf.byteLength);

    let ptr = 0;
    let manifestLocalOffset = -1;
    let manifestCompressedSize = 0;
    let manifestMethod = 0;

    const fallbackEntries: Array<{
      entryName: string;
      localHeaderOffset: number;
      compressedSize: number;
      uncompressedSize: number;
      compressionMethod: number;
      crc32: number;
    }> = [];

    while (ptr + 46 <= cdBuf.length) {
      const sig = cdView.getUint32(ptr, true);
      if (sig !== ZIP_CENTRAL_DIR_HEADER_SIG) break;

      const compressionMethod = cdView.getUint16(ptr + 10, true);
      const entryCrc = cdView.getUint32(ptr + 16, true);
      const compressedSize = cdView.getUint32(ptr + 20, true);
      const uncompressedSize = cdView.getUint32(ptr + 24, true);
      const nameLen = cdView.getUint16(ptr + 28, true);
      const extraLen = cdView.getUint16(ptr + 30, true);
      const commentLen = cdView.getUint16(ptr + 32, true);
      const localHeaderOffset = cdView.getUint32(ptr + 42, true);

      const entryName = new TextDecoder().decode(cdBuf.subarray(ptr + 46, ptr + 46 + nameLen));
      if (entryName === FAST_MANIFEST_ENTRY_NAME) {
        manifestLocalOffset = localHeaderOffset;
        manifestCompressedSize = compressedSize;
        manifestMethod = compressionMethod;
      } else if (entryName.endsWith(".token")) {
        fallbackEntries.push({
          entryName,
          localHeaderOffset,
          compressedSize,
          uncompressedSize,
          compressionMethod,
          crc32: entryCrc,
        });
      }

      ptr += 46 + nameLen + extraLen + commentLen;
    }

    // 3. Fast Path: Load `_index.json` manifest in a single slice
    if (manifestLocalOffset >= 0) {
      const dataStart = manifestLocalOffset + 30 + FAST_MANIFEST_ENTRY_NAME.length;
      const rawManifestSlice = new Uint8Array(
        await file.slice(dataStart, dataStart + manifestCompressedSize).arrayBuffer(),
      );
      const jsonBytes =
        manifestMethod === 8 ? Bun.inflateSync(rawManifestSlice) : rawManifestSlice;
      const parsed = JSON.parse(new TextDecoder().decode(jsonBytes)) as SerializedManifestEntry[];

      for (const item of parsed) {
        const entry: ZipTokenIndexEntry = {
          id: item.id,
          entryName: item.entryName,
          filename: item.filename,
          sourcePath: item.sourcePath,
          mimeType: item.mimeType,
          page: item.page,
          totalPages: item.totalPages,
          width: item.width,
          height: item.height,
          originalFileSize: item.originalFileSize,
          previewFormat: item.previewFormat,
          flags: item.flags,
          ocrText: item.ocrText,
          createdAt: item.createdAt,
          localHeaderOffset: item.localHeaderOffset,
          dataOffset: item.dataOffset,
          compressedSize: item.compressedSize,
          uncompressedSize: item.uncompressedSize,
          compressionMethod: item.compressionMethod,
          crc32: item.crc32,
          merkleRootHex: item.merkleRootHex,
          snarkVerified: item.snarkVerified,
          vqIndices: hexToVqGrid(item.vqHex),
          ...(item.dinoHex ? { dinoEmbedding: hexToDinoEmbedding(item.dinoHex) } : {}),
          ...(item.embeddingModelId !== undefined
            ? { embeddingModelId: item.embeddingModelId }
            : {}),
        };
        this.metadataIndex.set(entry.id, entry);
        this.metadataIndex.set(entry.entryName, entry);
        this.metadataStore.push(entry);
      }
    } else {
      // Fallback: Read each .token header from the ZIP
      for (const fb of fallbackEntries) {
        const nameLen = new TextEncoder().encode(fb.entryName).length;
        const dataOffset = fb.localHeaderOffset + 30 + nameLen;
        const rawSlice = new Uint8Array(
          await file.slice(dataOffset, dataOffset + fb.compressedSize).arrayBuffer(),
        );
        const tokenBytes = fb.compressionMethod === 8 ? Bun.inflateSync(rawSlice) : rawSlice;
        const decoded = decodeBinaryToken(tokenBytes);
        const entry: ZipTokenIndexEntry = {
          id: decoded.id,
          entryName: fb.entryName,
          filename: decoded.metadata.filename,
          sourcePath: decoded.metadata.sourcePath,
          mimeType: decoded.metadata.mimeType,
          page: decoded.page,
          totalPages: decoded.metadata.totalPages,
          width: decoded.width,
          height: decoded.height,
          originalFileSize: decoded.originalFileSize,
          previewFormat: decoded.previewFormat,
          flags: decoded.flags,
          ocrText: decoded.metadata.ocrText,
          createdAt: decoded.metadata.createdAt,
          localHeaderOffset: fb.localHeaderOffset,
          dataOffset,
          compressedSize: fb.compressedSize,
          uncompressedSize: fb.uncompressedSize,
          compressionMethod: fb.compressionMethod,
          crc32: fb.crc32,
          merkleRootHex: decoded.merkleRootHex,
          snarkVerified: decoded.snarkAttestation.valid,
          vqIndices: decoded.vqIndices,
          ...(decoded.metadata.dinoEmbeddingHex
            ? { dinoEmbedding: hexToDinoEmbedding(decoded.metadata.dinoEmbeddingHex) }
            : {}),
          embeddingModelId: decoded.embeddingModelId,
        };
        this.metadataIndex.set(entry.id, entry);
        this.metadataIndex.set(entry.entryName, entry);
        this.metadataStore.push(entry);
      }
    }

    this.lastBootIndexMs = Math.round((performance.now() - startTime) * 100) / 100;
  }

  /**
   * Sub-millisecond in-memory OCR & filename search.
   */
  public searchOCR(query: string, limit = 50): SearchHitDTO[] {
    const lowerQuery = query.trim().toLowerCase();
    const results: SearchHitDTO[] = [];

    for (const item of this.metadataStore) {
      if (
        !lowerQuery ||
        item.filename.toLowerCase().includes(lowerQuery) ||
        item.ocrText.toLowerCase().includes(lowerQuery) ||
        item.id.toLowerCase().includes(lowerQuery)
      ) {
        results.push(this.toSearchHitDTO(item));
        if (results.length >= limit) break;
      }
    }
    return results;
  }

  /**
   * Sub-millisecond Hybrid DINOv2 (384B ViT-14 Int8) + VQ (256B 16x16 ADC) visual similarity search.
   * Ranks all indexed tokens by semantic DINOv2 cosine + perceptual VQ distance to `targetId`.
   */
  public searchSimilarVQ(
    targetIdOrVqGrid: string | Uint8Array,
    textFilter = "",
    limit = 50,
  ): SearchHitDTO[] {
    let queryGrid: Uint8Array;
    let queryDino: Int8Array | undefined;
    let queryModelId: EmbeddingModelId | undefined;
    let targetId = "";

    if (typeof targetIdOrVqGrid === "string") {
      const targetEntry = this.metadataIndex.get(targetIdOrVqGrid);
      if (!targetEntry) return [];
      targetId = targetEntry.id;
      queryGrid = targetEntry.vqIndices;
      queryDino = targetEntry.dinoEmbedding;
      queryModelId = targetEntry.embeddingModelId;
    } else {
      queryGrid = targetIdOrVqGrid;
    }

    const lowerFilter = textFilter.trim().toLowerCase();
    const scored: Array<{
      entry: ZipTokenIndexEntry;
      similarityScore: number;
      adcDistance: number;
      dinoSimilarity: number;
    }> = [];

    for (const item of this.metadataStore) {
      if (
        lowerFilter &&
        !item.filename.toLowerCase().includes(lowerFilter) &&
        !item.ocrText.toLowerCase().includes(lowerFilter)
      ) {
        continue;
      }

      if (targetId && item.id === targetId) {
        scored.push({
          entry: item,
          similarityScore: 100,
          adcDistance: 0,
          dinoSimilarity: 100,
        });
        continue;
      }

      const sim = computeHybridVisualSimilarity(
        queryGrid,
        item.vqIndices,
        queryDino,
        item.dinoEmbedding,
        queryModelId,
        item.embeddingModelId,
      );
      scored.push({
        entry: item,
        similarityScore: sim.similarityScore,
        adcDistance: sim.adcDistance,
        dinoSimilarity: sim.dinoSimilarityPct,
      });
    }

    scored.sort((a, b) => b.similarityScore - a.similarityScore || a.adcDistance - b.adcDistance);
    return scored.slice(0, limit).map(({ entry, similarityScore, adcDistance, dinoSimilarity }) =>
      this.toSearchHitDTO(entry, adcDistance, similarityScore, dinoSimilarity),
    );
  }

  private toSearchHitDTO(
    entry: ZipTokenIndexEntry,
    adcDist?: number,
    similarityScore?: number,
    dinoSimilarity?: number,
  ): SearchHitDTO {
    return {
      id: entry.id,
      filename: entry.filename,
      sourcePath: entry.sourcePath,
      mimeType: entry.mimeType,
      page: entry.page,
      totalPages: entry.totalPages,
      width: entry.width,
      height: entry.height,
      originalFileSize: entry.originalFileSize,
      tokenSize: entry.uncompressedSize,
      previewFormat: previewFormatToLabel(entry.previewFormat),
      flags: entry.flags,
      ocrSnippet: entry.ocrText.slice(0, 220),
      createdAt: entry.createdAt,
      merkleRootHex: entry.merkleRootHex,
      snarkVerified: entry.snarkVerified,
      vqHex: vqGridToHex(entry.vqIndices),
      ...(adcDist !== undefined
        ? {
            adcDistance: adcDist,
            similarityScore: similarityScore ?? adcDistanceToSimilarityPct(adcDist),
            ...(dinoSimilarity !== undefined ? { dinoSimilarity } : {}),
          }
        : {}),
    };
  }

  /**
   * On-demand zero-decompress extraction of a single 25 KB token via Bun.file().slice().
   */
  public async extractToken(idOrEntryName: string): Promise<Uint8Array | null> {
    const meta = this.metadataIndex.get(idOrEntryName);
    if (!meta) return null;

    const file = Bun.file(this.zipFilePath);
    const slice = file.slice(meta.dataOffset, meta.dataOffset + meta.compressedSize);
    const rawBytes = new Uint8Array(await slice.arrayBuffer());

    if (meta.compressionMethod === 8) {
      return Bun.inflateSync(rawBytes);
    }
    return rawBytes;
  }

  /**
   * Updates an in-memory and on-disk token slice in-place (for Homomorphic GF(2^8) state transitions)
   * when the entry is stored uncompressed (STORE mode).
   */
  public async writeMutatedTokenInPlace(
    idOrEntryName: string,
    mutatedTokenBytes: Uint8Array,
  ): Promise<boolean> {
    const meta = this.metadataIndex.get(idOrEntryName);
    if (!meta || meta.compressionMethod !== 0 || mutatedTokenBytes.length !== meta.compressedSize) {
      return false;
    }

    const file = Bun.file(this.zipFilePath);
    const fullArchive = new Uint8Array(await file.arrayBuffer());
    fullArchive.set(mutatedTokenBytes, meta.dataOffset);
    await Bun.write(this.zipFilePath, fullArchive);

    const decoded = decodeBinaryToken(mutatedTokenBytes);
    meta.flags = decoded.flags;
    return true;
  }
}

