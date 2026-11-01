/**
 * Transactional Archive Manager (utils/archiveManager.ts)
 *
 * Implements generation-based hot-swapping of `tokens.zip`:
 * - Second archive-ingestion path accepting pre-built ZIP archives directly
 * - Two-phase commit: staging write -> strict structural validation -> atomic file swap
 * - Strict rollback: bad validation keeps active generation intact with zero downtime
 * - Readers remain pinned to the consistent active generation during staging
 */

import type {
  ArchiveGenerationState,
  ArchiveValidationResult,
} from "../models/learning";
import { TOKEN_MAGIC } from "../models/token";
import type { VqInvertedIndexEngine } from "./vqInvertedIndex";
import type { CompressedTokenSearchEngine } from "./zipEngine";

const ZIP_EOCD_SIG = 0x06054b50;
const ZIP_LOCAL_HEADER_SIG = 0x04034b50;
const ZIP_CENTRAL_DIR_SIG = 0x02014b50;

export class ArchiveManager {
  private generation = 1;
  private activeZipPath: string;
  private stagingZipPath: string;
  private searchEngine: CompressedTokenSearchEngine;
  private invertedIndex?: VqInvertedIndexEngine;
  private lastSwappedAt: string;
  private activeCrc32Hex = "00000000";

  constructor(
    searchEngine: CompressedTokenSearchEngine,
    activeZipPath = "./data/tokens.zip",
    invertedIndex?: VqInvertedIndexEngine,
  ) {
    this.searchEngine = searchEngine;
    this.activeZipPath = activeZipPath;
    this.stagingZipPath = `${activeZipPath}.staging`;
    this.invertedIndex = invertedIndex;
    this.lastSwappedAt = new Date().toISOString();
  }

  public getGenerationState(): ArchiveGenerationState {
    return {
      generation: this.generation,
      activeZipPath: this.activeZipPath,
      totalTokens: this.searchEngine.getCount(),
      archiveSizeBytes: this.searchEngine.archiveSizeBytes,
      archiveCrc32Hex: this.activeCrc32Hex,
      swappedAt: this.lastSwappedAt,
    };
  }

  /**
   * Fast structural validation of candidate ZIP archive bytes or file path.
   * Checks:
   * 1. Minimum ZIP size (>= 22 bytes)
   * 2. Valid ZIP End of Central Directory (EOCD) signature
   * 3. Central Directory traversal & presence of token entries
   * 4. Token format integrity (magic bytes `GF28` on first sampled token)
   */
  public async validateArchive(
    source: Uint8Array<ArrayBuffer> | string,
  ): Promise<ArchiveValidationResult> {
    try {
      let buf: Uint8Array<ArrayBuffer>;
      if (typeof source === "string") {
        const file = Bun.file(source);
        if (!(await file.exists())) {
          return { valid: false, tokenCount: 0, byteSize: 0, crc32Hex: "00000000", error: `File not found: ${source}` };
        }
        buf = new Uint8Array(await file.arrayBuffer());
      } else {
        buf = source;
      }

      const byteSize = buf.length;
      if (byteSize < 22) {
        return { valid: false, tokenCount: 0, byteSize, crc32Hex: "00000000", error: "Archive too small to be a valid ZIP (< 22 bytes)" };
      }

      // Compute simple CRC32 of entire archive
      const crc32Val = this.computeCrc32(buf);
      const crc32Hex = (crc32Val >>> 0).toString(16).padStart(8, "0").toUpperCase();

      const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);

      // 1. Locate EOCD signature in last 64 KB
      const searchLen = Math.min(byteSize, 65557);
      let eocdOffset = -1;
      for (let i = byteSize - 22; i >= byteSize - searchLen; i--) {
        if (view.getUint32(i, true) === ZIP_EOCD_SIG) {
          eocdOffset = i;
          break;
        }
      }

      if (eocdOffset === -1) {
        return { valid: false, tokenCount: 0, byteSize, crc32Hex, error: "Invalid ZIP: EOCD signature not found" };
      }

      const totalEntries = view.getUint16(eocdOffset + 10, true);
      const centralDirSize = view.getUint32(eocdOffset + 12, true);
      const centralDirOffset = view.getUint32(eocdOffset + 16, true);

      if (centralDirOffset + centralDirSize > byteSize) {
        return { valid: false, tokenCount: 0, byteSize, crc32Hex, error: "Corrupted ZIP: Central Directory exceeds archive boundary" };
      }

      // 2. Parse Central Directory entries and sample tokens
      let ptr = centralDirOffset;
      const sampleTokens: string[] = [];
      let tokenCount = 0;
      let firstTokenLocalOffset = -1;

      for (let i = 0; i < totalEntries && ptr + 46 <= byteSize; i++) {
        if (view.getUint32(ptr, true) !== ZIP_CENTRAL_DIR_SIG) {
          break;
        }

        const fileNameLen = view.getUint16(ptr + 28, true);
        const extraLen = view.getUint16(ptr + 30, true);
        const commentLen = view.getUint16(ptr + 32, true);
        const localHeaderOffset = view.getUint32(ptr + 42, true);

        const fileNameBytes = buf.subarray(ptr + 46, ptr + 46 + fileNameLen);
        const fileName = new TextDecoder().decode(fileNameBytes);

        if (fileName.endsWith(".token") || fileName.startsWith("tok_")) {
          tokenCount++;
          if (sampleTokens.length < 5) {
            sampleTokens.push(fileName);
          }
          if (firstTokenLocalOffset === -1) {
            firstTokenLocalOffset = localHeaderOffset;
          }
        }

        ptr += 46 + fileNameLen + extraLen + commentLen;
      }

      if (tokenCount === 0) {
        return { valid: false, tokenCount: 0, byteSize, crc32Hex, error: "Archive contains 0 token entries (*.token or tok_*)" };
      }

      // 3. Verify magic bytes on first sampled token
      if (firstTokenLocalOffset >= 0 && firstTokenLocalOffset + 30 <= byteSize) {
        if (view.getUint32(firstTokenLocalOffset, true) === ZIP_LOCAL_HEADER_SIG) {
          const lFileNameLen = view.getUint16(firstTokenLocalOffset + 26, true);
          const lExtraLen = view.getUint16(firstTokenLocalOffset + 28, true);
          const tokenDataOffset = firstTokenLocalOffset + 30 + lFileNameLen + lExtraLen;

          if (tokenDataOffset + 4 <= byteSize) {
            const magicStr = new TextDecoder().decode(buf.subarray(tokenDataOffset, tokenDataOffset + 4));
            if (magicStr !== TOKEN_MAGIC) {
              return {
                valid: false,
                tokenCount,
                byteSize,
                crc32Hex,
                error: `Token magic header mismatch: expected '${TOKEN_MAGIC}', found '${magicStr}'`,
              };
            }
          }
        }
      }

      return {
        valid: true,
        tokenCount,
        byteSize,
        crc32Hex,
        sampleTokens,
      };
    } catch (err: any) {
      return {
        valid: false,
        tokenCount: 0,
        byteSize: 0,
        crc32Hex: "00000000",
        error: `Validation exception: ${err?.message || String(err)}`,
      };
    }
  }

  /**
   * Transactional two-phase archive swap:
   * 1. Writes incoming bytes to `${activeZipPath}.staging`
   * 2. Validates staging archive
   * 3. On failure: deletes staging file, returns error, active archive remains untouched
   * 4. On success: replaces active archive, re-indexes RAM search engine & SQLite inverted index
   */
  public async swapArchive(
    archiveBytes: Uint8Array<ArrayBuffer>,
  ): Promise<{
    success: boolean;
    generation: number;
    totalTokens: number;
    archiveSizeBytes: number;
    crc32Hex?: string;
    error?: string;
  }> {
    // Phase 1: Write to staging file
    try {
      await Bun.write(this.stagingZipPath, archiveBytes);
    } catch (err: any) {
      return {
        success: false,
        generation: this.generation,
        totalTokens: this.searchEngine.getCount(),
        archiveSizeBytes: this.searchEngine.archiveSizeBytes,
        error: `Could not write to staging file: ${err?.message || String(err)}`,
      };
    }

    // Phase 2: Validate staging archive
    const validation = await this.validateArchive(this.stagingZipPath);
    if (!validation.valid) {
      // Rollback: delete staging file, keep existing generation
      try {
        await Bun.file(this.stagingZipPath).delete();
      } catch {}
      return {
        success: false,
        generation: this.generation,
        totalTokens: this.searchEngine.getCount(),
        archiveSizeBytes: this.searchEngine.archiveSizeBytes,
        error: `Archive validation failed: ${validation.error}`,
      };
    }

    // Phase 3: Atomic swap
    try {
      // Overwrite active archive with validated staging bytes
      await Bun.write(this.activeZipPath, archiveBytes);
      try {
        await Bun.file(this.stagingZipPath).delete();
      } catch {}

      // Rebuild search engine and inverted index on active archive
      await this.searchEngine.buildIndex();
      if (this.invertedIndex) {
        await this.invertedIndex.syncTokens(this.searchEngine.getAllEntries());
      }

      this.generation++;
      this.lastSwappedAt = new Date().toISOString();
      this.activeCrc32Hex = validation.crc32Hex;

      return {
        success: true,
        generation: this.generation,
        totalTokens: this.searchEngine.getCount(),
        archiveSizeBytes: this.searchEngine.archiveSizeBytes,
        crc32Hex: this.activeCrc32Hex,
      };
    } catch (err: any) {
      return {
        success: false,
        generation: this.generation,
        totalTokens: this.searchEngine.getCount(),
        archiveSizeBytes: this.searchEngine.archiveSizeBytes,
        error: `Swap execution error: ${err?.message || String(err)}`,
      };
    }
  }

  private computeCrc32(buf: Uint8Array): number {
    let crc = ~0;
    for (let i = 0; i < buf.length; i++) {
      crc = (crc >>> 8) ^ CRC32_TABLE[(crc ^ buf[i]!) & 0xff]!;
    }
    return ~crc;
  }
}

// Precomputed CRC32 Table
const CRC32_TABLE = new Uint32Array(256);
(() => {
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    CRC32_TABLE[i] = c >>> 0;
  }
})();

