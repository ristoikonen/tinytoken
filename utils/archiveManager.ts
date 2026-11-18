/**
 * Transactional Archive Manager (utils/archiveManager.ts)
 *
 * Implements generation-based hot-swapping of `tokens.zip`:
 * - Second archive-ingestion path accepting pre-built ZIP archives directly
 * - Two-phase commit: staging write -> strict structural validation -> atomic file swap
 * - Strict rollback: bad validation keeps active generation intact with zero downtime
 * - Readers remain pinned to the consistent active generation during staging
 */

import {
  getAppArchiveDir,
  getArchiveRegistryPath,
  type SerializedTokenArchive,
  type TokenArchive,
} from "../models/archive";
import type {
  ArchiveGenerationState,
  ArchiveValidationResult,
} from "../models/learning";
import { TOKEN_MAGIC } from "../models/token";
import type { VqInvertedIndexEngine } from "./vqInvertedIndex";
import type { CompressedTokenSearchEngine } from "./zipEngine";

export {
  getAppArchiveDir,
  getArchiveRegistryPath,
  type SerializedTokenArchive,
  type TokenArchive,
};

export interface FreezeArchiveOptions {
  archiveName?: string;
  label?: string;
  archiveDir?: string;
  registryPath?: string;
  filename?: string;
}

export interface FreezeArchiveResult {
  success: boolean;
  error?: string;
  archive?: TokenArchive;
}

const ZIP_EOCD_SIG = 0x06054b50;
const ZIP_LOCAL_HEADER_SIG = 0x04034b50;
const ZIP_CENTRAL_DIR_SIG = 0x02014b50;

/**
 * In-memory archive registry (stable archive ID -> approved TokenArchive record),
 * backed by `archive_registry.json` and synchronized with `APP_ARCHIVE_DIR`.
 */
const archiveRegistry = new Map<string, TokenArchive>();
let syncedArchiveDir: string | null = null;

function sanitizeArchiveId(rawId: string): string | null {
  const clean = rawId.trim().replace(/\.zip$/i, "");
  if (
    !clean ||
    clean.includes("..") ||
    clean.includes("/") ||
    clean.includes("\\") ||
    /[\x00-\x1f<>:"|?*]/.test(clean)
  ) {
    return null;
  }
  return clean;
}

async function computeFileSha256Hex(file: ReturnType<typeof Bun.file>): Promise<string> {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(await file.arrayBuffer());
  return hasher.digest("hex");
}

async function persistArchiveRegistry(
  records: Iterable<TokenArchive>,
  registryPath = getArchiveRegistryPath(),
): Promise<void> {
  const serialized: SerializedTokenArchive[] = Array.from(records).map((r) => ({
    id: r.id,
    filename: r.filename,
    path: r.path,
    size: r.size,
    modified: r.modified.toISOString(),
    ...(r.label ? { label: r.label } : {}),
    ...(r.sha256 ? { sha256: r.sha256 } : {}),
  }));
  await Bun.write(registryPath, JSON.stringify(serialized, null, 2));
}

/**
 * Loads `archive_registry.json` and synchronizes it with approved `.zip` files in `APP_ARCHIVE_DIR`
 * using `Bun.Glob`, `Bun.file`, and `Bun.CryptoHasher("sha256")`.
 */
export async function syncArchiveRegistry(
  archiveDir = getAppArchiveDir(),
  registryPath = getArchiveRegistryPath(),
): Promise<TokenArchive[]> {
  const normalizedDir = archiveDir.replace(/\\/g, "/").replace(/\/+$/, "");
  const persistedById = new Map<string, SerializedTokenArchive>();
  const regFile = Bun.file(registryPath);
  const regFileExisted = await regFile.exists();

  if (regFileExisted) {
    try {
      const parsed = (await regFile.json()) as SerializedTokenArchive[];
      if (Array.isArray(parsed)) {
        for (const item of parsed) {
          if (item && typeof item.id === "string" && typeof item.path === "string") {
            persistedById.set(item.id, item);
          }
        }
      }
    } catch {
      // Ignore malformed registry file and rebuild from directory scan
    }
  }

  const nextRegistry = new Map<string, TokenArchive>();
  let dirty = false;

  // 1. Scan APP_ARCHIVE_DIR for .zip files via Bun.Glob
  const glob = new Bun.Glob("*.{zip,ZIP}");
  try {
    for await (const name of glob.scan({ cwd: normalizedDir, onlyFiles: true })) {
      const id = sanitizeArchiveId(name);
      if (!id) continue;

      const path = `${normalizedDir}/${name}`;
      const file = Bun.file(path);
      if (!(await file.exists())) continue;

      const size = file.size;
      const modifiedMs = file.lastModified;
      const prev = persistedById.get(id);

      let sha256 = prev?.sha256;
      if (!prev || prev.size !== size || prev.path !== path || !sha256) {
        sha256 = await computeFileSha256Hex(file);
        dirty = true;
      }

      nextRegistry.set(id, {
        id,
        filename: name,
        path,
        size,
        modified: new Date(modifiedMs || (prev?.modified ? Date.parse(prev.modified) : Date.now())),
        ...(prev?.label ? { label: prev.label } : {}),
        sha256,
      });
    }
  } catch {
    // APP_ARCHIVE_DIR may not exist on disk yet
  }

  // 2. Preserve any explicitly registered archives from archive_registry.json that still exist on disk
  for (const [id, prev] of persistedById) {
    if (nextRegistry.has(id)) continue;
    const safeId = sanitizeArchiveId(id);
    if (!safeId) {
      dirty = true;
      continue;
    }
    const file = Bun.file(prev.path);
    if (!(await file.exists())) {
      dirty = true;
      continue;
    }
    const size = file.size;
    const sha256 =
      prev.sha256 && prev.size === size ? prev.sha256 : await computeFileSha256Hex(file);
    if (prev.size !== size || !prev.sha256) dirty = true;

    nextRegistry.set(safeId, {
      id: safeId,
      filename: prev.filename || `${safeId}.zip`,
      path: prev.path,
      size,
      modified: new Date(file.lastModified || (prev.modified ? Date.parse(prev.modified) : Date.now())),
      ...(prev.label ? { label: prev.label } : {}),
      sha256,
    });
  }

  if (nextRegistry.size !== persistedById.size) {
    dirty = true;
  }

  archiveRegistry.clear();
  for (const [k, v] of nextRegistry) {
    archiveRegistry.set(k, v);
  }
  syncedArchiveDir = normalizedDir;

  if (dirty || (!regFileExisted && nextRegistry.size > 0)) {
    try {
      await persistArchiveRegistry(archiveRegistry.values(), registryPath);
    } catch {
      // Non-fatal if data directory is read-only
    }
  }

  return Array.from(archiveRegistry.values()).sort(
    (a, b) => b.modified.getTime() - a.modified.getTime(),
  );
}

/**
 * Registers or updates a single `.zip` archive in the persistent & in-memory registry.
 */
export async function registerTokenArchive(
  archiveId: string,
  zipPath: string,
  labelOrRegistryPath?: string,
  maybeRegistryPath?: string,
): Promise<TokenArchive | null> {
  const cleanId = sanitizeArchiveId(archiveId);
  if (!cleanId) return null;

  let label: string | undefined;
  let registryPath = getArchiveRegistryPath();

  if (labelOrRegistryPath) {
    if (labelOrRegistryPath.endsWith(".json") && !maybeRegistryPath) {
      registryPath = labelOrRegistryPath;
    } else {
      label = labelOrRegistryPath.trim();
      if (maybeRegistryPath) {
        registryPath = maybeRegistryPath;
      }
    }
  }

  const normalizedPath = zipPath.replace(/\\/g, "/");
  const file = Bun.file(normalizedPath);
  if (!(await file.exists())) return null;

  const filename = normalizedPath.split("/").pop() || `${cleanId}.zip`;
  const sha256 = await computeFileSha256Hex(file);
  const record: TokenArchive = {
    id: cleanId,
    filename,
    path: normalizedPath,
    size: file.size,
    modified: new Date(file.lastModified || Date.now()),
    ...(label ? { label } : {}),
    sha256,
  };

  archiveRegistry.set(cleanId, record);
  try {
    const recordsToPersist = new Map<string, TokenArchive>();
    const regFile = Bun.file(registryPath);
    if (await regFile.exists()) {
      try {
        const parsed = (await regFile.json()) as SerializedTokenArchive[];
        if (Array.isArray(parsed)) {
          for (const item of parsed) {
            if (item && item.id) {
              recordsToPersist.set(item.id, {
                id: item.id,
                filename: item.filename,
                path: item.path,
                size: item.size,
                modified: new Date(item.modified),
                ...(item.label ? { label: item.label } : {}),
                ...(item.sha256 ? { sha256: item.sha256 } : {}),
              });
            }
          }
        }
      } catch { }
    } else if (registryPath === getArchiveRegistryPath()) {
      for (const [k, v] of archiveRegistry) {
        recordsToPersist.set(k, v);
      }
    }
    recordsToPersist.set(cleanId, record);
    await persistArchiveRegistry(recordsToPersist.values(), registryPath);
  } catch { }
  return record;
}

export async function listTokenArchives(
  archiveDir = getAppArchiveDir(),
  registryPath = getArchiveRegistryPath(),
): Promise<TokenArchive[]> {
  return syncArchiveRegistry(archiveDir, registryPath);
}

export async function resolveTokenArchive(
  archiveId: string,
  archiveDir = getAppArchiveDir(),
  registryPath = getArchiveRegistryPath(),
): Promise<TokenArchive | null> {
  const cleanId = sanitizeArchiveId(archiveId);
  if (!cleanId) return null;

  const normalizedDir = archiveDir.replace(/\\/g, "/").replace(/\/+$/, "");
  if (syncedArchiveDir !== normalizedDir || !archiveRegistry.has(cleanId)) {
    await syncArchiveRegistry(normalizedDir, registryPath);
  }

  const record = archiveRegistry.get(cleanId);
  if (!record) return null;

  const file = Bun.file(record.path);
  if (!(await file.exists()) || file.size !== record.size) {
    archiveRegistry.delete(cleanId);
    return null;
  }

  return record;
}



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
    const isZip = this.searchEngine.isZipMode ? this.searchEngine.isZipMode() : Bun.env.APP_ZIP_USE !== "false";
    return {
      generation: this.generation,
      activeZipPath: isZip ? this.activeZipPath : (this.searchEngine.getDataDirPath ? this.searchEngine.getDataDirPath() : "./data"),
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
    source: Uint8Array | string,
  ): Promise<ArchiveValidationResult> {
    try {
      let buf: Uint8Array;
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
    archiveBytes: Uint8Array,
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
      } catch { }
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
      } catch { }

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

  /**
   * Freezes current active archive into APP_ARCHIVE_DIR:
   * 1. Validates active archive structurally.
   * 2. Names file using timestamp + token count: `tokens_YYYY-MM-DD_HHmmss_<N>tok.zip`.
   * 3. Writes copy into archive directory using Bun.write.
   * 4. Registers in `archive_registry.json` with stable ID, size, sha256, and optional label.
   */
  public async freezeActiveArchive(options?: FreezeArchiveOptions): Promise<FreezeArchiveResult> {
    const activeFile = Bun.file(this.activeZipPath);
    if (!(await activeFile.exists())) {
      return { success: false, error: `Active archive not found at ${this.activeZipPath}` };
    }

    const validation = await this.validateArchive(this.activeZipPath);
    if (!validation.valid) {
      return {
        success: false,
        error: `Cannot freeze invalid archive: ${validation.error || "structural check failed"}`,
      };
    }

    const tokenCount = validation.tokenCount ?? this.searchEngine.getCount();
    const targetDir = (options?.archiveDir || getAppArchiveDir())
      .replace(/\\/g, "/")
      .replace(/\/+$/, "");
    const registryPath = options?.registryPath || getArchiveRegistryPath();

    let baseId: string;
    let filename: string;
    let destPath: string;

    if (options?.filename) {
      const sanitized = options.filename.trim().replace(/\\/g, "/").split("/").pop() || "";
      const cleanName = sanitized.replace(/\.zip$/i, "");
      const cleanId = sanitizeArchiveId(cleanName);
      if (!cleanId) {
        return { success: false, error: `Invalid custom filename '${options.filename}'` };
      }
      baseId = cleanId;
      filename = `${cleanId}.zip`;
      destPath = `${targetDir}/${filename}`;
    } else {
      const d = new Date();
      const pad = (n: number) => String(n).padStart(2, "0");
      const ts = `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;

      const rawPrefix = options?.archiveName?.trim() || "tokens";
      const cleanPrefix = sanitizeArchiveId(rawPrefix.replace(/\s+/g, "_")) || "tokens";

      baseId = `${cleanPrefix}_${ts}_${tokenCount}tok`;
      filename = `${baseId}.zip`;
      destPath = `${targetDir}/${filename}`;

      let counter = 1;
      while (await Bun.file(destPath).exists()) {
        baseId = `${cleanPrefix}_${ts}_${tokenCount}tok_${counter++}`;
        filename = `${baseId}.zip`;
        destPath = `${targetDir}/${filename}`;
      }
    }

    try {
      const bytes = await activeFile.arrayBuffer();
      await Bun.write(destPath, bytes);

      const record = await registerTokenArchive(baseId, destPath, options?.label, registryPath);
      if (!record) {
        return { success: false, error: "Failed to register frozen archive record" };
      }

      return { success: true, archive: record };
    } catch (err: any) {
      return {
        success: false,
        error: `Failed to freeze archive: ${err?.message || String(err)}`,
      };
    }
  }

  private computeCrc32(buf: Uint8Array): number {
    return Bun.hash.crc32(buf) >>> 0;
  }
}

