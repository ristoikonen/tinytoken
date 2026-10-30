/**
 * HTTP API Handlers for TinyToken (handlers/tokenHandlers.ts)
 * Exposes sub-millisecond search, VQ ADC visual similarity, on-demand ZIP token slicing,
 * SNARK/SHARK provenance verification, and homomorphic GF(2^8) state mutation.
 */

import { TokenFlags } from "../models/token";
import { applyHomomorphicMaskGF28 } from "../utils/galois";
import { previewFormatToLabel, previewFormatToMime } from "../utils/imagePreview";
import type { Win11TokenPipeline } from "../utils/indexer";
import { errorResponse, jsonResponse } from "../utils/response";
import { decodeBinaryToken } from "../utils/tokenCodec";
import { VQ_CODEBOOK, vqGridToHex } from "../utils/vqCodebook";
import type { VqInvertedIndexEngine } from "../utils/vqInvertedIndex";
import type { CompressedTokenSearchEngine } from "../utils/zipEngine";

export class TokenApiHandlers {
  constructor(
    private engine: CompressedTokenSearchEngine,
    private pipeline: Win11TokenPipeline,
    private imageDir = "./images",
    private invertedIndex?: VqInvertedIndexEngine,
    private documentDir = "./documents",
  ) {}

  /**
   * GET /api/stats
   */
  public handleStats(): Response {
    const entries = this.engine.getAllEntries();
    let snarkVerifiedCount = 0;
    let totalTokenUncompressedBytes = 0;

    for (const e of entries) {
      if (e.snarkVerified) snarkVerifiedCount++;
      totalTokenUncompressedBytes += e.uncompressedSize;
    }

    const vqRamBytes = entries.length * 256;

    return jsonResponse({
      success: true,
      engine: "TinyToken GF(2^8) + VQ-16x16 + SHARK/SNARK v2",
      tokenCount: entries.length,
      snarkVerifiedCount,
      archivePath: this.engine.getZipFilePath(),
      archiveSizeBytes: this.engine.archiveSizeBytes,
      totalTokenUncompressedBytes,
      vqRamFootprintBytes: vqRamBytes,
      bootIndexTimeMs: this.engine.lastBootIndexMs,
      codebookSize: VQ_CODEBOOK.length,
    });
  }

  /**
   * GET /api/codebook
   * Returns the 256-entry RGB codebook so the client UI can render 16x16 VQ neural thumbnails in 0ms.
   */
  public handleCodebook(): Response {
    return jsonResponse({
      success: true,
      centroids: VQ_CODEBOOK,
    });
  }

  /**
   * GET /api/search?q=...&mode=ocr|vq&similarTo=...&limit=50
   */
  public handleSearch(url: URL): Response {
    const t0 = performance.now();
    const q = url.searchParams.get("q") ?? "";
    const similarTo = url.searchParams.get("similarTo") ?? "";
    const mode = url.searchParams.get("mode") ?? (similarTo ? "vq" : "ocr");
    const limit = Math.min(250, Math.max(1, Number(url.searchParams.get("limit") ?? 60)));

    const hits =
      mode === "vq" && similarTo
        ? this.engine.searchSimilarVQ(similarTo, q, limit)
        : this.engine.searchOCR(q, limit);

    const elapsedMs = Math.round((performance.now() - t0) * 1000) / 1000;

    return jsonResponse({
      success: true,
      mode: mode === "vq" && similarTo ? "vq-adc" : "ocr-index",
      query: q,
      similarTo: similarTo || null,
      count: hits.length,
      totalIndexed: this.engine.getCount(),
      searchTimeMs: elapsedMs,
      hits,
    });
  }

  /**
   * GET /api/token/:id
   * Extracts a single 25 KB token on-demand via Bun.file().slice(), verifies GF(2^8) RS parity
   * and the 256-byte SNARK/SHARK provenance stub, and returns detailed byte-map telemetry.
   */
  public async handleGetToken(id: string): Promise<Response> {
    const t0 = performance.now();
    const rawToken = await this.engine.extractToken(id);
    if (!rawToken) {
      return errorResponse(`Token '${id}' not found in ZIP index`, 404);
    }

    const sliceTimeMs = Math.round((performance.now() - t0) * 1000) / 1000;
    const tDecode = performance.now();
    const decoded = decodeBinaryToken(rawToken);
    const decodeTimeMs = Math.round((performance.now() - tDecode) * 1000) / 1000;
    const indexMeta = this.engine.getEntry(id);

    // Sample first 64 bytes of header & parity as hex for the Glow2 Byte-Map inspector
    let headerHexPreview = "";
    for (let i = 0; i < Math.min(64, rawToken.length); i++) {
      headerHexPreview += (rawToken[i] ?? 0).toString(16).padStart(2, "0").toUpperCase() + " ";
    }

    return jsonResponse({
      success: true,
      telemetry: {
        sliceExtractTimeMs: sliceTimeMs,
        rsAndSnarkVerifyTimeMs: decodeTimeMs,
        totalTimeMs: Math.round((performance.now() - t0) * 1000) / 1000,
        zipDataOffset: indexMeta?.dataOffset ?? 0,
        zipCompressedSize: indexMeta?.compressedSize ?? rawToken.length,
        zipCrc32Hex: (indexMeta?.crc32 ?? 0).toString(16).padStart(8, "0").toUpperCase(),
      },
      token: {
        id: decoded.id,
        magic: decoded.magic,
        version: decoded.version,
        flags: decoded.flags,
        flagLabels: {
          hasVqGrid: (decoded.flags & TokenFlags.HAS_VQ_GRID) !== 0,
          snarkAttested: (decoded.flags & TokenFlags.SNARK_ATTESTED) !== 0,
          quicklzCompressed: (decoded.flags & TokenFlags.QUICKLZ_COMPRESSED) !== 0,
          homomorphicMasked: (decoded.flags & TokenFlags.HOMOMORPHIC_MASKED) !== 0,
          multiPageDoc: (decoded.flags & TokenFlags.MULTI_PAGE_DOC) !== 0,
        },
        uuidHex: decoded.uuidHex,
        originalFileSize: decoded.originalFileSize,
        width: decoded.width,
        height: decoded.height,
        page: decoded.page,
        previewFormat: previewFormatToLabel(decoded.previewFormat),
        previewMimeType: previewFormatToMime(decoded.previewFormat),
        previewUrl: `/api/token/${encodeURIComponent(decoded.id)}/preview`,
        merkleRootHex: decoded.merkleRootHex,
        shardBlockId: decoded.shardBlockId,
        shardIndex: decoded.shardIndex,
        homomorphicTagHex: `0x${decoded.homomorphicTag.toString(16).padStart(2, "0").toUpperCase()}`,
        rsParityValid: decoded.rsParityValid,
        rsCorrectedErrors: decoded.rsCorrectedErrors,
        snarkAttestation: decoded.snarkAttestation,
        vqHex: vqGridToHex(decoded.vqIndices),
        metadata: decoded.metadata,
        byteMap: decoded.byteMap,
        headerHexPreview: headerHexPreview.trim(),
      },
    });
  }

  /**
   * GET /api/token/:id/preview
   * Streams the embedded Section III preview image directly from the sliced 25 KB token.
   */
  public async handleGetTokenPreview(id: string): Promise<Response> {
    const t0 = performance.now();
    const rawToken = await this.engine.extractToken(id);
    if (!rawToken) {
      return errorResponse(`Token '${id}' not found`, 404);
    }

    const decoded = decodeBinaryToken(rawToken);
    const mimeType = previewFormatToMime(decoded.previewFormat);
    const extractMs = (performance.now() - t0).toFixed(3);

    return new Response(new Uint8Array(decoded.previewBytes), {
      status: 200,
      headers: {
        "Content-Type": mimeType,
        "Cache-Control": "public, max-age=3600",
        "X-Token-Extract-Ms": extractMs,
        "X-RS-Parity-Valid": String(decoded.rsParityValid),
        "X-SNARK-Attested": String(decoded.snarkAttestation.valid),
      },
    });
  }

  /**
   * Sanitizes a requested asset filename while preserving spaces and parentheses
   * (e.g. `grid (2).svg`, `Beyond Deviation and Deficit (1).pdf`).
   */
  private sanitizeFilename(name: string): string | null {
    const trimmed = name.trim();
    if (
      !trimmed ||
      trimmed.includes("..") ||
      trimmed.includes("/") ||
      trimmed.includes("\\") ||
      /[\x00-\x1f<>:"|?*]/.test(trimmed)
    ) {
      return null;
    }
    return trimmed;
  }

  /**
   * GET /api/image/:name
   * Streams the original full-resolution image from `./images` using Bun.file().
   */
  public async handleGetSourceImage(name: string): Promise<Response> {
    const safeName = this.sanitizeFilename(name);
    if (!safeName) {
      return errorResponse("Invalid image filename", 400);
    }

    const path = `${this.imageDir}/${safeName}`;
    const file = Bun.file(path);
    if (!(await file.exists())) {
      return errorResponse(`Image '${safeName}' not found`, 404);
    }

    return new Response(file, {
      status: 200,
      headers: {
        "Content-Type": file.type || "image/jpeg",
        "Cache-Control": "public, max-age=3600",
      },
    });
  }

  /**
   * GET /api/document/:name
   * Streams the original document from `./documents` using Bun.file().
   */
  public async handleGetSourceDocument(name: string): Promise<Response> {
    const safeName = this.sanitizeFilename(name);
    if (!safeName) {
      return errorResponse("Invalid document filename", 400);
    }

    const path = `${this.documentDir}/${safeName}`;
    const file = Bun.file(path);
    if (!(await file.exists())) {
      return errorResponse(`Document '${safeName}' not found`, 404);
    }

    return new Response(file, {
      status: 200,
      headers: {
        "Content-Type": file.type || "application/pdf",
        "Cache-Control": "public, max-age=3600",
      },
    });
  }

  /**
   * POST /api/token/:id/mutate
   * Executes a blind Homomorphic State Transition in GF(2^8) on the raw 25 KB token slice.
   */
  public async handleHomomorphicMutate(id: string, req: Request): Promise<Response> {
    const rawToken = await this.engine.extractToken(id);
    if (!rawToken) {
      return errorResponse(`Token '${id}' not found`, 404);
    }

    let scalar = 0x5a;
    try {
      const body = (await req.json()) as { scalar?: number };
      if (typeof body.scalar === "number") {
        scalar = body.scalar & 0xff;
      }
    } catch {
      // use default scalar
    }

    const t0 = performance.now();
    const mutated = applyHomomorphicMaskGF28(
      rawToken,
      scalar || 0x5a,
      TokenFlags.HOMOMORPHIC_MASKED,
    );
    await this.engine.writeMutatedTokenInPlace(id, mutated);
    const decoded = decodeBinaryToken(mutated);
    const elapsedUs = Math.round((performance.now() - t0) * 1000);

    return jsonResponse({
      success: true,
      id: decoded.id,
      homomorphicTimeUs: elapsedUs,
      flags: decoded.flags,
      homomorphicMasked: (decoded.flags & TokenFlags.HOMOMORPHIC_MASKED) !== 0,
      homomorphicTagHex: `0x${decoded.homomorphicTag.toString(16).padStart(2, "0").toUpperCase()}`,
      rsParityValid: decoded.rsParityValid,
      snarkValid: decoded.snarkAttestation.valid,
    });
  }

  /**
   * POST /api/index/rebuild
   * Re-scans `./images` (`IMAGE_DIR`) and `./documents` (`DOCUMENT_DIR`) and rebuilds the ZIP container.
   */
  public async handleRebuildIndex(_req?: Request): Promise<Response> {
    const buildRes = await this.pipeline.buildTokenArchive();
    await this.engine.buildIndex();
    if (this.invertedIndex) {
      await this.invertedIndex.syncTokens(this.engine.getAllEntries());
    }

    return jsonResponse({
      success: true,
      totalTokens: buildRes.totalTokens,
      imageTokens: buildRes.imageTokens,
      documentTokens: buildRes.documentTokens,
      archiveBytes: buildRes.archiveBytes,
      buildElapsedMs: buildRes.elapsedMs,
      bootIndexTimeMs: this.engine.lastBootIndexMs,
    });
  }

  /**
   * GET /api/search/pattern?tokens=42,85,128&threshold=0.5
   * Or GET /api/search/pattern?similarTo=tok_00001_HectorDoor.png&threshold=0.5
   * Queries the LibSQL / Turso Discrete VQ Codebook Inverted Index via set intersection.
   */
  public async handlePatternSearch(url: URL): Promise<Response> {
    if (!this.invertedIndex) {
      return errorResponse("VQ Inverted Index is not initialized", 503);
    }

    const t0 = performance.now();
    const threshold = Number(url.searchParams.get("threshold") ?? 0.5);
    const limit = Math.min(250, Math.max(1, Number(url.searchParams.get("limit") ?? 50)));
    const similarTo = url.searchParams.get("similarTo") ?? "";
    const tokensParam = url.searchParams.get("tokens") ?? "";

    let queryTokens: number[] = [];
    if (similarTo) {
      const target = this.engine.getEntry(similarTo);
      if (!target) {
        return errorResponse(`Token '${similarTo}' not found`, 404);
      }
      queryTokens = Array.from(target.vqIndices);
    } else if (tokensParam) {
      queryTokens = tokensParam
        .split(",")
        .map((s) => Number.parseInt(s.trim(), 10))
        .filter((n) => !Number.isNaN(n));
    }

    const matches = await this.invertedIndex.findImagesByTokenPattern(
      queryTokens,
      threshold,
      limit,
    );
    const elapsedMs = Math.round((performance.now() - t0) * 1000) / 1000;

    return jsonResponse({
      success: true,
      mode: "vq-inverted-index-sql",
      threshold,
      uniqueQueryCentroids: new Set(queryTokens).size,
      searchTimeMs: elapsedMs,
      count: matches.length,
      matches,
    });
  }
}

