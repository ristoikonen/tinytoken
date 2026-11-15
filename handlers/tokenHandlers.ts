/**
 * HTTP API Handlers for TinyToken (handlers/tokenHandlers.ts)
 * Exposes sub-millisecond search, VQ ADC visual similarity, on-demand ZIP token slicing,
 * SNARK/SHARK provenance verification, and homomorphic GF(2^8) state mutation.
 */

import {
  getAppDocumentDir,
  getAppImageDir,
  PreviewFormat,
  redactImageDirForDisplay,
  TokenFlags,
} from "../models/token";
import { applyHomomorphicMaskGF28 } from "../utils/galois";
import { previewFormatToLabel, previewFormatToMime } from "../utils/imagePreview";
import type { Win11TokenPipeline } from "../utils/indexer";
import {
  errorResponse,
  fileResponse,
  jsonResponse,
  verifySameOriginMutation,
} from "../utils/response";
import {
  decodeBinaryToken,
  extractTokenMetadata,
  extractTokenPreviewBytes,
  inspectBinaryTokenHeader,
} from "../utils/tokenCodec";
import { VQ_CODEBOOK, vqGridToHex } from "../utils/vqCodebook";
import {
  DEFAULT_OBSERVATION_LATITUDE,
  DEFAULT_OBSERVATION_LONGITUDE,
  visualTraitRegistry,
} from "../utils/visualTraitRegistry";
import type { VqInvertedIndexEngine } from "../utils/vqInvertedIndex";
import {
  buildNativeTokenZipBuffer,
  type CompressedTokenSearchEngine,
} from "../utils/zipEngine";

export class TokenApiHandlers {
  private writeLockInProgress = false;

  constructor(
    private engine: CompressedTokenSearchEngine,
    private pipeline: Win11TokenPipeline,
    private imageDir = getAppImageDir(),
    private invertedIndex?: VqInvertedIndexEngine,
    private documentDir = getAppDocumentDir(),
  ) { }

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
    const dinoRamBytes = entries.length * 384;
    const bgeRamBytes = entries.length * 384;

    return jsonResponse({
      success: true,
      engine: "TinyToken GF(2^8) + DINOv3-384B + BGE-384B + VQ-16x16 + SHARK/SNARK v1",
      tokenCount: entries.length,
      snarkVerifiedCount,
      archivePath: this.engine.getZipFilePath(),
      sourceDirectory: redactImageDirForDisplay(this.imageDir),
      imageDirectory: this.imageDir,
      documentDirectory: this.documentDir,
      archiveSizeBytes: this.engine.archiveSizeBytes,
      totalTokenUncompressedBytes,
      vqRamFootprintBytes: vqRamBytes,
      dinoRamFootprintBytes: dinoRamBytes,
      bgeRamFootprintBytes: bgeRamBytes,
      bootIndexTimeMs: this.engine.lastBootIndexMs,
      codebookSize: VQ_CODEBOOK.length,
      useMyGooglePhotos: Bun.env.USE_MY_GOOGLE_PHOTOS === "true",
      appZipUse: this.engine.isZipMode ? this.engine.isZipMode() : Bun.env.APP_ZIP_USE !== "false",
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
    const t0 = Bun.nanoseconds();
    const q = url.searchParams.get("q") ?? "";
    const similarTo = url.searchParams.get("similarTo") ?? "";
    const mode = url.searchParams.get("mode") ?? (similarTo ? "vq" : "ocr");
    const limit = Math.min(250, Math.max(1, Number(url.searchParams.get("limit") ?? 200)));

    let hits;
    if (mode === "vq" && similarTo) {
      hits = this.engine.searchSimilarVQ(similarTo, q, limit);
    } else if (q.trim() && this.invertedIndex) {
      const ftsIds = this.invertedIndex.searchFts5(q, limit);
      if (ftsIds.length > 0) {
        const ftsHits = this.engine
          .searchOCR("", 500)
          .filter((h) => ftsIds.includes(h.id))
          .sort((a, b) => ftsIds.indexOf(a.id) - ftsIds.indexOf(b.id));
        const subHits = this.engine.searchOCR(q, limit);
        const seen = new Set(ftsHits.map((h) => h.id));
        for (const sh of subHits) {
          if (!seen.has(sh.id)) ftsHits.push(sh);
        }
        hits = ftsHits.slice(0, limit);
      } else {
        hits = this.engine.searchOCR(q, limit);
      }
    } else {
      hits = this.engine.searchOCR(q, limit);
    }

    const elapsedMs = Math.round(((Bun.nanoseconds() - t0) / 1e6) * 1000) / 1000;

    const sanitizedHits = hits.map((h) => ({
      ...h,
      ocrSnippet: h.ocrSnippet ? redactImageDirForDisplay(h.ocrSnippet) : h.ocrSnippet,
    }));

    return jsonResponse({
      success: true,
      mode: mode === "vq" && similarTo ? "vq-adc" : "ocr-index",
      query: q,
      similarTo: similarTo || null,
      count: sanitizedHits.length,
      totalIndexed: this.engine.getCount(),
      searchTimeMs: elapsedMs,
      hits: sanitizedHits,
    });
  }

  /**
   * GET /api/token/:id
   * Extracts a single 25 KB token on-demand via Bun.file().slice(), verifies GF(2^8) RS parity
   * and the 256-byte SNARK/SHARK provenance stub, and returns detailed byte-map telemetry.
   */
  public async handleGetToken(id: string): Promise<Response> {
    const t0 = Bun.nanoseconds();
    const rawToken = await this.engine.extractToken(id);
    if (!rawToken) {
      return errorResponse(`Token '${id}' not found in ZIP index`, 404);
    }

    const sliceTimeMs = Math.round(((Bun.nanoseconds() - t0) / 1e6) * 1000) / 1000;
    const tDecode = Bun.nanoseconds();
    const decoded = await decodeBinaryToken(rawToken);
    const decodeTimeMs = Math.round(((Bun.nanoseconds() - tDecode) / 1e6) * 1000) / 1000;
    const indexMeta = this.engine.getEntry(id);

    // Sample first 64 bytes of header & parity as hex for the Glow2 Byte-Map inspector
    const headerHexPreview = rawToken
      .subarray(0, 64)
      .toHex()
      .toUpperCase()
      .replace(/../g, "$& ")
      .trim();

    // Retrieve object registry intelligence associated with this token
    const evidenceList = await visualTraitRegistry.getEvidence();
    const tokenEvidence = evidenceList.filter(
      (e) =>
        e.observedTokenId === id ||
        e.observedTokenId === decoded.id ||
        (indexMeta && (e.observedTokenId === indexMeta.id || e.observedTokenId === indexMeta.filename)),
    );
    const recipes = await visualTraitRegistry.getRecipes();

    const objectAnnotations = tokenEvidence.map((e) => {
      const r = recipes.find(
        (rcp) => rcp.recipeId === e.recipeId || rcp.label.toLowerCase() === e.label.toLowerCase(),
      );
      return {
        id: e.evidenceId,
        recipeId: e.recipeId || r?.recipeId || `rcp_${e.label.toLowerCase().replace(/[^a-z0-9]/g, "_")}`,
        label: e.label,
        category: e.category,
        boundingBox: e.boundingBox,
        centroidIds: e.observedCentroids,
        source: e.source || r?.source || "human_verified",
        confidence: e.confidence ?? r?.confidence ?? 1.0,
        version: r?.version ?? 1,
        observationCount: r?.observationCount ?? 1,
        createdAt: r?.createdAt || e.utcTimestamp || e.observedAt,
        updatedAt: r?.updatedAt || e.utcTimestamp || e.observedAt,
        latitude: e.latitude,
        longitude: e.longitude,
      };
    });

    const primaryObjectLabel =
      objectAnnotations.length > 0
        ? `${objectAnnotations[0]!.label} · ${objectAnnotations[0]!.category}`
        : null;

    return jsonResponse({
      success: true,
      telemetry: {
        sliceExtractTimeMs: sliceTimeMs,
        rsAndSnarkVerifyTimeMs: decodeTimeMs,
        totalTimeMs: Math.round(((Bun.nanoseconds() - t0) / 1e6) * 1000) / 1000,
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
          hasDinoEmbedding: (decoded.flags & TokenFlags.HAS_DINO_EMBEDDING) !== 0,
          hasBgeEmbedding: (decoded.flags & TokenFlags.HAS_BGE_EMBEDDING) !== 0,
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
        contentId: decoded.contentId,
        shardBlockId: decoded.shardBlockId,
        shardIndex: decoded.shardIndex,
        homomorphicTagHex: `0x${decoded.homomorphicTag.toString(16).padStart(2, "0").toUpperCase()}`,
        embeddingModelId: decoded.embeddingModelId,
        bgeModelId: decoded.metadata.bgeModelId ?? null,
        rsParityValid: decoded.rsParityValid,
        rsCorrectedErrors: decoded.rsCorrectedErrors,
        snarkAttestation: decoded.snarkAttestation,
        vqHex: vqGridToHex(decoded.vqIndices),
        metadata: {
          ...decoded.metadata,
          ocrText: redactImageDirForDisplay(decoded.metadata.ocrText),
        },
        byteMap: decoded.byteMap,
        traits: visualTraitRegistry.evaluateGrid(decoded.vqIndices),
        headerHexPreview,
        objectAnnotations,
        primaryObjectLabel,
      },
    });
  }

  /**
   * GET /api/registry
   * Returns the full 256-codebook visual trait dictionary mapping.
   */
  public handleGetRegistry(): Response {
    return jsonResponse({
      success: true,
      registry: visualTraitRegistry.getAll(),
    });
  }

  /**
   * POST /api/token/:id/teach
   * Teaches an object by mapping the selected region & centroids to a named label.
   */
  public async handleTeachObject(id: string, req: Request): Promise<Response> {
    try {
      const body = (await req.json()) as {
        label?: string;
        category?: string;
        boundingBox?: [number, number, number, number];
        centroidIds?: number[];
      };

      const label = String(body.label || "").trim();
      const category = (body.category || "Silhouette") as any;
      const boundingBox = (Array.isArray(body.boundingBox) ? body.boundingBox : [0, 0, 1, 1]) as [
        number,
        number,
        number,
        number,
      ];
      let targetCentroids = Array.isArray(body.centroidIds)
        ? body.centroidIds.map(Number).filter((n) => !Number.isNaN(n) && n >= 0 && n <= 255)
        : [];

      if (!label) {
        return errorResponse("Object label is required", 400);
      }

      const rawToken = await this.engine.extractToken(id);
      if (!rawToken) {
        return errorResponse(`Token '${id}' not found`, 404);
      }

      // Fast inspection: extract VQ grid from Section I without full decode
      const insp = inspectBinaryTokenHeader(rawToken);

      // If centroidIds was empty, derive them from 16x16 grid and boundingBox
      if (targetCentroids.length === 0 && insp.vqIndices && insp.vqIndices.length === 256) {
        const [bx, by, bw, bh] = boundingBox;
        const startX = Math.max(0, Math.min(15, Math.floor(bx * 16)));
        const endX = Math.max(0, Math.min(15, Math.floor((bx + bw) * 16)));
        const startY = Math.max(0, Math.min(15, Math.floor(by * 16)));
        const endY = Math.max(0, Math.min(15, Math.floor((by + bh) * 16)));

        const extracted = new Set<number>();
        for (let y = startY; y <= endY; y++) {
          for (let x = startX; x <= endX; x++) {
            const cellIdx = y * 16 + x;
            extracted.add(insp.vqIndices[cellIdx]!);
          }
        }
        targetCentroids = Array.from(extracted);
      }

      let geoCoordinates: { latitude?: number; longitude?: number } | undefined;
      try {
        const meta = extractTokenMetadata(rawToken);
        geoCoordinates = meta.geoCoordinates;
      } catch {
        // Metadata unpack optional
      }

      const geo = geoCoordinates;
      const { taughtCentroids, annotation, evidence, recipe } = await visualTraitRegistry.teachObject(
        id,
        label,
        category,
        boundingBox,
        targetCentroids,
        {
          latitude: geo?.latitude ?? (body as any).latitude,
          longitude: geo?.longitude ?? (body as any).longitude,
          source: (body as any).source,
          confidence: (body as any).confidence,
        },
      );

      const updatedTraits = visualTraitRegistry.evaluateGrid(insp.vqIndices);

      return jsonResponse({
        success: true,
        message: `Taught '${label}' across ${taughtCentroids.length} centroids`,
        annotation,
        taughtCentroids,
        traits: updatedTraits,
        evidence,
        recipe,
      });
    } catch (err: any) {
      return errorResponse(`Failed to teach object: ${err?.message || String(err)}`, 500);
    }
  }

  /**
   * GET /api/objects
   * Returns all taught object annotations.
   */
  public async handleGetObjectAnnotations(): Promise<Response> {
    const annotations = await visualTraitRegistry.getObjectAnnotations();
    return jsonResponse({
      success: true,
      count: annotations.length,
      annotations,
    });
  }

  /**
   * GET /api/token/:id/preview
   * Streams the embedded Section III preview image directly from the sliced 25 KB token.
   */
  public async handleGetTokenPreview(id: string): Promise<Response> {
    const t0 = Bun.nanoseconds();
    const rawToken = await this.engine.extractToken(id);
    if (!rawToken) {
      return errorResponse(`Token '${id}' not found`, 404);
    }

    // Zero-copy inspection: read format from Section I & slice Section III preview directly
    const insp = inspectBinaryTokenHeader(rawToken);
    const previewBytes = extractTokenPreviewBytes(rawToken);
    const mimeType = previewFormatToMime(insp.previewFormat);
    const extractMs = ((Bun.nanoseconds() - t0) / 1e6).toFixed(3);

    const headers: Record<string, string> = {
      "Content-Type": mimeType,
      "Cache-Control": "no-cache",
      "X-Token-Extract-Ms": extractMs,
      "X-Preview-Format": insp.previewFormatLabel,
      "X-Content-Type-Options": "nosniff",
    };

    if (insp.previewFormat === PreviewFormat.SVG || mimeType === "image/svg+xml") {
      headers["Content-Security-Policy"] = "default-src 'none'; sandbox";
    }

    return new Response(new Uint8Array(previewBytes), {
      status: 200,
      headers,
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
    return fileResponse(file, file.type || "image/jpeg");
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

    return fileResponse(file, file.type || "application/pdf");
  }

  /**
   * POST /api/token/:id/mutate
   * Executes a blind Homomorphic State Transition in GF(2^8) on the raw 25 KB token slice.
   */
  public async handleHomomorphicMutate(id: string, req: Request): Promise<Response> {
    const forbidden = verifySameOriginMutation(req);
    if (forbidden) return forbidden;

    if (this.writeLockInProgress) {
      return errorResponse("Another archive write or rebuild is currently in progress", 409);
    }

    this.writeLockInProgress = true;
    try {
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

      const t0 = Bun.nanoseconds();
      const mutated = applyHomomorphicMaskGF28(
        rawToken,
        scalar || 0x5a,
        TokenFlags.HOMOMORPHIC_MASKED,
      );
      await this.engine.writeMutatedTokenInPlace(id, mutated);
      const decoded = await decodeBinaryToken(mutated);
      const elapsedUs = Math.round((Bun.nanoseconds() - t0) / 1e3);

      try {
        const stampPath = `${this.engine.getDataDirPath ? this.engine.getDataDirPath() : "./data"}/.indexed_dir`;
        await Bun.write(stampPath, this.imageDir);
      } catch {}

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
    } finally {
      this.writeLockInProgress = false;
    }
  }

  /**
   * POST /api/index/rebuild
   * Re-scans `imageDir` and `documentDir` and rebuilds the ZIP container.
   * Optionally accepts `{ sourceDirectory?: string }` to dynamically switch image directory.
   */
  public async handleRebuildIndex(req?: Request): Promise<Response> {
    const forbidden = verifySameOriginMutation(req);
    if (forbidden) return forbidden;

    if (this.writeLockInProgress) {
      return errorResponse("Another archive write or rebuild is currently in progress", 409);
    }

    let overrideDir: string | undefined;
    if (req && req.headers.get("content-type")?.includes("application/json")) {
      try {
        const body = (await req.json()) as { sourceDirectory?: string };
        if (body.sourceDirectory && typeof body.sourceDirectory === "string") {
          overrideDir = body.sourceDirectory.trim().replace(/\\/g, "/");
        }
      } catch {}
    }

    if (overrideDir) {
      this.imageDir = overrideDir;
    }

    this.writeLockInProgress = true;
    try {
      const buildRes = await this.pipeline.buildTokenArchive(overrideDir);
      await this.engine.buildIndex();
      if (this.invertedIndex) {
        await this.invertedIndex.syncTokens(this.engine.getAllEntries());
      }

      try {
        const stampPath = `${this.engine.getDataDirPath ? this.engine.getDataDirPath() : "./data"}/.indexed_dir`;
        await Bun.write(stampPath, this.imageDir);
      } catch {}

      return jsonResponse({
        success: true,
        sourceDirectory: redactImageDirForDisplay(this.imageDir),
        imageDirectory: this.imageDir,
        documentDirectory: this.documentDir,
        totalTokens: buildRes.totalTokens,
        imageTokens: buildRes.imageTokens,
        documentTokens: buildRes.documentTokens,
        archiveBytes: buildRes.archiveBytes,
        buildElapsedMs: buildRes.elapsedMs,
        bootIndexTimeMs: this.engine.lastBootIndexMs,
        metrics: buildRes.benchmarkReport,
      });
    } finally {
      this.writeLockInProgress = false;
    }
  }

  /**
   * POST /api/archive/create
   * Creates a new tokens.zip archive from:
   * 1. One or more folders (`{ folders: string[] }`), scanning both images & documents in each folder, or
   * 2. Selected/filtered token IDs (`{ tokenIds: string[] }`) sliced directly from the active index.
   */
  public async handleCreateArchive(req: Request): Promise<Response> {
    const forbidden = verifySameOriginMutation(req);
    if (forbidden) return forbidden;

    if (this.writeLockInProgress) {
      return errorResponse("Another archive write or rebuild is currently in progress", 409);
    }

    let body: { folders?: string[]; tokenIds?: string[] } = {};
    try {
      body = (await req.json()) as { folders?: string[]; tokenIds?: string[] };
    } catch {
      return errorResponse("Invalid JSON payload for /api/archive/create", 400);
    }

    const tokenIds = Array.isArray(body.tokenIds)
      ? body.tokenIds.map((id) => String(id || "").trim()).filter(Boolean)
      : [];
    const rawFolders = Array.isArray(body.folders)
      ? body.folders.map((f) => String(f || "").trim()).filter(Boolean)
      : [];

    if (tokenIds.length === 0 && rawFolders.length === 0) {
      return errorResponse(
        "Provide at least one folder in 'folders' or at least one token ID in 'tokenIds'",
        400,
      );
    }

    this.writeLockInProgress = true;
    try {
      const t0 = performance.now();

      // Mode 1: Create archive from selected/filtered token IDs in current index
      if (tokenIds.length > 0) {
        const items: Array<{
          entryName: string;
          tokenBytes: Uint8Array;
          meta: any;
        }> = [];
        for (const id of tokenIds) {
          const entry = this.engine.getEntry(id);
          if (!entry) continue;
          const tokenBytes = await this.engine.extractToken(entry.id);
          if (!tokenBytes) continue;
          const {
            localHeaderOffset: _localHeaderOffset,
            dataOffset: _dataOffset,
            compressedSize: _compressedSize,
            uncompressedSize: _uncompressedSize,
            compressionMethod: _compressionMethod,
            crc32: _crc32,
            ...meta
          } = entry;
          items.push({ entryName: entry.entryName, tokenBytes, meta });
        }

        if (items.length === 0) {
          return errorResponse("None of the requested tokenIds were found in the active index", 404);
        }

        const zipBuf = buildNativeTokenZipBuffer(items);
        await Bun.write(this.engine.getZipFilePath(), zipBuf);
        await this.engine.buildIndex();
        if (this.invertedIndex) {
          await this.invertedIndex.syncTokens(this.engine.getAllEntries());
        }

        const buildElapsedMs = Math.round(performance.now() - t0);
        return jsonResponse({
          success: true,
          mode: "selection",
          totalTokens: items.length,
          archiveBytes: zipBuf.length,
          buildElapsedMs,
        });
      }

      // Mode 2: Create archive from 1 or more folders (auto-routing images + documents)
      const resolvedFolders = rawFolders.map((f) =>
        f === "APP_IMAGE_DIR" ? getAppImageDir() : f.replace(/\\/g, "/"),
      );
      this.imageDir = resolvedFolders[0]!;

      const buildRes = await this.pipeline.buildTokenArchive({ folders: resolvedFolders });
      await this.engine.buildIndex();
      if (this.invertedIndex) {
        await this.invertedIndex.syncTokens(this.engine.getAllEntries());
      }

      try {
        const stampPath = `${this.engine.getDataDirPath ? this.engine.getDataDirPath() : "./data"}/.indexed_dir`;
        await Bun.write(stampPath, this.imageDir);
      } catch {}

      return jsonResponse({
        success: true,
        mode: "folders",
        folders: resolvedFolders,
        sourceDirectory: redactImageDirForDisplay(this.imageDir),
        imageDirectory: this.imageDir,
        documentDirectory: this.documentDir,
        totalTokens: buildRes.totalTokens,
        imageTokens: buildRes.imageTokens,
        documentTokens: buildRes.documentTokens,
        reusedTokens: buildRes.reusedTokens,
        archiveBytes: buildRes.archiveBytes,
        buildElapsedMs: buildRes.elapsedMs,
        bootIndexTimeMs: this.engine.lastBootIndexMs,
      });
    } finally {
      this.writeLockInProgress = false;
    }
  }

  /**
   * GET /api/archive/export
   * Streams the active tokens.zip archive (or a filtered subset when `?ids=tok_1,tok_2` is provided)
   * with `Content-Disposition: attachment` for browser Save As.
   */
  public async handleExportArchive(url: URL): Promise<Response> {
    const idsParam = (url.searchParams.get("ids") ?? "").trim();

    if (idsParam) {
      const requestedIds = idsParam
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
      const items: Array<{
        entryName: string;
        tokenBytes: Uint8Array;
        meta: any;
      }> = [];
      for (const id of requestedIds) {
        const entry = this.engine.getEntry(id);
        if (!entry) continue;
        const tokenBytes = await this.engine.extractToken(entry.id);
        if (!tokenBytes) continue;
        const {
          localHeaderOffset: _localHeaderOffset,
          dataOffset: _dataOffset,
          compressedSize: _compressedSize,
          uncompressedSize: _uncompressedSize,
          compressionMethod: _compressionMethod,
          crc32: _crc32,
          ...meta
        } = entry;
        items.push({ entryName: entry.entryName, tokenBytes, meta });
      }

      if (items.length === 0) {
        return errorResponse("No matching tokens found for export", 404);
      }

      const zipBuf = buildNativeTokenZipBuffer(items);
      return new Response(new Uint8Array(zipBuf), {
        status: 200,
        headers: {
          "Content-Type": "application/zip",
          "Content-Disposition": `attachment; filename="tinytoken_subset_${items.length}.zip"`,
          "Content-Length": String(zipBuf.length),
          "Cache-Control": "no-store",
        },
      });
    }

    const zipPath = this.engine.getZipFilePath();
    const isZip = this.engine.isZipMode ? this.engine.isZipMode() : true;
    const zipFile = Bun.file(zipPath);

    if (isZip && (await zipFile.exists()) && zipFile.size >= 22) {
      return new Response(zipFile, {
        status: 200,
        headers: {
          "Content-Type": "application/zip",
          "Content-Disposition": `attachment; filename="tokens.zip"`,
          "Content-Length": String(zipFile.size),
          "Cache-Control": "no-store",
        },
      });
    }

    // Fallback when APP_ZIP_USE=false: pack all active entries on the fly
    const entries = this.engine.getAllEntries();
    if (entries.length === 0) {
      return errorResponse("Active archive contains 0 tokens", 404);
    }
    const items: Array<{
      entryName: string;
      tokenBytes: Uint8Array;
      meta: any;
    }> = [];
    for (const entry of entries) {
      const tokenBytes = await this.engine.extractToken(entry.id);
      if (!tokenBytes) continue;
      const {
        localHeaderOffset: _localHeaderOffset,
        dataOffset: _dataOffset,
        compressedSize: _compressedSize,
        uncompressedSize: _uncompressedSize,
        compressionMethod: _compressionMethod,
        crc32: _crc32,
        ...meta
      } = entry;
      items.push({ entryName: entry.entryName, tokenBytes, meta });
    }
    const zipBuf = buildNativeTokenZipBuffer(items);
    return new Response(new Uint8Array(zipBuf), {
      status: 200,
      headers: {
        "Content-Type": "application/zip",
        "Content-Disposition": `attachment; filename="tokens.zip"`,
        "Content-Length": String(zipBuf.length),
        "Cache-Control": "no-store",
      },
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

    const t0 = Bun.nanoseconds();
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
    const elapsedMs = Math.round(((Bun.nanoseconds() - t0) / 1e6) * 1000) / 1000;

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

