/**
 * Streaming Archive API Endpoints (v1) — handlers/streamingArchiveHandlers.ts
 *
 * Implements high-throughput binary streaming:
 * - Content negotiation: Accept: application/vnd.tinytoken.stream
 * - 4-byte Uint32LE length-prefixed binary TinyToken frames
 * - format=data (omits visual preview thumbnail, reducing wire size by ~80%)
 * - format=full (complete 25 KB token envelopes)
 * - Resumable streaming via `?after=N` (integer index or token ID) and `?limit=K`
 * - Fallback to explanatory text when standard text/plain or browsers inspect routes
 */

import type { ArchiveManager } from "../utils/archiveManager";
import { createFramedToken } from "../utils/streamConsumer";
import type { CompressedTokenSearchEngine } from "../utils/zipEngine";

export class StreamingArchiveHandlers {
  constructor(
    private searchEngine?: CompressedTokenSearchEngine,
    private archiveManager?: ArchiveManager,
  ) { }

  public setSearchEngine(engine: CompressedTokenSearchEngine): void {
    this.searchEngine = engine;
  }

  public setArchiveManager(manager: ArchiveManager): void {
    this.archiveManager = manager;
  }

  /**
   * GET /api/v1/archives
   * Lists or streams catalog of available token archives.
   */
  public handleListArchives(): Response {
    return new Response(
      [
        "ENDPOINT: GET /api/v1/archives",
        "STATUS: Active (Binary Streaming Available)",
        "DESCRIPTION: Streams or lists catalog of all available archives (active dataset + frozen archives in ./data/archive).",
        "FIELDS RETURNED: archiveId, filename, label, tokenCount, sizeBytes, sha256Hex, generation, createdAt.",
        "ACCEPT: application/vnd.tinytoken.stream for binary frame streaming, or application/json.",
      ].join("\n"),
      {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      },
    );
  }

  /**
   * GET /api/v1/archives/{archiveId}/manifest
   * Returns archive structural manifest & cryptographic attestation details.
   */
  public handleGetArchiveManifest(archiveId: string): Response {
    return new Response(
      [
        `ENDPOINT: GET /api/v1/archives/${archiveId}/manifest`,
        "STATUS: Active",
        `DESCRIPTION: Structural manifest and cryptographic integrity attestation for archive "${archiveId}".`,
        "FIELDS RETURNED: archiveId, totalTokens, archiveCrc32, merkleRootSha256, files: Array<{ filename, sizeBytes, crc32, offset }>, ed25519AttestationStatus.",
        "FORMAT: JSON manifest document.",
      ].join("\n"),
      {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      },
    );
  }

  /**
   * GET /api/v1/archives/{archiveId}/tokens?format=full|data&after=N&limit=K
   * Streams token collection using 4-byte length-prefixed binary frames.
   */
  public async handleStreamArchiveTokens(
    archiveId: string,
    req: Request,
    url: URL,
  ): Promise<Response> {
    const isBinaryRequested =
      req.headers.get("accept")?.includes("application/vnd.tinytoken.stream") ||
      url.searchParams.get("stream") === "binary";

    const format = url.searchParams.get("format") ?? "full";
    const isDataOnly = format === "data";

    // If client does not request binary streaming, return explanatory schema
    if (!isBinaryRequested) {
      const explanation = isDataOnly
        ? "FORMAT: format=data (Data-only stream: omits visual preview thumbnail bytes, reducing payload size by ~80% for high-throughput vector search, embeddings, metadata, and OCR analysis)."
        : "FORMAT: format=full (Complete 25 KB token envelopes including full visual preview thumbnail image bytes, DINOv3 vectors, VQ grid, OCR text, and cryptographic proofs).";

      return new Response(
        [
          `ENDPOINT: GET /api/v1/archives/${archiveId}/tokens?format=${format}`,
          "STATUS: Active",
          `DESCRIPTION: Streams all tokens in archive "${archiveId}" sequentially as 4-byte length-prefixed binary frames.`,
          explanation,
          "STREAMING PROTOCOL: Transfer-Encoding: chunked, Content-Type: application/vnd.tinytoken.stream.",
          "REQUEST HEADER: Pass 'Accept: application/vnd.tinytoken.stream' to initiate binary stream.",
        ].join("\n"),
        {
          headers: {
            "Content-Type": "text/plain; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
          },
        },
      );
    }

    if (!this.searchEngine) {
      return new Response("SearchEngine not initialized", { status: 503 });
    }

    const allEntries = this.searchEngine.getAllEntries();
    let startIndex = 0;

    // Resumable Cursor: `after` can be an integer index or a token ID
    const afterParam = url.searchParams.get("after");
    if (afterParam) {
      const numericAfter = parseInt(afterParam, 10);
      if (!isNaN(numericAfter) && numericAfter >= 0) {
        startIndex = numericAfter;
      } else {
        const foundIdx = allEntries.findIndex(
          (e) => e.id === afterParam || e.entryName === afterParam,
        );
        if (foundIdx !== -1) {
          startIndex = foundIdx + 1;
        }
      }
    }

    const limitParam = url.searchParams.get("limit");
    const isLimitZero = limitParam === "0";
    const limit = isLimitZero ? 0 : limitParam ? Math.max(1, parseInt(limitParam, 10)) : undefined;
    const remainingCount = Math.max(0, allEntries.length - startIndex);
    const targetEntries =
      limit === 0
        ? []
        : limit !== undefined
          ? allEntries.slice(startIndex, startIndex + limit)
          : allEntries.slice(startIndex);

    const searchEngine = this.searchEngine;

    // Bun Native ReadableStream producer
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        try {
          for (const entry of targetEntries) {
            const rawToken = await searchEngine.extractToken(entry.id);
            if (!rawToken || rawToken.byteLength < 704) continue;

            let payload = rawToken;
            if (isDataOnly) {
              const view = new DataView(rawToken.buffer, rawToken.byteOffset, rawToken.byteLength);
              const metaLength = view.getUint32(28, true); // Section II length
              const safeEnd = Math.min(rawToken.byteLength, 704 + metaLength);
              payload = rawToken.subarray(0, safeEnd);
            }

            // 4-byte length-prefixed binary frame (Uint32LE)
            const frame = createFramedToken(payload);
            controller.enqueue(frame);
          }
          controller.close();
        } catch (err) {
          controller.error(err);
        }
      },
    });

    return new Response(stream, {
      headers: {
        "Content-Type": "application/vnd.tinytoken.stream",
        "Transfer-Encoding": "chunked",
        "X-Archive-Id": archiveId,
        "X-Total-Tokens": String(limit === 0 ? remainingCount : targetEntries.length),
        "X-Archive-Capacity": String(allEntries.length),
        "X-Start-Index": String(startIndex),
        "Access-Control-Allow-Origin": "*",
      },
    });
  }

  /**
   * GET /api/v1/archives/{archiveId}/tokens/{tokenId}?format=full|data
   * Streams a single token using a 4-byte length-prefixed binary frame.
   */
  public async handleGetArchiveToken(
    archiveId: string,
    tokenId: string,
    req: Request,
    url: URL,
  ): Promise<Response> {
    const isBinaryRequested =
      req.headers.get("accept")?.includes("application/vnd.tinytoken.stream") ||
      url.searchParams.get("stream") === "binary";

    const format = url.searchParams.get("format") ?? "full";
    const isDataOnly = format === "data";

    if (!isBinaryRequested) {
      const explanation = isDataOnly
        ? "FORMAT: format=data (Data-only: metadata, OCR text, 384-dim DINOv3 vector, and 16x16 VQ lattice without visual preview thumbnail section)."
        : "FORMAT: format=full (Complete token envelope: includes visual preview thumbnail image section, metadata, OCR, DINOv3, and VQ lattice).";

      return new Response(
        [
          `ENDPOINT: GET /api/v1/archives/${archiveId}/tokens/${tokenId}?format=${format}`,
          "STATUS: Active",
          `DESCRIPTION: Fetches and streams token "${tokenId}" from archive "${archiveId}".`,
          explanation,
          "REQUEST HEADER: Pass 'Accept: application/vnd.tinytoken.stream' for binary frame output.",
        ].join("\n"),
        {
          headers: {
            "Content-Type": "text/plain; charset=utf-8",
            "Access-Control-Allow-Origin": "*",
          },
        },
      );
    }

    if (!this.searchEngine) {
      return new Response("SearchEngine not initialized", { status: 503 });
    }

    const rawToken = await this.searchEngine.extractToken(tokenId);
    if (!rawToken || rawToken.byteLength < 704) {
      return new Response(`Token not found: ${tokenId}`, { status: 404 });
    }

    let payload = rawToken;
    if (isDataOnly) {
      const view = new DataView(rawToken.buffer, rawToken.byteOffset, rawToken.byteLength);
      const metaLength = view.getUint32(28, true);
      const safeEnd = Math.min(rawToken.byteLength, 704 + metaLength);
      payload = rawToken.subarray(0, safeEnd);
    }

    const frame = createFramedToken(payload);
    return new Response(frame, {
      headers: {
        "Content-Type": "application/vnd.tinytoken.stream",
        "X-Token-Id": tokenId,
        "X-Payload-Size": String(payload.byteLength),
        "Access-Control-Allow-Origin": "*",
      },
    });
  }

  /**
   * Route dispatcher for /api/v1/archives/*
   */
  public async handleRoute(path: string, req: Request, url: URL): Promise<Response | null> {
    if (!path.startsWith("/api/v1/archives")) return null;
    const sub = path.slice("/api/v1/archives".length).replace(/^\/+/, "");

    // GET /api/v1/archives
    if (!sub) {
      return this.handleListArchives();
    }

    const parts = sub.split("/").map(decodeURIComponent);
    const archiveId = parts[0];

    // GET /api/v1/archives/{archiveId}/manifest
    if (parts.length === 2 && parts[1] === "manifest") {
      return this.handleGetArchiveManifest(archiveId!);
    }

    // GET /api/v1/archives/{archiveId}/tokens
    // GET /api/v1/archives/{archiveId}/tokens/{tokenId}
    if (parts[1] === "tokens") {
      if (parts.length === 2) {
        return await this.handleStreamArchiveTokens(archiveId!, req, url);
      }
      if (parts.length === 3) {
        return await this.handleGetArchiveToken(archiveId!, parts[2]!, req, url);
      }
    }

    return new Response(`Not Found: ${path}`, {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }
}

export const streamingArchiveHandlers = new StreamingArchiveHandlers();
