/**
 * Streaming Archive API Endpoints (v1) — handlers/streamingArchiveHandlers.ts
 *
 * Scaffolds streaming endpoints for high-throughput token ingestion & streaming:
 * - GET /api/v1/archives
 * - GET /api/v1/archives/{archiveId}/tokens?format=full
 * - GET /api/v1/archives/{archiveId}/tokens?format=data
 * - GET /api/v1/archives/{archiveId}/tokens/{tokenId}?format=full
 * - GET /api/v1/archives/{archiveId}/manifest
 */

export class StreamingArchiveHandlers {
  /**
   * GET /api/v1/archives
   * Lists or streams catalog of available token archives.
   */
  public handleListArchives(): Response {
    return new Response(
      [
        "ENDPOINT: GET /api/v1/archives",
        "STATUS: Planned / Scaffolded (Dummy)",
        "DESCRIPTION: Streams or lists catalog of all available archives (active dataset + frozen archives in ./data/archive).",
        "FIELDS RETURNED: archiveId, filename, label, tokenCount, sizeBytes, sha256Hex, generation, createdAt.",
        "FORMAT: JSON / NDJSON stream.",
      ].join("\n"),
      {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      }
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
        "STATUS: Planned / Scaffolded (Dummy)",
        `DESCRIPTION: Structural manifest and cryptographic integrity attestation for archive "${archiveId}".`,
        "FIELDS RETURNED: archiveId, totalTokens, archiveCrc32, merkleRootSha256, files: Array<{ filename, sizeBytes, crc32, offset }>, ed25519AttestationStatus.",
        "FORMAT: JSON manifest document.",
      ].join("\n"),
      {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }

  /**
   * GET /api/v1/archives/{archiveId}/tokens?format=full|data
   * Streams token collection from specified archive.
   */
  public handleStreamArchiveTokens(archiveId: string, url: URL): Response {
    const format = url.searchParams.get("format") ?? "full";
    const isDataOnly = format === "data";

    const explanation = isDataOnly
      ? "FORMAT: format=data (Data-only stream: omits visual preview thumbnail bytes/data URI, reducing payload size by ~80% for high-throughput vector search, embeddings, metadata, and OCR analysis)."
      : "FORMAT: format=full (Complete 25 KB token envelopes including full visual preview thumbnail image bytes/data URI, DINOv3 vectors, VQ grid, OCR text, and cryptographic proofs).";

    return new Response(
      [
        `ENDPOINT: GET /api/v1/archives/${archiveId}/tokens?format=${format}`,
        "STATUS: Planned / Scaffolded (Dummy)",
        `DESCRIPTION: Streams all tokens in archive "${archiveId}" sequentially as an NDJSON / chunked stream.`,
        explanation,
        "STREAMING PROTOCOL: Transfer-Encoding: chunked, Content-Type: application/x-ndjson.",
      ].join("\n"),
      {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }

  /**
   * GET /api/v1/archives/{archiveId}/tokens/{tokenId}?format=full|data
   * Streams a single token from specified archive.
   */
  public handleGetArchiveToken(archiveId: string, tokenId: string, url: URL): Response {
    const format = url.searchParams.get("format") ?? "full";
    const isDataOnly = format === "data";

    const explanation = isDataOnly
      ? "FORMAT: format=data (Data-only: metadata, OCR text, 384-dim DINOv3 vector, and 16x16 VQ lattice without visual preview thumbnail section)."
      : "FORMAT: format=full (Complete token envelope: includes visual preview thumbnail image section, metadata, OCR, DINOv3, and VQ lattice).";

    return new Response(
      [
        `ENDPOINT: GET /api/v1/archives/${archiveId}/tokens/${tokenId}?format=${format}`,
        "STATUS: Planned / Scaffolded (Dummy)",
        `DESCRIPTION: Fetches and streams token "${tokenId}" from archive "${archiveId}".`,
        explanation,
        "CONTENT-TYPE: application/json or binary token chunk.",
      ].join("\n"),
      {
        headers: {
          "Content-Type": "text/plain; charset=utf-8",
          "Access-Control-Allow-Origin": "*",
        },
      }
    );
  }

  /**
   * Route dispatcher for /api/v1/archives/*
   */
  public handleRoute(path: string, req: Request, url: URL): Response | null {
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
        return this.handleStreamArchiveTokens(archiveId!, url);
      }
      if (parts.length === 3) {
        return this.handleGetArchiveToken(archiveId!, parts[2]!, url);
      }
    }

    return new Response(`Not Found: ${path}`, { status: 404, headers: { "Content-Type": "text/plain; charset=utf-8" } });
  }
}

export const streamingArchiveHandlers = new StreamingArchiveHandlers();
