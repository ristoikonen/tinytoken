# TinyToken Project Architecture & Code Review Analysis (Remediated V2)
**Date:** October 30, 2026  
**Project Context:** Oksa Visual Containers / TinyToken Ecosystem  
**Runtime Environment:** Bun + TypeScript (`GF(2^8)` Reed-Solomon + `16×16` VQ Codebook + `256B` SHARK/SNARK)

---

## 1. Storage Stratification & Architecture Blueprint

When managing large sets of discrete vector quantization (VQ) codebook indices (such as a 500 MB ZIP archive of 21,000 tokens), traditional float vector databases (e.g., Pinecone, Qdrant) or heavy self-hosted object storage instances present severe performance and operational bottlenecks:

1. **Object Storage Limitations:** Key-Value stores are built for raw payload retention (`GET /bucket/key`). Running nearest-neighbor, subset intersection, or keyword searches within files requires dragging binary blobs over HTTP to the local application layer.
2. **Stratified Hybrid Pipeline:** Keep the cold/warm random-access 500 MB `.zip` container on local NVMe or **Cloudflare R2** (10 GB zero-egress free tier, sliced via HTTP `Range` or `Bun.file().slice()`), while indexing the discrete `0..255` VQ codebook arrays in:
   - **Hot Tier (In-Memory ADC Lookup):** `5.37 MB` contiguous typed arrays for `< 0.5 ms` Asymmetric Distance Computation (ADC).
   - **Relational Inverted Index Tier (Turso / LibSQL):** Maps discrete codebook centroid IDs (`0..255`) to `image_id` and `frequency` for SQL set-intersection and motif queries.

```
[ 500MB ZIP Core Archive ] ──> Local NVMe (`Bun.file().slice()`) or Cloudflare R2 Range Slice
          │
          ▼ (Extracted via Native Bun Central Directory + Manifest Pipeline)
[ Inverted Index DB Tier ] ──> Hosted on Turso / LibSQL (`@libsql/client`)
          ├──> Table 1: `images` (Maps token IDs to paths and inferred labels)
          └──> Table 2: `image_tokens` (Maps Codebook Centroid IDs 0..255 to image keys & frequencies)
```

---

## 2. Remediated Inverted Index Engine Implementation (`utils/vqInvertedIndex.ts`)

Because VQ codebook indices are discrete categorical values (`0..255` in $\mathbb{F}_{256}$), finding semantic/structural matches (such as *"images sharing grassland centroid patterns"*) becomes a fast relational set-intersection query.

### Key Fixes Applied in V2
1. **Parameterized SQL Placeholders (`?`):** Eliminates string-interpolation SQL injection in `WHERE it.token_id IN (...)`.
2. **Centroid Deduplication Before `minMatches`:** A `16×16` VQ grid has `256` cells with repeated centroid IDs. Deduplicating `uniqueTokens` ensures `HAVING overlapping_tokens >= ?` never compares distinct SQL row counts against an un-deduplicated 256-item length.
3. **Histogram Frequency Ranking (`SUM(it.frequency)`):** Uses the `frequency` column to rank ties by total matching spatial cells.
4. **Native `Bun.env` Configuration:** Uses `Bun.env.TURSO_DATABASE_URL` with an automatic fallback to `file:./data/vq_index.db`.

### Database Schema (SQLite / LibSQL compatible)
```sql
CREATE TABLE IF NOT EXISTS images (
    id TEXT PRIMARY KEY,
    r2_url TEXT NOT NULL,
    inferred_label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS image_tokens (
    token_id INTEGER NOT NULL,
    image_id TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    PRIMARY KEY (token_id, image_id),
    FOREIGN KEY (image_id) REFERENCES images(id)
);

CREATE INDEX IF NOT EXISTS idx_token ON image_tokens(token_id);
```

### High-Speed Index Search Component
```typescript
import { createClient } from "@libsql/client";

const db = createClient({
  url: Bun.env.TURSO_DATABASE_URL || "file:./data/vq_index.db",
  authToken: Bun.env.TURSO_AUTH_TOKEN,
});

export async function findImagesByTokenPattern(
  queryTokens: number[],
  threshold = 0.5,
  limit = 50,
) {
  const uniqueTokens = Array.from(
    new Set(
      queryTokens
        .filter((n) => Number.isFinite(n))
        .map((n) => Math.max(0, Math.min(255, Math.floor(n)))),
    ),
  );
  if (uniqueTokens.length === 0) return [];

  const minMatches = Math.max(1, Math.ceil(uniqueTokens.length * threshold));
  const placeholders = uniqueTokens.map(() => "?").join(", ");

  const query = `
    SELECT 
        i.id AS id, 
        i.r2_url AS r2_url, 
        i.inferred_label AS inferred_label,
        COUNT(it.token_id) AS overlapping_tokens,
        SUM(it.frequency) AS weighted_frequency
    FROM image_tokens it
    JOIN images i ON it.image_id = i.id
    WHERE it.token_id IN (${placeholders})
    GROUP BY i.id
    HAVING overlapping_tokens >= ?
    ORDER BY overlapping_tokens DESC, weighted_frequency DESC
    LIMIT ?;
  `;

  const rs = await db.execute({
    sql: query,
    args: [...uniqueTokens, minMatches, limit],
  });

  return rs.rows;
}
```

---

## 3. Remediated `TokenApiHandlers` Controller (`handlers/tokenHandlers.ts`)

### Key Fixes Applied in V2
1. **3-Argument `applyHomomorphicMaskGF28` Call:** Passes `(rawToken, scalar, TokenFlags.HOMOMORPHIC_MASKED)` and persists the mutated slice back to the archive via `await this.engine.writeMutatedTokenInPlace(id, mutated)`.
2. **Graceful Default Body Handling on `POST /api/token/:id/mutate`:** Empty POST bodies default cleanly to scalar `0x5a` instead of throwing `400`.
3. **Preserved `handleRebuildIndex` & Added `handlePatternSearch`:** Supports live archive rebuilds (`POST /api/index/rebuild`) and SQL inverted-index pattern queries (`GET /api/search/pattern`).

---

## 4. Remediated `decodeBinaryToken` Parser Component (`utils/tokenCodec.ts`)

### Key Fixes Applied in V2
1. **Fixed 1 KB Section I Header (`0..1023`) Ahead of Variable Payloads:** Keeps `Header (0..127) + SHARK Stub (128..383) + VQ 16×16 Grid (384..639) + GF(2^8) RS Parity (640..1023)` at fixed offsets so the entire header and VQ grid can be verified in a single 1 KB slice before touching variable-length metadata (`Section II`) or preview bytes (`Section III`).
2. **Real Algebraic $GF(2^8)$ Reed-Solomon & SHARK Verification:** Runs `verifyAndHealHeader640()` across all 3 header lanes and `verifySnarkAttestationStub()` over the Merkle root and $GF(2^8)$ polynomial witness tags instead of reading an unverified boolean flag byte.
3. **Exact `metaLength` Slicing (No Trailing NUL Byte Crashes):** Slices `rawToken.subarray(metaStart, metaStart + metaLength)` and decompresses via `QuickLZ.decompress()` when `TokenFlags.QUICKLZ_COMPRESSED` is set.
4. **Strict `noUncheckedIndexedAccess` Safety:** All byte accesses guard against `undefined` (`?? 0`) and use consistent Little-Endian (`true`) `DataView` reads.

---

## 5. Remediated Automated Pipeline Synthesis Tool (`generateMockTinyTokenBuffer`)

Generates a spec-compliant, cryptographically attested 25 KB `.token` buffer (`GF28` v2 magic, real 384B Reed-Solomon parity, real 256B SHARK/SNARK witness, 256B VQ grid, and Canberra locale metadata):

```typescript
import { PreviewFormat, VQ_GRID_BYTES, type TokenMetadataPayload } from "../models/token";
import { encodeBinaryToken } from "./tokenCodec";

export function generateMockTinyTokenBuffer(
  tokenId = "testimg000000000",
  environmentClass = "grassland_synthetic",
): Uint8Array<ArrayBuffer> {
  const vqIndices = new Uint8Array(VQ_GRID_BYTES);
  for (let i = 0; i < VQ_GRID_BYTES; i++) {
    vqIndices[i] = (i * 3 + 7) & 0xff;
  }

  const mockPreview = new Uint8Array(2000);
  mockPreview[0] = 0xff;
  mockPreview[1] = 0xd8;
  mockPreview[mockPreview.length - 2] = 0xff;
  mockPreview[mockPreview.length - 1] = 0xd9;

  const metadata: TokenMetadataPayload = {
    id: tokenId,
    filename: `${tokenId}.jpg`,
    sourcePath: `/api/token/${encodeURIComponent(tokenId)}/preview`,
    mimeType: "image/jpeg",
    page: 1,
    totalPages: 1,
    width: 1024,
    height: 768,
    colorDepth: 24,
    createdAt: new Date(1793345600 * 1000).toISOString(),
    ocrText: `Automated Pipeline Asset Synthesis Engine • Class: ${environmentClass} • Canberra Locale (-35.1764, 149.1311)`,
    tags: ["synthetic", environmentClass, "canberra"],
    environmentClass,
    geoCoordinates: { latitude: -35.1764, longitude: 149.1311 },
  };

  return encodeBinaryToken({
    originalFileSize: 1024 * 768 * 3,
    width: 1024,
    height: 768,
    page: 1,
    previewFormat: PreviewFormat.JPEG,
    shardBlockId: 999,
    shardIndex: 4,
    vqIndices,
    metadata,
    previewBytes: mockPreview,
  });
}
```

