/**
 * 25 KB Binary Token Packer & Unpacker (utils/tokenCodec.ts)
 * Implements the 3-section GF28 V2 specification:
 * - Section I   (0..1023,   1,024 B): Header + 256B SNARK/SHARK Stub + 256B VQ Grid + 384B GF(2^8) RS Parity
 * - Section II  (1024..1024+metaLen, <= 4,096 B): QuickLZ/UTF-8 JSON Metadata & OCR Text
 * - Section III (1024+metaLen..,     <= 20,480 B): Embedded 220x220 Preview Image
 */

import {
  EmbeddingModelId,
  MAX_TOKEN_BYTES,
  PreviewFormat,
  RS_PARITY_BYTES,
  RS_PARITY_OFFSET,
  SECTION1_HEADER_BYTES,
  SECTION2_MAX_META_BYTES,
  SECTION3_MAX_PREVIEW_BYTES,
  SNARK_STUB_BYTES,
  SNARK_STUB_OFFSET,
  TOKEN_MAGIC,
  TOKEN_VERSION,
  TokenFlags,
  VQ_GRID_BYTES,
  VQ_GRID_OFFSET,
  type DecodedToken,
  type TokenMetadataPayload,
} from "../models/token";
import { encodeHeaderParity384, verifyAndHealHeader640 } from "./galois";
import { QuickLZ } from "./quicklz";
import {
  bytesToHex,
  generateSnarkAttestationStub,
  verifySnarkAttestationStub,
  type AuthorityAttestationKeys,
} from "./snarkAttest";

export interface EncodeTokenInput {
  uuidBytes?: Uint8Array;
  originalFileSize: number;
  width: number;
  height: number;
  page: number;
  previewFormat: PreviewFormat;
  shardBlockId?: number;
  shardIndex?: number;
  embeddingModelId?: EmbeddingModelId;
  vqIndices: Uint8Array; // 256 bytes
  metadata: TokenMetadataPayload;
  previewBytes: Uint8Array;
  compressMetadata?: boolean;
  signingKeys?: AuthorityAttestationKeys;
}

export interface DecodeTokenOptions {
  trustedPublicKeyRaw?: Uint8Array;
}

function uuidStringToBytes(uuid: string): Uint8Array {
  const hex = uuid.replace(/-/g, "").padEnd(32, "0").slice(0, 32);
  const out = new Uint8Array(16);
  for (let i = 0; i < 16; i++) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16) || 0;
  }
  return out;
}

/**
 * Encodes and Ed25519-signs a complete binary token guaranteed to fit within MAX_TOKEN_BYTES (25,600 bytes).
 */
export async function encodeBinaryToken(
  input: EncodeTokenInput,
): Promise<Uint8Array<ArrayBuffer>> {
  const uuidBytes =
    input.uuidBytes && input.uuidBytes.length === 16
      ? input.uuidBytes
      : uuidStringToBytes(crypto.randomUUID());

  const vqIndices = new Uint8Array(VQ_GRID_BYTES);
  vqIndices.set(input.vqIndices.subarray(0, VQ_GRID_BYTES));

  // Serialize Section II Metadata
  const rawMetaBytes = new TextEncoder().encode(JSON.stringify(input.metadata));
  let metaBytes: Uint8Array = rawMetaBytes;
  let flags = TokenFlags.HAS_VQ_GRID | TokenFlags.SNARK_ATTESTED;
  if (input.metadata.totalPages > 1) {
    flags |= TokenFlags.MULTI_PAGE_DOC;
  }
  if (input.metadata.dinoEmbeddingHex) {
    flags |= TokenFlags.HAS_DINO_EMBEDDING;
  }

  if (input.compressMetadata !== false && rawMetaBytes.length > 128) {
    const compressed = QuickLZ.compress(rawMetaBytes);
    if (compressed.length < rawMetaBytes.length) {
      metaBytes = compressed;
      flags |= TokenFlags.QUICKLZ_COMPRESSED;
    }
  }

  if (metaBytes.length > SECTION2_MAX_META_BYTES) {
    // Truncate OCR text gracefully if metadata exceeds 4 KB allocation
    const truncatedMeta: TokenMetadataPayload = {
      ...input.metadata,
      ocrText: input.metadata.ocrText.slice(0, 2400),
    };
    metaBytes = new TextEncoder().encode(JSON.stringify(truncatedMeta)).subarray(0, SECTION2_MAX_META_BYTES);
    flags &= ~TokenFlags.QUICKLZ_COMPRESSED;
  }

  // Cap Section III Preview Image so total token never exceeds 25 KB (25,600 bytes)
  const maxAllowedPreview = Math.min(
    SECTION3_MAX_PREVIEW_BYTES,
    MAX_TOKEN_BYTES - SECTION1_HEADER_BYTES - metaBytes.length,
  );
  const previewBytes = input.previewBytes.subarray(0, maxAllowedPreview);

  const resolvedEmbeddingModelId =
    input.embeddingModelId ??
    (input.metadata.dinoEmbeddingHex ? EmbeddingModelId.DINOV2_VIT_S14_Q8 : EmbeddingModelId.NONE);

  // Generate 256B Ed25519 + 4-Leaf SHA-256 Merkle Provenance Attestation & 32B Merkle Root
  const { snarkStub, merkleRoot } = await generateSnarkAttestationStub({
    uuidBytes,
    originalFileSize: input.originalFileSize,
    width: input.width,
    height: input.height,
    page: input.page,
    previewFormat: input.previewFormat,
    embeddingModelId: resolvedEmbeddingModelId,
    vqIndices,
    metaBytes,
    previewBytes,
    signingKeys: input.signingKeys,
  });

  const totalTokenSize = SECTION1_HEADER_BYTES + metaBytes.length + previewBytes.length;
  const tokenBuffer = new Uint8Array(totalTokenSize);
  const view = new DataView(tokenBuffer.buffer);

  // --- Section I: 1,024-Byte Fixed Header ---
  // 0..3: Magic "GF28"
  tokenBuffer[0] = 0x47; // 'G'
  tokenBuffer[1] = 0x46; // 'F'
  tokenBuffer[2] = 0x32; // '2'
  tokenBuffer[3] = 0x38; // '8'

  // 4..5: Version (Uint16LE)
  view.setUint16(4, TOKEN_VERSION, true);
  // 6..7: Flags (Uint16LE)
  view.setUint16(6, flags, true);
  // 8..23: 16-byte UUID
  tokenBuffer.set(uuidBytes, 8);
  // 24..27: Original source file size
  view.setUint32(24, input.originalFileSize, true);
  // 28..31: Section II Metadata byte length
  view.setUint32(28, metaBytes.length, true);
  // 32..35: Section III Preview byte length
  view.setUint32(32, previewBytes.length, true);
  // 36..37: Original width
  view.setUint16(36, input.width, true);
  // 38..39: Original height
  view.setUint16(38, input.height, true);
  // 40..41: Page number
  view.setUint16(40, input.page, true);
  // 42..43: Preview format enum
  view.setUint16(42, input.previewFormat, true);
  // 44..75: 32-byte SHA-256 Merkle Root
  tokenBuffer.set(merkleRoot, 44);
  // 76..79: Hierarchical RS Shard Block ID
  view.setUint32(76, input.shardBlockId ?? 0, true);
  // 80..81: Shard Index within block (0..63)
  view.setUint16(80, input.shardIndex ?? 0, true);
  // 82..87: 48-bit Timestamp ms (Uint32LE low at 82, Uint16LE high at 86)
  const now = Date.now();
  view.setUint32(82, now >>> 0, true);
  view.setUint16(86, Math.floor(now / 0x100000000) & 0xffff, true);
  // 88: Homomorphic GF(2^8) state tag byte
  tokenBuffer[88] = 0x01;
  // 89: Embedding Model ID byte (0x01 = Color/HOG fallback, 0x02 = DINOv2 ViT-S/14 ONNX q8)
  tokenBuffer[89] = resolvedEmbeddingModelId;

  // 128..383: 256-Byte Ed25519 + SHA-256 Merkle Provenance Attestation Block
  tokenBuffer.set(snarkStub, SNARK_STUB_OFFSET);

  // 384..639: 256-Byte VQ 16x16 Latent Codebook Grid
  tokenBuffer.set(vqIndices, VQ_GRID_OFFSET);

  // 640..1023: 384-Byte GF(2^8) Reed-Solomon Parity over bytes 0..639
  const rsParity = encodeHeaderParity384(tokenBuffer.subarray(0, RS_PARITY_OFFSET));
  tokenBuffer.set(rsParity, RS_PARITY_OFFSET);

  // --- Section II: Metadata & OCR Layer ---
  tokenBuffer.set(metaBytes, SECTION1_HEADER_BYTES);

  // --- Section III: Preview Image ---
  tokenBuffer.set(previewBytes, SECTION1_HEADER_BYTES + metaBytes.length);

  return tokenBuffer;
}

/**
 * Unpacks, heals (via GF(2^8) Reed-Solomon), and cryptographically verifies (via Ed25519 + full-token
 * SHA-256 Merkle commitments over Origin Header, VQ Grid, Section II, and Section III) a 25 KB binary token.
 */
export async function decodeBinaryToken(
  rawToken: Uint8Array,
  options?: DecodeTokenOptions,
): Promise<DecodedToken> {
  if (rawToken.length < SECTION1_HEADER_BYTES) {
    throw new Error(
      `Invalid token buffer: expected at least ${SECTION1_HEADER_BYTES} bytes, got ${rawToken.length}`,
    );
  }

  // 1. Verify & heal Section I Header (0..639) using 384-byte GF(2^8) Reed-Solomon parity (640..1023)
  const headerPayload640 = rawToken.subarray(0, RS_PARITY_OFFSET);
  const parity384 = rawToken.subarray(RS_PARITY_OFFSET, SECTION1_HEADER_BYTES);
  const rsResult = verifyAndHealHeader640(headerPayload640, parity384);

  const healedHeader = rsResult.healedPayload;
  const view = new DataView(healedHeader.buffer, healedHeader.byteOffset, healedHeader.byteLength);

  const magic = String.fromCharCode(
    healedHeader[0] ?? 0,
    healedHeader[1] ?? 0,
    healedHeader[2] ?? 0,
    healedHeader[3] ?? 0,
  );
  if (magic !== TOKEN_MAGIC) {
    throw new Error(`Invalid token magic: expected '${TOKEN_MAGIC}', found '${magic}'`);
  }

  const version = view.getUint16(4, true);
  const flags = view.getUint16(6, true);
  const uuidBytes = healedHeader.subarray(8, 24);
  const uuidHex = bytesToHex(uuidBytes);
  const originalFileSize = view.getUint32(24, true);
  const metaLength = view.getUint32(28, true);
  const previewLength = view.getUint32(32, true);
  const width = view.getUint16(36, true);
  const height = view.getUint16(38, true);
  const page = view.getUint16(40, true);
  const previewFormat = view.getUint16(42, true) as PreviewFormat;
  const merkleRoot = healedHeader.subarray(44, 76);
  const merkleRootHex = bytesToHex(merkleRoot);
  const shardBlockId = view.getUint32(76, true);
  const shardIndex = view.getUint16(80, true);
  const tsLow = view.getUint32(82, true);
  const tsHigh = view.getUint16(86, true);
  const timestampMs = tsHigh * 0x100000000 + tsLow;
  const homomorphicTag = healedHeader[88] ?? 0;
  const embeddingModelId = (healedHeader[89] ?? 0) as EmbeddingModelId;

  const snarkStub = new Uint8Array(
    healedHeader.subarray(SNARK_STUB_OFFSET, SNARK_STUB_OFFSET + SNARK_STUB_BYTES),
  );
  const vqIndices = new Uint8Array(
    healedHeader.subarray(VQ_GRID_OFFSET, VQ_GRID_OFFSET + VQ_GRID_BYTES),
  );

  // 2. Slice Section II (Metadata & OCR) and Section III (Preview) BEFORE cryptographic verification
  const expectedTotalBytes = SECTION1_HEADER_BYTES + metaLength + previewLength;
  const envelopeBoundsValid = expectedTotalBytes === rawToken.length;

  const metaStart = SECTION1_HEADER_BYTES;
  const metaEnd = Math.min(rawToken.length, metaStart + metaLength);
  const rawMetaSlice = rawToken.subarray(metaStart, metaEnd);

  const previewStart = metaEnd;
  const previewEnd = Math.min(rawToken.length, previewStart + previewLength);
  const previewBytes = rawToken.subarray(previewStart, previewEnd);

  // 3. Cryptographically verify Ed25519 signature + SHA-256 commitments over ALL sections
  const snarkAttestation = await verifySnarkAttestationStub({
    snarkStub,
    merkleRoot,
    uuidBytes,
    originalFileSize,
    width,
    height,
    page,
    previewFormat,
    embeddingModelId,
    vqIndices,
    metaBytes: rawMetaSlice,
    previewBytes,
    envelopeBoundsValid,
    trustedPublicKeyRaw: options?.trustedPublicKeyRaw,
  });

  // 4. Unpack Section II Metadata & OCR (guarded if Section II was tampered/corrupted)
  let metadata: TokenMetadataPayload;
  try {
    let decodedMetaBytes = rawMetaSlice;
    if ((flags & TokenFlags.QUICKLZ_COMPRESSED) !== 0 && rawMetaSlice.length >= 9) {
      const claimedDecompressedLen =
        ((rawMetaSlice[1] ?? 0) |
          ((rawMetaSlice[2] ?? 0) << 8) |
          ((rawMetaSlice[3] ?? 0) << 16) |
          ((rawMetaSlice[4] ?? 0) << 24)) >>>
        0;
      if (claimedDecompressedLen > SECTION2_MAX_META_BYTES * 8) {
        throw new Error("Section II QuickLZ decompressed size exceeds safety limit");
      }
      decodedMetaBytes = QuickLZ.decompress(rawMetaSlice);
    }
    metadata = JSON.parse(new TextDecoder().decode(decodedMetaBytes)) as TokenMetadataPayload;
  } catch {
    metadata = {
      id: `corrupted_${uuidHex.slice(0, 12)}`,
      filename: "corrupted_metadata",
      sourcePath: "",
      mimeType: "application/octet-stream",
      page,
      totalPages: 1,
      width,
      height,
      colorDepth: 24,
      createdAt: new Date(0).toISOString(),
      ocrText: "[CORRUPTED OR TAMPERED SECTION II METADATA]",
    };
  }

  return {
    id: metadata.id,
    magic,
    version,
    flags,
    uuidHex,
    originalFileSize,
    metaLength,
    previewLength,
    width,
    height,
    page,
    previewFormat,
    merkleRootHex,
    shardBlockId,
    shardIndex,
    timestampMs,
    homomorphicTag,
    embeddingModelId,
    snarkStub,
    snarkAttestation,
    vqIndices,
    rsParityValid: rsResult.valid,
    rsCorrectedErrors: rsResult.correctedErrors,
    metadata,
    previewBytes,
    byteMap: {
      totalTokenBytes: rawToken.length,
      section1HeaderBytes: SECTION1_HEADER_BYTES,
      snarkStubBytes: SNARK_STUB_BYTES,
      vqGridBytes: VQ_GRID_BYTES,
      rsParityBytes: RS_PARITY_BYTES,
      section2MetaBytes: metaLength,
      section3PreviewBytes: previewBytes.length,
    },
  };
}

/**
 * Automated Pipeline Synthesis Tool:
 * Generates a spec-compliant, Ed25519-signed 25 KB (`25,600` bytes max) `.token` buffer
 * with real GF(2^8) Reed-Solomon parity, 256B Ed25519+Merkle provenance attestation, 256B VQ 16x16 grid,
 * and Canberra locale/geo metadata for boundary and pipeline testing.
 */
export async function generateMockTinyTokenBuffer(
  tokenId = "testimg000000000",
  environmentClass = "grassland_synthetic",
): Promise<Uint8Array<ArrayBuffer>> {
  const vqIndices = new Uint8Array(VQ_GRID_BYTES);
  for (let i = 0; i < VQ_GRID_BYTES; i++) {
    vqIndices[i] = (i * 3 + 7) & 0xff;
  }

  // Simulate a 2,000-byte JPEG preview frame (SOI 0xFFD8 ... EOI 0xFFD9)
  const mockPreview = new Uint8Array(2000);
  mockPreview[0] = 0xff;
  mockPreview[1] = 0xd8;
  for (let i = 2; i < mockPreview.length - 2; i++) {
    mockPreview[i] = (i * 19 + 0x5a) & 0xff;
  }
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
    geoCoordinates: {
      latitude: -35.1764,
      longitude: 149.1311,
    },
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


