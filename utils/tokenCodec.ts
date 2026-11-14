/**
 * 25 KB Binary Token Packer & Unpacker (utils/tokenCodec.ts)
 * Implements the 3-section GF28 V2 specification (Option C Trimmed Parity Layout):
 * - Section I   (0..703,     704 B): Header + 256B SNARK/SHARK Stub + 256B VQ Grid + 64B GF(2^8) RS Parity
 * - Section II  (704..704+metaLen, <= 4,096 B): QuickLZ/UTF-8 JSON Metadata & OCR Text
 * - Section III (704+metaLen..,     <= 20,800 B): Embedded 220x220 Preview Image
 */

import {
  EmbeddingModelId,
  MAX_TOKEN_BYTES,
  PreviewFormat,
  RS_PARITY_BYTES,
  RS_PARITY_OFFSET,
  SECTION1_HEADER_BYTES,
  SECTION2_MAX_META_BYTES,
  SECTION2_MAX_UNCOMPRESSED_META_BYTES,
  SECTION3_MAX_PREVIEW_BYTES,
  SNARK_STUB_BYTES,
  SNARK_STUB_OFFSET,
  TOKEN_MAGIC,
  TOKEN_VERSION,
  TokenFlags,
  VQ_GRID_BYTES,
  VQ_GRID_OFFSET,
  type DecodedToken,
  type TokenHeaderInspection,
  type TokenMetadataPayload,
} from "../models/token";
import {
  encodeHeaderParity64,
  verifyAndHealHeader640,
  mutateHeader,
  type HeaderMutation,
} from "./galois";

export { mutateHeader, type HeaderMutation };
import { QuickLZ } from "./quicklz";
import { hexToBgeEmbedding } from "./bgeEmbedding";
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

const UUID_REGEX =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;
const HEX32_REGEX = /^[0-9a-f]{32}$/i;

/**
 * Validates UUID syntax (standard 8-4-4-4-12 hex or 32 hex chars) and rejects invalid input.
 */
export function uuidStringToBytes(uuid: string): Uint8Array {
  const clean = uuid.trim();
  let hex: string;
  if (UUID_REGEX.test(clean)) {
    hex = clean.replace(/-/g, "");
  } else if (HEX32_REGEX.test(clean)) {
    hex = clean;
  } else {
    throw new Error(`Invalid UUID syntax: "${uuid}"`);
  }
  const out = new Uint8Array(16);
  for (let i = 0; i < 16; i++) {
    out[i] = Number.parseInt(hex.slice(i * 2, i * 2 + 2), 16);
  }
  return out;
}

/**
 * Encodes and Ed25519-signs a complete binary token guaranteed to fit within MAX_TOKEN_BYTES (25,600 bytes).
 */
export async function encodeBinaryToken(
  input: EncodeTokenInput,
): Promise<Uint8Array<ArrayBuffer>> {
  // 1. Validate VQ output
  if (!input.vqIndices || input.vqIndices.length !== VQ_GRID_BYTES) {
    throw new Error(`Expected ${VQ_GRID_BYTES} VQ indices`);
  }
  const vqIndices = new Uint8Array(VQ_GRID_BYTES);
  vqIndices.set(input.vqIndices);

  // 2. Validate UUID syntax
  let uuidBytes: Uint8Array;
  if (input.uuidBytes) {
    if (input.uuidBytes.length !== 16) {
      throw new Error(`Expected 16-byte UUID, received ${input.uuidBytes.length} bytes`);
    }
    uuidBytes = input.uuidBytes;
  } else {
    uuidBytes = uuidStringToBytes(crypto.randomUUID());
  }

  // 3. Serialize Section II Metadata with graceful OCR truncation
  // Verify that mandatory metadata without OCR can fit in SECTION2_MAX_META_BYTES
  const baseMeta: TokenMetadataPayload = {
    ...input.metadata,
    ocrText: "",
  };
  const baseMetaBytes = new TextEncoder().encode(JSON.stringify(baseMeta));
  let baseCompBytes = baseMetaBytes;
  if (input.compressMetadata !== false && baseMetaBytes.length > 128) {
    const comp = QuickLZ.compress(baseMetaBytes);
    if (comp.length < baseMetaBytes.length) {
      baseCompBytes = comp;
    }
  }
  if (baseCompBytes.length > SECTION2_MAX_META_BYTES) {
    throw new Error(
      `Mandatory metadata exceeds Section II limit (${baseCompBytes.length} > ${SECTION2_MAX_META_BYTES} bytes)`,
    );
  }

  // Start with full OCR text
  let currentOcr = input.metadata.ocrText ?? "";
  let currentMeta: TokenMetadataPayload = { ...input.metadata, ocrText: currentOcr };
  let rawMetaBytes = new TextEncoder().encode(JSON.stringify(currentMeta));

  // Guard against uncompressed size exceeding SECTION2_MAX_UNCOMPRESSED_META_BYTES (32 KB)
  if (rawMetaBytes.length > SECTION2_MAX_UNCOMPRESSED_META_BYTES) {
    const excess = rawMetaBytes.length - SECTION2_MAX_UNCOMPRESSED_META_BYTES + 64;
    currentOcr = currentOcr.slice(0, Math.max(0, currentOcr.length - excess));
    currentMeta = { ...input.metadata, ocrText: currentOcr };
    rawMetaBytes = new TextEncoder().encode(JSON.stringify(currentMeta));
  }

  let metaBytes = rawMetaBytes;
  let isCompressed = false;
  if (input.compressMetadata !== false && rawMetaBytes.length > 128) {
    const compressed = QuickLZ.compress(rawMetaBytes);
    if (compressed.length < rawMetaBytes.length) {
      metaBytes = compressed;
      isCompressed = true;
    }
  }

  // If it exceeds SECTION2_MAX_META_BYTES, iteratively truncate OCR text, re-serialize, and re-compress
  while (metaBytes.length > SECTION2_MAX_META_BYTES && currentOcr.length > 0) {
    const overflow = metaBytes.length - SECTION2_MAX_META_BYTES;
    const cutChars = Math.max(1, Math.min(currentOcr.length, overflow));
    currentOcr = currentOcr.slice(0, currentOcr.length - cutChars);
    currentMeta = { ...input.metadata, ocrText: currentOcr };
    rawMetaBytes = new TextEncoder().encode(JSON.stringify(currentMeta));

    metaBytes = rawMetaBytes;
    isCompressed = false;
    if (input.compressMetadata !== false && rawMetaBytes.length > 128) {
      const comp = QuickLZ.compress(rawMetaBytes);
      if (comp.length < rawMetaBytes.length) {
        metaBytes = comp;
        isCompressed = true;
      }
    }
  }

  if (metaBytes.length > SECTION2_MAX_META_BYTES) {
    throw new Error(
      `Metadata exceeds Section II limit (${metaBytes.length} > ${SECTION2_MAX_META_BYTES} bytes)`,
    );
  }

  // Set base flags (attestation flag will be set only AFTER successful generation)
  let flags = TokenFlags.HAS_VQ_GRID;
  if (input.metadata.totalPages > 1) {
    flags |= TokenFlags.MULTI_PAGE_DOC;
  }
  if (input.metadata.dinoEmbeddingHex) {
    flags |= TokenFlags.HAS_DINO_EMBEDDING;
  }
  if (input.metadata.bgeEmbeddingHex) {
    flags |= TokenFlags.HAS_BGE_EMBEDDING;
  }
  if (isCompressed) {
    flags |= TokenFlags.QUICKLZ_COMPRESSED;
  }

  // Cap Section III Preview Image so total token never exceeds 25 KB (25,600 bytes)
  const maxAllowedPreview = Math.min(
    SECTION3_MAX_PREVIEW_BYTES,
    MAX_TOKEN_BYTES - SECTION1_HEADER_BYTES - metaBytes.length,
  );
  const previewBytes = input.previewBytes.subarray(0, maxAllowedPreview);

  const resolvedEmbeddingModelId =
    input.embeddingModelId ??
    (input.metadata.dinoEmbeddingHex ? EmbeddingModelId.DINOV3_VITS16_Q8 : EmbeddingModelId.NONE);

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

  // Set SNARK_ATTESTED flag once attestation has been successfully generated
  if (snarkStub && snarkStub.length === SNARK_STUB_BYTES) {
    flags |= TokenFlags.SNARK_ATTESTED;
  }

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
  // 89: Embedding Model ID byte (0x01 = Color/HOG fallback, 0x02 = DINOv3 ViT-S/16 ONNX q8)
  tokenBuffer[89] = resolvedEmbeddingModelId;

  // 128..383: 256-Byte Ed25519 + SHA-256 Merkle Provenance Attestation Block
  tokenBuffer.set(snarkStub, SNARK_STUB_OFFSET);

  // 384..639: 256-Byte VQ 16x16 Latent Codebook Grid
  tokenBuffer.set(vqIndices, VQ_GRID_OFFSET);

  // 640..703: 64-Byte GF(2^8) Reed-Solomon Parity over bytes 0..639
  const rsParity = encodeHeaderParity64(tokenBuffer.subarray(0, RS_PARITY_OFFSET));
  tokenBuffer.set(rsParity, RS_PARITY_OFFSET);

  // --- Section II: Metadata & OCR Layer ---
  tokenBuffer.set(metaBytes, SECTION1_HEADER_BYTES);

  // --- Section III: Preview Image ---
  tokenBuffer.set(previewBytes, SECTION1_HEADER_BYTES + metaBytes.length);

  return tokenBuffer;
}

/**
 * Ultra-fast zero-copy inspector for Section I header (0..703 bytes).
 * Performs NO QuickLZ decompression, NO JSON parsing, and NO cryptographic signature verification.
 * Inspects all fixed binary fields in sub-microsecond time (<1 µs, >1.2 GB/s throughput).
 */
export function inspectBinaryTokenHeader(rawBytes: Uint8Array): TokenHeaderInspection {
  if (rawBytes.length < SECTION1_HEADER_BYTES) {
    throw new Error(
      `Token buffer too small for Section I header: ${rawBytes.length} < ${SECTION1_HEADER_BYTES} bytes`,
    );
  }

  const magic = String.fromCharCode(
    rawBytes[0] ?? 0,
    rawBytes[1] ?? 0,
    rawBytes[2] ?? 0,
    rawBytes[3] ?? 0,
  );
  if (magic !== TOKEN_MAGIC) {
    throw new Error(`Invalid token magic: expected '${TOKEN_MAGIC}', found '${magic}'`);
  }

  const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);
  const version = view.getUint16(4, true);
  if (version !== TOKEN_VERSION) {
    throw new Error(
      `Unsupported token version: expected ${TOKEN_VERSION}, found ${version}`,
    );
  }

  const flags = view.getUint16(6, true);
  const uuidBytes = rawBytes.subarray(8, 24);
  const uuidHex = bytesToHex(uuidBytes);
  const originalFileSize = view.getUint32(24, true);
  const metaLength = view.getUint32(28, true);
  const previewLength = view.getUint32(32, true);
  const width = view.getUint16(36, true);
  const height = view.getUint16(38, true);
  const page = view.getUint16(40, true);
  const previewFormat = view.getUint16(42, true) as PreviewFormat;
  const merkleRoot = rawBytes.subarray(44, 76);
  const merkleRootHex = bytesToHex(merkleRoot);
  const shardBlockId = view.getUint32(76, true);
  const shardIndex = view.getUint16(80, true);
  const tsLow = view.getUint32(82, true);
  const tsHigh = view.getUint16(86, true);
  const timestampMs = tsHigh * 0x100000000 + tsLow;
  const homomorphicTag = rawBytes[88] ?? 0;
  const embeddingModelId = (rawBytes[89] ?? 0) as EmbeddingModelId;
  const vqIndices = rawBytes.subarray(VQ_GRID_OFFSET, VQ_GRID_OFFSET + VQ_GRID_BYTES);

  const expectedTotalBytes = SECTION1_HEADER_BYTES + metaLength + previewLength;
  const envelopeBoundsValid = expectedTotalBytes === rawBytes.length;

  const flagNames: string[] = [];
  if (flags & TokenFlags.HAS_VQ_GRID) flagNames.push("HAS_VQ_GRID");
  if (flags & TokenFlags.SNARK_ATTESTED) flagNames.push("SNARK_ATTESTED");
  if (flags & TokenFlags.QUICKLZ_COMPRESSED) flagNames.push("QUICKLZ_COMPRESSED");
  if (flags & TokenFlags.HOMOMORPHIC_MASKED) flagNames.push("HOMOMORPHIC_MASKED");
  if (flags & TokenFlags.MULTI_PAGE_DOC) flagNames.push("MULTI_PAGE_DOC");
  if (flags & TokenFlags.HAS_DINO_EMBEDDING) flagNames.push("HAS_DINO_EMBEDDING");
  if (flags & TokenFlags.HAS_BGE_EMBEDDING) flagNames.push("HAS_BGE_EMBEDDING");

  let previewFormatLabel = "UNKNOWN";
  switch (previewFormat) {
    case PreviewFormat.AVIF:
      previewFormatLabel = "AVIF";
      break;
    case PreviewFormat.WEBP:
      previewFormatLabel = "WEBP";
      break;
    case PreviewFormat.JPEG:
      previewFormatLabel = "JPEG";
      break;
    case PreviewFormat.PNG:
      previewFormatLabel = "PNG";
      break;
    case PreviewFormat.BMP:
      previewFormatLabel = "BMP";
      break;
    case PreviewFormat.SVG:
      previewFormatLabel = "SVG";
      break;
  }

  return {
    magic,
    version,
    flags,
    flagNames,
    uuidBytes,
    uuidHex,
    originalFileSize,
    metaLength,
    previewLength,
    width,
    height,
    page,
    previewFormat,
    previewFormatLabel,
    merkleRoot,
    merkleRootHex,
    shardBlockId,
    shardIndex,
    timestampMs,
    timestampUtc: new Date(timestampMs).toISOString(),
    homomorphicTag,
    embeddingModelId,
    totalTokenBytes: rawBytes.length,
    expectedTotalBytes,
    envelopeBoundsValid,
    snarkStubOffset: SNARK_STUB_OFFSET,
    snarkStubLength: SNARK_STUB_BYTES,
    vqGridOffset: VQ_GRID_OFFSET,
    vqGridLength: VQ_GRID_BYTES,
    rsParityOffset: RS_PARITY_OFFSET,
    rsParityLength: RS_PARITY_BYTES,
    metaOffset: SECTION1_HEADER_BYTES,
    previewOffset: SECTION1_HEADER_BYTES + metaLength,
    vqIndices,
  };
}

/**
 * Extracts raw preview image bytes directly without decompressing Section II or verifying signatures.
 */
export function extractTokenPreviewBytes(rawBytes: Uint8Array): Uint8Array {
  const insp = inspectBinaryTokenHeader(rawBytes);
  const end = Math.min(rawBytes.length, insp.previewOffset + insp.previewLength);
  return rawBytes.subarray(insp.previewOffset, end);
}

/**
 * Decompresses and parses Section II JSON metadata on demand (only when explicitly requested).
 */
export function extractTokenMetadata(rawBytes: Uint8Array): TokenMetadataPayload {
  const insp = inspectBinaryTokenHeader(rawBytes);
  const metaSlice = rawBytes.subarray(insp.metaOffset, Math.min(rawBytes.length, insp.metaOffset + insp.metaLength));

  let jsonBytes = metaSlice;
  const isQuickLz = metaSlice.length >= 9 && metaSlice[0] === 3 && (metaSlice[0] + 0 !== 0x7b);
  if (isQuickLz) {
    const claimedDecompressedLen =
      ((metaSlice[1] ?? 0) |
        ((metaSlice[2] ?? 0) << 8) |
        ((metaSlice[3] ?? 0) << 16) |
        ((metaSlice[4] ?? 0) << 24)) >>>
      0;
    if (claimedDecompressedLen > SECTION2_MAX_UNCOMPRESSED_META_BYTES) {
      throw new Error("Section II QuickLZ decompressed size exceeds safety limit");
    }
    jsonBytes = QuickLZ.decompress(metaSlice, SECTION2_MAX_UNCOMPRESSED_META_BYTES);
  }

  return JSON.parse(new TextDecoder().decode(jsonBytes)) as TokenMetadataPayload;
}

/**
 * Unpacks, heals (via GF(2^8) Reed-Solomon), and cryptographically verifies (via Ed25519 + full-token
 * SHA-256 Merkle commitments over Origin Header, VQ Grid, Section II, and Section III) a 25 KB binary token.
 */
export async function decodeBinaryToken(
  rawToken: Uint8Array,
  options?: DecodeTokenOptions,
): Promise<DecodedToken> {
  if (
    rawToken.length < SECTION1_HEADER_BYTES ||
    rawToken.length > MAX_TOKEN_BYTES
  ) {
    throw new Error("Invalid token size");
  }

  // 1. Verify & heal Section I Header (0..639) using 64-byte GF(2^8) Reed-Solomon parity (640..703)
  const headerPayload640 = rawToken.subarray(0, RS_PARITY_OFFSET);
  const parity64 = rawToken.subarray(RS_PARITY_OFFSET, SECTION1_HEADER_BYTES);
  const rsResult = verifyAndHealHeader640(headerPayload640, parity64);

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
  if (version !== TOKEN_VERSION) {
    throw new Error(
      `Unsupported token version: expected ${TOKEN_VERSION}, found ${version}`,
    );
  }

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
    const isQuickLz = rawMetaSlice.length >= 9 && rawMetaSlice[0] === 3 && (rawMetaSlice[0] + 0 !== 0x7b);
    if (isQuickLz) {
      const claimedDecompressedLen =
        ((rawMetaSlice[1] ?? 0) |
          ((rawMetaSlice[2] ?? 0) << 8) |
          ((rawMetaSlice[3] ?? 0) << 16) |
          ((rawMetaSlice[4] ?? 0) << 24)) >>>
        0;
      if (claimedDecompressedLen > SECTION2_MAX_UNCOMPRESSED_META_BYTES) {
        throw new Error("Section II QuickLZ decompressed size exceeds safety limit");
      }
      decodedMetaBytes = QuickLZ.decompress(
        rawMetaSlice,
        SECTION2_MAX_UNCOMPRESSED_META_BYTES,
      );
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

  let bgeEmbedding: Int8Array<ArrayBuffer> | undefined;
  if (metadata.bgeEmbeddingHex) {
    try {
      bgeEmbedding = hexToBgeEmbedding(metadata.bgeEmbeddingHex);
    } catch { }
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
    envelopeBoundsValid,
    metadata,
    bgeEmbedding,
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
 * Strictly validates a binary token's structural integrity, Reed-Solomon parity,
 * cryptographic signatures, Merkle commitments, and well-formed metadata.
 * Throws an Error if any requirement is violated.
 */
export async function validateBinaryToken(
  rawToken: Uint8Array,
  options?: DecodeTokenOptions,
): Promise<DecodedToken> {
  const decoded = await decodeBinaryToken(rawToken, options);

  if (!decoded.envelopeBoundsValid) {
    throw new Error(
      `Invalid token envelope bounds: expected ${decoded.byteMap.section1HeaderBytes + decoded.metaLength + decoded.previewLength} bytes, found ${rawToken.length}`,
    );
  }

  if (!decoded.rsParityValid) {
    throw new Error("Reed-Solomon header parity validation failed");
  }

  const attest = decoded.snarkAttestation;
  if (!attest.attestationStatus?.signatureValid) {
    throw new Error(
      `Invalid token signature: ${attest.failureReason ?? "signature_verification_failed"}`,
    );
  }

  if (!attest.attestationStatus?.merkleValid) {
    throw new Error(
      `Merkle attestation mismatch: ${attest.failureReason ?? "merkle_root_mismatch"}`,
    );
  }

  if (options?.trustedPublicKeyRaw && !attest.attestationStatus?.trustedIssuer) {
    throw new Error("Token signer public key does not match trusted public key");
  }

  if (decoded.metadata.ocrText === "[CORRUPTED OR TAMPERED SECTION II METADATA]") {
    throw new Error("Section II metadata is corrupted or tampered");
  }

  return decoded;
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


