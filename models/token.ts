/**
 * TinyToken Binary Token Specification V2 (GF28 + VQ + SNARK/SHARK)
 * Fixed 25 KB (25,600 bytes max) hybrid token envelope:
 * - Section I   (0..703, 704 B): Header, 256B SNARK/SHARK Attestation Stub, 256B VQ 16x16 Grid, 64B GF(2^8) RS Parity
 * - Section II  (704..4799, up to 4,096 B): Compact Metadata & OCR Text (QuickLZ / UTF-8 JSON)
 * - Section III (4800..25599, up to 20,800 B): High-Efficiency 220x220 Preview Image
 */

export const TOKEN_MAGIC = "GF28";
export const TOKEN_VERSION = 2;

export const MAX_TOKEN_BYTES = 25 * 1024; // 25,600 bytes
export const SECTION1_HEADER_BYTES = 704;
export const SECTION2_MAX_META_BYTES = 4096;
export const SECTION2_MAX_UNCOMPRESSED_META_BYTES = SECTION2_MAX_META_BYTES * 8; // 32,768 bytes (32 KB)
export const SECTION3_MAX_PREVIEW_BYTES = 20800; // 25,600 - 704 - 4,096

export const SNARK_STUB_OFFSET = 128;
export const SNARK_STUB_BYTES = 256; // 128..383

export const VQ_GRID_OFFSET = 384;
export const VQ_GRID_BYTES = 256; // 384..639 (16x16 Uint8 codebook indices in GF(2^8))

export const RS_PARITY_OFFSET = 640;
export const RS_PARITY_BYTES = 64; // 640..703 (64B Reed-Solomon parity over 0..639: 3 lanes [22B, 21B, 21B])

/**
 * 32-byte SHA-256 Merkle root hex string (64 characters).
 * Serves as the token's immutable, content-addressed cryptographic identity (CID).
 * Commits across all 4 leaves: Origin identity, Section III Preview, Section II Metadata, and Section I VQ grid.
 */
export type ContentId = string;

export enum TokenFlags {
  NONE = 0,
  HAS_VQ_GRID = 1 << 0,        // 0x01: 16x16 VQ neural thumbnail present
  SNARK_ATTESTED = 1 << 1,     // 0x02: 256B SNARK/SHARK provenance stub verified
  QUICKLZ_COMPRESSED = 1 << 2, // 0x04: Section II metadata is QuickLZ compressed
  HOMOMORPHIC_MASKED = 1 << 3, // 0x08: GF(2^8) homomorphic redaction/watermark applied
  MULTI_PAGE_DOC = 1 << 4,     // 0x10: Token represents a page from a multi-page document
  HAS_DINO_EMBEDDING = 1 << 5, // 0x20: 384-byte DINOv3 ViT-S/16 Int8 semantic embedding present
  HAS_BGE_EMBEDDING = 1 << 6,  // 0x40: 384-byte BGE-small-en-v1.5 Int8 semantic text embedding present
}

export enum PreviewFormat {
  UNKNOWN = 0,
  AVIF = 1,
  WEBP = 2,
  JPEG = 3,
  PNG = 4,
  BMP = 5,
  SVG = 6,
}

export enum EmbeddingModelId {
  NONE = 0x00,
  COLOR_HOG_FALLBACK = 0x01, // Synchronous 384B Color/HOG fallback descriptor
  DINOV3_VITS16_Q8 = 0x02,   // Real Meta DINOv3 ViT-S/16 ONNX (384-dim CLS int8)
  BGE_TEXT_FALLBACK = 0x03,
  BGE_SMALL_EN_V15_Q8 = 0x04,
}

export const DEFAULT_CLIP_TEXT_LIMIT = 2200;

/**
 * Returns the app-wide text clipping ceiling in characters, read from APP_CLIP_TEXT_LIMIT_CHARS (or fallback).
 */
export function getClipTextLimit(): number {
  const val = Bun.env.APP_CLIP_TEXT_LIMIT_CHARS ?? Bun.env.APP_CLIP_TEXT_LIMIT_CHAR;
  const num = val ? Number(val) : NaN;
  return Number.isFinite(num) && num > 0 ? num : DEFAULT_CLIP_TEXT_LIMIT;
}

export interface TokenMetadataPayload {
  id: string;
  filename: string;
  sourcePath: string;
  mimeType: string;
  page: number;
  totalPages: number;
  width: number;
  height: number;
  colorDepth: number;
  createdAt: string;
  ocrText: string;
  dinoEmbeddingHex?: string; // 384-byte (768 hex chars) Int8 DINOv3 ViT-S/16 semantic embedding
  bgeEmbeddingHex?: string;  // 384-byte (768 hex chars) Int8 BGE-small-en-v1.5 semantic text embedding
  bgeModelId?: EmbeddingModelId;
  tags?: string[];
  environmentClass?: string;
  geoCoordinates?: {
    latitude: number;
    longitude: number;
  };
}

export interface AttestationStatus {
  signatureValid: boolean;
  trustedIssuer: boolean;
  merkleValid: boolean;
  envelopeValid: boolean;
}

export interface SnarkAttestationInfo {
  protocol: "ED25519-MERKLE-GF28-v3";
  valid: boolean;
  signatureVerified: boolean;
  trustedKeyMatch: boolean;
  attestationStatus?: AttestationStatus;
  sectionsVerified: {
    originHeader: boolean;
    vqGrid: boolean;
    section2Metadata: boolean;
    section3Preview: boolean;
    merkleRoot: boolean;
  };
  signerKeyIdHex: string;
  signerPublicKeyHex: string;
  circuitCommitmentHex: string;
  assetOriginHashHex: string;
  previewHashHex: string;
  ocrWitnessHashHex: string;
  vqCommitmentHex: string;
  gfPolynomialEval: number;
  verifiedInUs: number;
  failureReason?: string;
}

export interface DecodedToken {
  id: string;
  magic: string;
  version: number;
  flags: number;
  uuidHex: string;
  originalFileSize: number;
  metaLength: number;
  previewLength: number;
  width: number;
  height: number;
  page: number;
  previewFormat: PreviewFormat;
  merkleRootHex: string;
  contentId?: ContentId;
  shardBlockId: number;
  shardIndex: number;
  timestampMs: number;
  homomorphicTag: number;
  embeddingModelId: EmbeddingModelId;
  snarkStub: Uint8Array;
  snarkAttestation: SnarkAttestationInfo;
  vqIndices: Uint8Array; // 256 bytes (16x16 codebook indices)
  rsParityValid: boolean;
  rsCorrectedErrors: number;
  envelopeBoundsValid: boolean;
  metadata: TokenMetadataPayload;
  bgeEmbedding?: Int8Array;
  previewBytes: Uint8Array;
  byteMap: {
    totalTokenBytes: number;
    section1HeaderBytes: number;
    snarkStubBytes: number;
    vqGridBytes: number;
    rsParityBytes: number;
    section2MetaBytes: number;
    section3PreviewBytes: number;
  };
}

export interface TokenHeaderInspection {
  magic: string;
  version: number;
  flags: number;
  flagNames: string[];
  uuidBytes: Uint8Array;
  uuidHex: string;
  originalFileSize: number;
  metaLength: number;
  previewLength: number;
  width: number;
  height: number;
  page: number;
  previewFormat: PreviewFormat;
  previewFormatLabel: string;
  merkleRoot: Uint8Array;
  merkleRootHex: string;
  contentId: ContentId;
  shardBlockId: number;
  shardIndex: number;
  timestampMs: number;
  timestampUtc: string;
  homomorphicTag: number;
  embeddingModelId: EmbeddingModelId;
  totalTokenBytes: number;
  expectedTotalBytes: number;
  envelopeBoundsValid: boolean;
  snarkStubOffset: number;
  snarkStubLength: number;
  vqGridOffset: number;
  vqGridLength: number;
  rsParityOffset: number;
  rsParityLength: number;
  metaOffset: number;
  previewOffset: number;
  vqIndices: Uint8Array;
}

export interface ZipTokenIndexEntry {
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
  sourceHash?: string;
  previewFormat: PreviewFormat;
  flags: number;
  ocrText: string;
  createdAt: string;
  localHeaderOffset: number;
  dataOffset: number;
  compressedSize: number;
  uncompressedSize: number;
  compressionMethod: number; // 0 = STORE, 8 = DEFLATE
  crc32: number;
  merkleRootHex: string;
  contentId?: ContentId;
  snarkVerified: boolean;
  vqIndices: Uint8Array; // 256 bytes kept in RAM for 0ms neural preview & ADC search
  dinoEmbedding?: Int8Array; // 384 bytes kept in RAM for sub-ms DINOv3 semantic similarity
  bgeEmbedding?: Int8Array; // 384 bytes kept in RAM for sub-ms BGE semantic text similarity
  embeddingModelId?: EmbeddingModelId;
  bgeModelId?: EmbeddingModelId;
}

export interface SearchHitDTO {
  id: string;
  filename: string;
  sourcePath: string;
  mimeType: string;
  page: number;
  totalPages: number;
  width: number;
  height: number;
  originalFileSize: number;
  tokenSize: number;
  previewFormat: string;
  flags: number;
  ocrSnippet: string;
  createdAt: string;
  merkleRootHex: string;
  contentId?: ContentId;
  snarkVerified: boolean;
  vqHex: string; // 512-char hex of 256-byte VQ 16x16 grid
  similarityScore?: number; // 0..100% hybrid DINOv3 + VQ similarity score
  dinoSimilarity?: number | null; // 0..100% DINOv3 ViT-S/16 semantic cosine score (null if un-evaluated or incompatible)
  bgeSimilarity?: number | null; // 0..100% BGE semantic text cosine score
  adcDistance?: number;
}
