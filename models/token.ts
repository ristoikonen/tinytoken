/**
 * TinyToken Binary Token Specification V2 (GF28 + VQ + SNARK/SHARK)
 * Fixed 25 KB (25,600 bytes max) hybrid token envelope:
 * - Section I   (0..1023, 1,024 B): Header, 256B SNARK/SHARK Attestation Stub, 256B VQ 16x16 Grid, 384B GF(2^8) RS Parity
 * - Section II  (1024..5119, up to 4,096 B): Compact Metadata & OCR Text (QuickLZ / UTF-8 JSON)
 * - Section III (5120..25599, up to 20,480 B): High-Efficiency 220x220 Preview Image
 */

export const TOKEN_MAGIC = "GF28";
export const TOKEN_VERSION = 2;

export const MAX_TOKEN_BYTES = 25 * 1024; // 25,600 bytes
export const SECTION1_HEADER_BYTES = 1024;
export const SECTION2_MAX_META_BYTES = 4096;
export const SECTION3_MAX_PREVIEW_BYTES = 20480;

export const SNARK_STUB_OFFSET = 128;
export const SNARK_STUB_BYTES = 256; // 128..383

export const VQ_GRID_OFFSET = 384;
export const VQ_GRID_BYTES = 256; // 384..639 (16x16 Uint8 codebook indices in GF(2^8))

export const RS_PARITY_OFFSET = 640;
export const RS_PARITY_BYTES = 384; // 640..1023 (3x128B Reed-Solomon parity over 0..639)

export enum TokenFlags {
  NONE = 0,
  HAS_VQ_GRID = 1 << 0,        // 0x01: 16x16 VQ neural thumbnail present
  SNARK_ATTESTED = 1 << 1,     // 0x02: 256B SNARK/SHARK provenance stub verified
  QUICKLZ_COMPRESSED = 1 << 2, // 0x04: Section II metadata is QuickLZ compressed
  HOMOMORPHIC_MASKED = 1 << 3, // 0x08: GF(2^8) homomorphic redaction/watermark applied
  MULTI_PAGE_DOC = 1 << 4,     // 0x10: Token represents a page from a multi-page document
  HAS_DINO_EMBEDDING = 1 << 5, // 0x20: 384-byte DINOv2 ViT-14 Int8 semantic embedding present
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
  DINOV2_VIT_S14_Q8 = 0x02,  // Real Meta/Xenova DINOv2 ViT-S/14 ONNX (384-dim CLS int8)
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
  dinoEmbeddingHex?: string; // 384-byte (768 hex chars) Int8 DINOv2 ViT-14 semantic embedding
  tags?: string[];
  environmentClass?: string;
  geoCoordinates?: {
    latitude: number;
    longitude: number;
  };
}

export interface SnarkAttestationInfo {
  protocol: "SHARK-GF28-Groth16Stub-v2";
  valid: boolean;
  circuitCommitmentHex: string;
  assetOriginHashHex: string;
  ocrWitnessHashHex: string;
  vqCommitmentHex: string;
  gfPolynomialEval: number;
  verifiedInUs: number;
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
  metadata: TokenMetadataPayload;
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
  snarkVerified: boolean;
  vqIndices: Uint8Array; // 256 bytes kept in RAM for 0ms neural preview & ADC search
  dinoEmbedding?: Int8Array; // 384 bytes kept in RAM for sub-ms DINOv2 semantic similarity
  embeddingModelId?: EmbeddingModelId;
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
  snarkVerified: boolean;
  vqHex: string; // 512-char hex of 256-byte VQ 16x16 grid
  similarityScore?: number; // 0..100% hybrid DINOv2 + VQ similarity score
  dinoSimilarity?: number; // 0..100% DINOv2 ViT-14 semantic cosine score
  adcDistance?: number;
}

