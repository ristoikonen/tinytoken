/**
 * TinyToken Binary Token Specification V2 (GF28 + VQ + SNARK/SHARK)
 * Fixed 25 KB (25,600 bytes max) hybrid token envelope:
 * - Section I   (0..703, 704 B): Header, 256B SNARK/SHARK Attestation Stub, 256B VQ 16x16 Grid, 64B GF(2^8) RS Parity
 * - Section II  (704..4799, up to 4,096 B): Compact Metadata & OCR Text (QuickLZ / UTF-8 JSON)
 * - Section III (4800..25599, up to 20,800 B): High-Efficiency 320x320 Preview Image
 */

export const TOKEN_MAGIC = "GF28";
export const TOKEN_VERSION = 2;

export const MAX_TOKEN_BYTES = 25 * 1024; // 25,600 bytes
export const SECTION1_HEADER_BYTES = 704;
export const SECTION2_MAX_META_BYTES = 4096;
export const SECTION2_MAX_UNCOMPRESSED_META_BYTES = SECTION2_MAX_META_BYTES * 8; // 32,768 bytes (32 KB)
export const MAX_PREVIEW_BYTES = 16 * 1024; // 16,384 bytes (16 KB global preview byte cap)
export const SECTION3_MAX_PREVIEW_BYTES = MAX_PREVIEW_BYTES;

/**
 * Single centralized source of truth for WebP encoding, preview dimensions, and rasterization settings.
 */
export const PREVIEW_TARGET_DIMENSION = 320;
export const PREVIEW_TARGET_WIDTH = 320;
export const PREVIEW_TARGET_HEIGHT = 320;
export const PREVIEW_FALLBACK_DIMENSION = 240;
export const PREVIEW_PDF_TARGET_WIDTH = 320;
export const PREVIEW_PDF_TARGET_HEIGHT = 452;
export const PREVIEW_PDF_FALLBACK_WIDTH = 240;
export const PREVIEW_PDF_FALLBACK_HEIGHT = 340;
export const PDF_RENDER_SCALE = 1.5;
export const DEFAULT_DOC_PAGE_WIDTH = 1240;
export const DEFAULT_DOC_PAGE_HEIGHT = 1754;
export const DINO_DECODE_DIMENSION = 384;
export const WEBP_PRIMARY_QUALITY = 78;
export const WEBP_FALLBACK_QUALITY = 68;
export const WEBP_WARNING_QUALITY = 75;
export const WEBP_EFFORT = 2;
export const WEBP_SMART_SUBSAMPLE = false;

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

export function previewFormatToLabel(format: PreviewFormat): string {
  switch (format) {
    case PreviewFormat.AVIF:
      return "AVIF";
    case PreviewFormat.WEBP:
      return "WEBP";
    case PreviewFormat.JPEG:
      return "JPEG";
    case PreviewFormat.PNG:
      return "PNG";
    case PreviewFormat.BMP:
      return "BMP";
    case PreviewFormat.SVG:
      return "SVG";
    default:
      return "UNKNOWN";
  }
}

export enum EmbeddingModelId {
  NONE = 0x00,
  COLOR_HOG_FALLBACK = 0x01, // Synchronous 384B Color/HOG fallback descriptor
  DINOV3_VITS16_Q8 = 0x02,   // Real Meta DINOv3 ViT-S/16 ONNX (384-dim CLS int8)
  BGE_TEXT_FALLBACK = 0x03,
  BGE_SMALL_EN_V15_Q8 = 0x04,
}

export const DEFAULT_CLIP_TEXT_LIMIT = 2200;
export const DEFAULT_MAX_DOCX_MB = 20;
export const DEFAULT_MAX_PAGES_PER_DOC = 4;
export const DEFAULT_WORKER_COUNT = 4;

export function getMaxDocxMb(): number {
  const raw = Bun.env.APP_MAX_DOCX_MB;
  const num = raw ? Number(raw) : NaN;
  return Number.isFinite(num) && num > 0 ? num : DEFAULT_MAX_DOCX_MB;
}

export function getMaxDecompressedDocxBytes(): number {
  return getMaxDocxMb() * 1024 * 1024;
}

export function getMaxPagesPerDoc(): number {
  const raw = Bun.env.APP_MAX_PAGES_PER_DOC;
  const num = raw ? Number.parseInt(raw, 10) : NaN;
  return Number.isFinite(num) && num > 0 ? num : DEFAULT_MAX_PAGES_PER_DOC;
}

export function getWorkerCount(): number {
  const raw = Bun.env.APP_WORKER_COUNT;
  const num = raw ? Number.parseInt(raw, 10) : NaN;
  return Number.isFinite(num) && num > 0 ? num : DEFAULT_WORKER_COUNT;
}

/**
 * Centralized Paths, Directories & Model Identifiers (configurable via .env).
 */
export const DEFAULT_IMAGE_DIR = "./images";
export const DEFAULT_DOCUMENT_DIR = "./documents";
export const DEFAULT_SQLITE_DB_PATH = ":memory:";
export const DEFAULT_TRANSFORMERS_CACHE_DIR = "./.cache/transformers";
export const DINO_MODEL_ID = "onnx-community/dinov3-vits16-pretrain-lvd1689m-ONNX";
export const BGE_MODEL_ID = "Xenova/bge-small-en-v1.5";

export function getAppImageDir(cliOverride?: string): string {
  return (
    Bun.env.APP_IMAGE_DIR ??
    Bun.env.IMAGE_DIR ??
    cliOverride ??
    DEFAULT_IMAGE_DIR
  );
}

export function getAppDocumentDir(): string {
  return (
    Bun.env.APP_DOCUMENT_DIR ??
    Bun.env.DOCUMENT_DIR ??
    DEFAULT_DOCUMENT_DIR
  );
}

export function getAppSqliteDbPath(): string {
  return (
    Bun.env.APP_SQLITE_DB_PATH ??
    Bun.env.SQLITE_DB_PATH ??
    DEFAULT_SQLITE_DB_PATH
  );
}

export function getTransformersCacheDir(): string {
  return Bun.env.APP_TRANSFORMERS_CACHE_DIR ?? DEFAULT_TRANSFORMERS_CACHE_DIR;
}

/**
 * Redacts the configured APP_IMAGE_DIR filesystem path to the literal label "APP_IMAGE_DIR"
 * for debug logging, UI responses, and OCR source notes.
 */
export function redactImageDirForDisplay(text: string): string {
  if (!text) return text;
  const rawAppImageDir = (Bun.env.APP_IMAGE_DIR ?? "").trim().replace(/[\\/]+$/, "");
  if (!rawAppImageDir || rawAppImageDir === "." || rawAppImageDir === "./images") {
    return text;
  }
  const forwardVariant = rawAppImageDir.replace(/\\/g, "/");
  const backVariant = rawAppImageDir.replace(/\//g, "\\");
  let out = text;
  if (forwardVariant) {
    out = out.split(forwardVariant).join("APP_IMAGE_DIR");
  }
  if (backVariant && backVariant !== forwardVariant) {
    out = out.split(backVariant).join("APP_IMAGE_DIR");
  }
  return out;
}

/**
 * Default Canberra ACT / Australia geographic coordinates when asset metadata lacks GPS.
 */
export const DEFAULT_GEO_LATITUDE = -35.1843;
export const DEFAULT_GEO_LONGITUDE = 149.1328;

/**
 * Dual-Score Hybrid Visual Similarity Weight (α = 0.85):
 * Fused Score = α · DINOv3_Cosine + (1 - α) · VQ_Overlap
 */
export const HYBRID_SIMILARITY_ALPHA = 0.85;

export function getHybridSimilarityAlpha(): number {
  const raw = Bun.env.APP_HYBRID_SIMILARITY_ALPHA;
  if (raw !== undefined) {
    const parsed = Number(raw);
    if (Number.isFinite(parsed) && parsed >= 0 && parsed <= 1) {
      return parsed;
    }
  }
  return HYBRID_SIMILARITY_ALPHA;
}

/**
 * Single centralized registry of allowed asset file extensions across the server, indexer, CLI, and document processors.
 */
export class AllowedFileTypes {
  public static readonly IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".avif",
    ".bmp",
    ".svg",
  ] as const;

  public static readonly TEXT_DOCUMENT_EXTENSIONS = [
    ".txt",
    ".text",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".log",
    ".yaml",
    ".yml",
  ] as const;

  public static readonly DOCUMENT_EXTENSIONS = [
    ".pdf",
    ".docx",
    ".doc",
    ...AllowedFileTypes.TEXT_DOCUMENT_EXTENSIONS,
  ] as const;

  public static readonly ALL_EXTENSIONS = [
    ...AllowedFileTypes.IMAGE_EXTENSIONS,
    ...AllowedFileTypes.DOCUMENT_EXTENSIONS,
  ] as const;

  public static readonly IMAGE_SET: ReadonlySet<string> = new Set(
    AllowedFileTypes.IMAGE_EXTENSIONS,
  );
  public static readonly TEXT_DOCUMENT_SET: ReadonlySet<string> = new Set(
    AllowedFileTypes.TEXT_DOCUMENT_EXTENSIONS,
  );
  public static readonly DOCUMENT_SET: ReadonlySet<string> = new Set(
    AllowedFileTypes.DOCUMENT_EXTENSIONS,
  );
  public static readonly ALL_SET: ReadonlySet<string> = new Set(
    AllowedFileTypes.ALL_EXTENSIONS,
  );

  public static normalizeExtension(ext: string): string {
    const trimmed = ext.trim().toLowerCase();
    if (!trimmed) return "";
    return trimmed.startsWith(".") ? trimmed : `.${trimmed}`;
  }

  public static list(): string[] {
    return [...AllowedFileTypes.ALL_EXTENSIONS];
  }

  public static isAllowed(ext: string): boolean {
    return AllowedFileTypes.ALL_SET.has(AllowedFileTypes.normalizeExtension(ext));
  }

  public static isImage(ext: string): boolean {
    return AllowedFileTypes.IMAGE_SET.has(AllowedFileTypes.normalizeExtension(ext));
  }

  public static isDocument(ext: string): boolean {
    return AllowedFileTypes.DOCUMENT_SET.has(AllowedFileTypes.normalizeExtension(ext));
  }

  public static isTextDocument(ext: string): boolean {
    return AllowedFileTypes.TEXT_DOCUMENT_SET.has(AllowedFileTypes.normalizeExtension(ext));
  }
}

export const ALLOWED_IMAGE_EXTENSIONS = AllowedFileTypes.IMAGE_EXTENSIONS;
export const ALLOWED_TEXT_DOCUMENT_EXTENSIONS = AllowedFileTypes.TEXT_DOCUMENT_EXTENSIONS;
export const ALLOWED_DOCUMENT_EXTENSIONS = AllowedFileTypes.DOCUMENT_EXTENSIONS;
export const ALLOWED_EXTENSIONS = AllowedFileTypes.ALL_EXTENSIONS;

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
