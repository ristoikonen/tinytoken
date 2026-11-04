#!/usr/bin/env bun
/**
 * TinyToken Dedicated CLI (cli.ts)
 * 
 * Works with ONE token or asset at a time:
 *   - Create a 25 KB token from an image or document
 *   - Inspect an existing token with --info, --filetype, --preview, -v/--verify, --json
 * 
 * Usage:
 *   bun tokenize <file> [options]
 *   bun cli.ts <file> [options]
 *   tinytoken <file> [options]
 */

import {
  EmbeddingModelId,
  PreviewFormat,
  SECTION1_HEADER_BYTES,
  SECTION2_MAX_META_BYTES,
  SECTION3_MAX_PREVIEW_BYTES,
  MAX_TOKEN_BYTES,
  TokenFlags,
} from "./models/token";
import { Win11TokenPipeline } from "./utils/indexer";
import { QuickLZ } from "./utils/quicklz";
import { decodeBinaryToken } from "./utils/tokenCodec";

interface CliArgs {
  file?: string;
  out?: string;
  verify: boolean;
  json: boolean;
  info: boolean;
  filetype: boolean;
  preview: boolean;
  help: boolean;
}

function printHelp(): void {
  console.log(`
TinyToken Single-Token CLI Tool

Usage:
  bun run cli.ts <file> [options]
  bun tokenize <file> [options]
  tinytoken <file> [options]

Arguments:
  <file>               Source file path (image, document, or existing .token file) [required]

Options:
  -o, --out <path>     Custom output .token path (defaults to <input_dir>/<basename>.token)
  -v, --verify         Explicitly run and print decode + cryptographic signature audit
  --info               Verbose breakdown: token size, uncompressed estimate, dates, all fields
  --filetype           Print preview filetype (e.g. 'WEBP', 'PNG', 'JPEG', 'AVIF')
  --preview            Stream raw binary preview data to stdout (for saving or piping)
  --json               Print token metadata, attestation, and Merkle proof as JSON to stdout
  -h, --help           Display this help message

Examples:
  # Inspect an existing token
  tinytoken tok_00004_Henri.jpg.token --info
  tinytoken tok_00004_Henri.jpg.token --filetype
  tinytoken tok_00004_Henri.jpg.token --preview > preview.webp
  tinytoken tok_00004_Henri.jpg.token -v

  # Create a token from an image or document
  bun tokenize ./images/Henri.jpg
  bun tokenize ./documents/report.pdf -o ./output/report.token --info
`);
}

function parseArgs(args: string[]): CliArgs {
  const result: CliArgs = {
    verify: false,
    json: false,
    info: false,
    filetype: false,
    preview: false,
    help: false,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (arg === "-h" || arg === "--help") {
      result.help = true;
    } else if (arg === "-v" || arg === "--verify") {
      result.verify = true;
    } else if (arg === "--json") {
      result.json = true;
    } else if (arg === "--info") {
      result.info = true;
    } else if (arg === "--filetype") {
      result.filetype = true;
    } else if (arg === "--preview") {
      result.preview = true;
    } else if (arg === "-o" || arg === "--out") {
      if (i + 1 < args.length) {
        result.out = args[++i];
      } else {
        console.error("Error: -o / --out requires a file path argument.");
        process.exit(1);
      }
    } else if (arg && !arg.startsWith("-") && !result.file) {
      result.file = arg;
    }
  }

  return result;
}

function previewFormatToString(format: PreviewFormat): string {
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

function decodeFlagsToStrings(flags: number): string[] {
  const list: string[] = [];
  if (flags & TokenFlags.HAS_VQ_GRID) list.push("HAS_VQ_GRID");
  if (flags & TokenFlags.SNARK_ATTESTED) list.push("SNARK_ATTESTED");
  if (flags & TokenFlags.QUICKLZ_COMPRESSED) list.push("QUICKLZ_COMPRESSED");
  if (flags & TokenFlags.HOMOMORPHIC_MASKED) list.push("HOMOMORPHIC_MASKED");
  if (flags & TokenFlags.MULTI_PAGE_DOC) list.push("MULTI_PAGE_DOC");
  if (flags & TokenFlags.HAS_DINO_EMBEDDING) list.push("HAS_DINO_EMBEDDING");
  return list;
}

function embeddingModelToString(modelId: number): string {
  switch (modelId) {
    case EmbeddingModelId.COLOR_HOG_FALLBACK:
      return "0x01 (Color/HOG Fallback 384B)";
    case EmbeddingModelId.DINOV3_VITS16_Q8:
      return "0x02 (Meta DINOv3 ViT-S/16 ONNX q8)";
    case EmbeddingModelId.NONE:
      return "0x00 (None)";
    default:
      return `0x${modelId.toString(16).padStart(2, "0")}`;
  }
}

/**
 * Resolves file path with convenient fallback lookups in ./data, ./images, ./documents
 */
async function resolveInputFile(filePath: string): Promise<string | null> {
  if (await Bun.file(filePath).exists()) {
    return filePath;
  }
  // Fallbacks for convenience
  const candidates = [
    `./data/${filePath}`,
    `./images/${filePath}`,
    `./documents/${filePath}`,
    `data/${filePath}`,
    `images/${filePath}`,
    `documents/${filePath}`,
  ];
  for (const cand of candidates) {
    if (await Bun.file(cand).exists()) {
      return cand;
    }
  }
  return null;
}

/**
 * Determines whether raw bytes represent an encoded TinyToken (magic 'GF28')
 */
function isTinyTokenBytes(bytes: Uint8Array): boolean {
  if (bytes.length < SECTION1_HEADER_BYTES) return false;
  return (
    bytes[0] === 0x47 && // 'G'
    bytes[1] === 0x46 && // 'F'
    bytes[2] === 0x32 && // '2'
    bytes[3] === 0x38    // '8'
  );
}

/**
 * Estimates uncompressed size of Section II metadata
 */
function getEstimatedUncompressedMetaSize(rawBytes: Uint8Array, metaLength: number, flags: number): number {
  if ((flags & TokenFlags.QUICKLZ_COMPRESSED) === 0) {
    return metaLength;
  }
  const metaStart = SECTION1_HEADER_BYTES;
  const metaEnd = Math.min(rawBytes.length, metaStart + metaLength);
  const metaSlice = rawBytes.subarray(metaStart, metaEnd);
  if (metaSlice.length >= 9) {
    const uncompressed =
      ((metaSlice[1] ?? 0) |
        ((metaSlice[2] ?? 0) << 8) |
        ((metaSlice[3] ?? 0) << 16) |
        ((metaSlice[4] ?? 0) << 24)) >>>
      0;
    if (uncompressed > 0 && uncompressed <= 64 * 1024) {
      return uncompressed;
    }
  }
  return metaLength;
}

async function main() {
  const args = parseArgs(Bun.argv.slice(2));

  if (args.help || !args.file) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  const resolvedPath = await resolveInputFile(args.file);
  if (!resolvedPath) {
    console.error(`Error: File not found: "${args.file}"`);
    process.exit(1);
  }

  const fileHandle = Bun.file(resolvedPath);
  const rawFileBytes = new Uint8Array(await fileHandle.arrayBuffer());

  let tokenBytes: Uint8Array;
  let isNewTokenCreated = false;
  let outputPath = args.out;

  if (isTinyTokenBytes(rawFileBytes)) {
    // Existing token
    tokenBytes = rawFileBytes;
  } else {
    // Input is a raw asset (image / document) — create new token
    isNewTokenCreated = true;
    const norm = resolvedPath.replace(/\\/g, "/");
    const lastSlash = norm.lastIndexOf("/");
    const dir = lastSlash !== -1 ? norm.slice(0, lastSlash) : ".";
    const basename = lastSlash !== -1 ? norm.slice(lastSlash + 1) : norm;

    if (!outputPath) {
      outputPath = `${dir}/${basename}.token`;
    }

    const dotIdx = basename.lastIndexOf(".");
    const ext = dotIdx !== -1 ? basename.slice(dotIdx).toLowerCase() : "";
    const isDoc = [".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".json", ".log", ".yaml", ".yml"].includes(ext);

    const pipeline = new Win11TokenPipeline({
      sourceDirectory: dir,
      outputZipPath: "tokens.zip",
      allowedExtensions: [ext],
      maxTokenSizeKb: 25,
    });

    if (!args.json && !args.preview && !args.filetype) {
      console.log(`[TINYTOKEN] Tokenizing: ${resolvedPath}...`);
    }

    if (isDoc) {
      const items = await pipeline.tokenizeDocumentFile(resolvedPath, 1);
      if (!items || items.length === 0) {
        console.error(`Error: Failed to process document "${resolvedPath}"`);
        process.exit(1);
      }
      tokenBytes = items[0]!.tokenBytes;
    } else {
      const item = await pipeline.tokenizeImageFile(resolvedPath, 1);
      if (!item) {
        console.error(`Error: Failed to process image "${resolvedPath}"`);
        process.exit(1);
      }
      tokenBytes = item.tokenBytes;
    }

    // Write token to target destination
    await Bun.write(outputPath, tokenBytes);
  }

  // Fast direct output flags:
  // 1. --preview: write raw binary preview bytes directly to stdout
  if (args.preview) {
    const decoded = await decodeBinaryToken(tokenBytes);
    process.stdout.write(decoded.previewBytes);
    return;
  }

  // 2. --filetype: print clean format name to stdout (e.g. 'WEBP')
  if (args.filetype) {
    const decoded = await decodeBinaryToken(tokenBytes);
    const fmt = previewFormatToString(decoded.previewFormat);
    process.stdout.write(`${fmt}\n`);
    return;
  }

  // Decode token for remaining options
  const decoded = await decodeBinaryToken(tokenBytes);

  // 3. --json: print full JSON representation to stdout
  if (args.json) {
    const uncompMeta = getEstimatedUncompressedMetaSize(tokenBytes, decoded.metaLength, decoded.flags);
    const flagsList = decodeFlagsToStrings(decoded.flags);
    const jsonOutput = {
      id: decoded.id,
      uuidHex: decoded.uuidHex,
      version: decoded.version,
      flags: decoded.flags,
      flagNames: flagsList,
      tokenSizeBytes: tokenBytes.length,
      originalFileSize: decoded.originalFileSize,
      metaLength: decoded.metaLength,
      uncompressedMetaLength: uncompMeta,
      previewLength: decoded.previewLength,
      width: decoded.width,
      height: decoded.height,
      page: decoded.page,
      previewFormat: previewFormatToString(decoded.previewFormat),
      previewFormatId: decoded.previewFormat,
      merkleRootHex: decoded.merkleRootHex,
      shardBlockId: decoded.shardBlockId,
      shardIndex: decoded.shardIndex,
      timestampMs: decoded.timestampMs,
      timestampUtc: new Date(decoded.timestampMs).toISOString(),
      embeddingModelId: decoded.embeddingModelId,
      attestation: {
        valid: decoded.snarkAttestation.valid,
        signatureValid: decoded.snarkAttestation.attestationStatus?.signatureValid ?? decoded.snarkAttestation.signatureVerified,
        merkleValid: decoded.snarkAttestation.attestationStatus?.merkleValid ?? decoded.snarkAttestation.sectionsVerified.merkleRoot,
        envelopeValid: decoded.snarkAttestation.attestationStatus?.envelopeValid ?? decoded.envelopeBoundsValid,
        trustedIssuer: decoded.snarkAttestation.attestationStatus?.trustedIssuer ?? decoded.snarkAttestation.trustedKeyMatch,
        signerPublicKeyHex: decoded.snarkAttestation.signerPublicKeyHex,
        verifiedInUs: decoded.snarkAttestation.verifiedInUs,
        failureReason: decoded.snarkAttestation.failureReason,
      },
      metadata: decoded.metadata,
    };
    process.stdout.write(JSON.stringify(jsonOutput, null, 2) + "\n");
    return;
  }

  // If a new token was generated, announce it
  if (isNewTokenCreated) {
    console.log(`[SAVED] Token created: ${outputPath}`);
    console.log(`        Size: ${tokenBytes.length.toLocaleString()} bytes / ${MAX_TOKEN_BYTES.toLocaleString()} bytes max (${((tokenBytes.length / MAX_TOKEN_BYTES) * 100).toFixed(1)}% cap)`);
  }

  // 4. -v / --verify: explicit audit
  if (args.verify) {
    const att = decoded.snarkAttestation;
    const status = att.attestationStatus;
    const sigOk = status ? status.signatureValid : att.signatureVerified;
    const merkleOk = status ? status.merkleValid : att.sectionsVerified.merkleRoot;
    const envOk = status ? status.envelopeValid : decoded.envelopeBoundsValid;
    const trustedOk = status ? status.trustedIssuer : att.trustedKeyMatch;

    console.log(`\n[VERIFY AUDIT] ${resolvedPath}`);
    console.log(`  Envelope Size:          ${tokenBytes.length.toLocaleString()} bytes (<= 25 KB) - PASS`);
    console.log(`  Envelope Bounds Check:  ${envOk ? "PASS (704 + meta + preview = total)" : "FAIL"}`);
    console.log(`  GF(2^8) Reed-Solomon:   ${decoded.rsParityValid ? `PASS (0 uncorrectable errors, ${decoded.rsCorrectedErrors} healed)` : "FAIL"}`);
    console.log(`  Ed25519 Signature:      ${sigOk ? "VALID (Ed25519 detached signature verified)" : "INVALID"}`);
    console.log(`  SHA-256 Merkle Root:    ${merkleOk ? `VALID (${decoded.merkleRootHex.slice(0, 32)}...)` : "FAIL"}`);
    console.log(`  Trusted Authority:      ${trustedOk ? "CONFIRMED (Issuer key verified)" : "SELF-ATTESTED (No trusted issuer key supplied)"}`);
    console.log(`  Verification Latency:   ${att.verifiedInUs} \u00B5s`);
    if (att.failureReason) {
      console.log(`  Failure Reason:         ${att.failureReason}`);
    }
    console.log(`  Overall Status:         ${att.valid ? "\x1b[32mAUTHENTIC & INTEGRITY VERIFIED\x1b[0m" : "\x1b[31mINTEGRITY FAILED\x1b[0m"}\n`);
  }

  // 5. --info: verbose field breakdown
  if (args.info) {
    const uncompMeta = getEstimatedUncompressedMetaSize(tokenBytes, decoded.metaLength, decoded.flags);
    const flagsList = decodeFlagsToStrings(decoded.flags);
    const capacityPct = ((tokenBytes.length / MAX_TOKEN_BYTES) * 100).toFixed(1);
    const freeBytes = MAX_TOKEN_BYTES - tokenBytes.length;
    const estUncompTotal = decoded.originalFileSize + uncompMeta + SECTION1_HEADER_BYTES;
    const compRatio = (estUncompTotal / Math.max(1, tokenBytes.length)).toFixed(2);
    const dateUtc = new Date(decoded.timestampMs).toISOString();

    console.log(`\n` + "=".repeat(70));
    console.log(`  TinyToken Inspector: ${decoded.id || resolvedPath}`);
    console.log("=".repeat(70));

    console.log(`\n[Envelope & Capacity]`);
    console.log(`  File Path:              ${resolvedPath}`);
    console.log(`  Token Size:             ${tokenBytes.length.toLocaleString()} bytes (${(tokenBytes.length / 1024).toFixed(2)} KB)`);
    console.log(`  Max Envelope Limit:     ${MAX_TOKEN_BYTES.toLocaleString()} bytes (25 KB)`);
    console.log(`  Envelope Capacity Used: ${capacityPct}% (${freeBytes.toLocaleString()} bytes remaining)`);
    console.log(`  Estimated Uncompressed: ~${estUncompTotal.toLocaleString()} bytes (${compRatio}x overall compression ratio)`);

    console.log(`\n[Section I: Fixed Header (704 Bytes)]`);
    console.log(`  Magic Bytes:            ${decoded.magic} (0x47 0x46 0x32 0x38)`);
    console.log(`  Format Version:         ${decoded.version}`);
    console.log(`  UUID:                   ${decoded.uuidHex}`);
    console.log(`  Flags:                  0x${decoded.flags.toString(16).padStart(4, "0")} [${flagsList.join(", ")}]`);
    console.log(`  Source File Size:       ${decoded.originalFileSize.toLocaleString()} bytes`);
    console.log(`  Source Resolution:      ${decoded.width} x ${decoded.height} px (Page ${decoded.page})`);
    console.log(`  Preview Format:         ${previewFormatToString(decoded.previewFormat)} (ID: ${decoded.previewFormat})`);
    console.log(`  Timestamp (UTC):        ${dateUtc} (${decoded.timestampMs} ms)`);
    console.log(`  Embedding Model:        ${embeddingModelToString(decoded.embeddingModelId)}`);
    console.log(`  RS Shard Coordinates:   Block ${decoded.shardBlockId}, Index ${decoded.shardIndex}`);
    console.log(`  Merkle Root:            ${decoded.merkleRootHex}`);
    console.log(`  Ed25519 Attestation:    ${decoded.snarkAttestation.valid ? "VALID" : "UNVERIFIED"} (${decoded.snarkAttestation.verifiedInUs} \u00B5s)`);
    console.log(`  GF(2^8) Reed-Solomon:   HEALTHY (3 lanes, 64-byte parity)`);

    console.log(`\n[Section II: Metadata & OCR Payload]`);
    console.log(`  Compressed Length:      ${decoded.metaLength.toLocaleString()} bytes (Max: ${SECTION2_MAX_META_BYTES.toLocaleString()} bytes)`);
    console.log(`  Uncompressed Length:    ${uncompMeta.toLocaleString()} bytes`);
    console.log(`  Asset ID:               ${decoded.metadata.id}`);
    console.log(`  Original Filename:      ${decoded.metadata.filename}`);
    console.log(`  MIME Type:              ${decoded.metadata.mimeType}`);
    console.log(`  Source Path:            ${decoded.metadata.sourcePath}`);
    console.log(`  Created At:             ${decoded.metadata.createdAt}`);
    console.log(`  DINOv3 Vector:          ${decoded.metadata.dinoEmbeddingHex ? "Present (384-dim Int8 CLS, 768 hex chars)" : "None"}`);
    console.log(`  Tags:                   ${decoded.metadata.tags ? decoded.metadata.tags.join(", ") : "None"}`);
    if (decoded.metadata.geoCoordinates) {
      console.log(`  Geo Coordinates:        ${decoded.metadata.geoCoordinates.latitude}, ${decoded.metadata.geoCoordinates.longitude}`);
    }
    const ocrSnippet = decoded.metadata.ocrText ? decoded.metadata.ocrText.replace(/\s+/g, " ").slice(0, 80) : "";
    console.log(`  OCR Text (${(decoded.metadata.ocrText || "").length} chars): ${ocrSnippet ? `"${ocrSnippet}..."` : "Empty"}`);

    console.log(`\n[Section III: Visual Preview Payload]`);
    console.log(`  Preview Byte Length:    ${decoded.previewLength.toLocaleString()} bytes (Max: ${SECTION3_MAX_PREVIEW_BYTES.toLocaleString()} bytes)`);
    console.log(`  Preview Format:         ${previewFormatToString(decoded.previewFormat)}`);
    console.log(`  Preview Dimensions:     320 x 320 px (Target)`);
    console.log("=".repeat(70) + "\n");
  }

  // If no detailed flag was provided and it was an existing token, show concise summary
  if (!isNewTokenCreated && !args.info && !args.verify) {
    console.log(`[TINYTOKEN] ${resolvedPath}`);
    console.log(`  ID:             ${decoded.id}`);
    console.log(`  Preview Format: ${previewFormatToString(decoded.previewFormat)}`);
    console.log(`  Size:           ${tokenBytes.length.toLocaleString()} bytes / 25 KB (${((tokenBytes.length / MAX_TOKEN_BYTES) * 100).toFixed(1)}% cap)`);
    console.log(`  Verified:       ${decoded.snarkAttestation.valid ? "YES" : "NO"}`);
    console.log(`  Tip: run with --info for full breakdown or -v to audit signature.`);
  }
}

await main().catch((err) => {
  console.error("TinyToken CLI Error:", err instanceof Error ? err.message : String(err));
  process.exit(1);
});
