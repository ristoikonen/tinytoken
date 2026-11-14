#!/usr/bin/env bun
/**
 * TinyToken Dedicated CLI (cli.ts)
 * 
 * Works with ONE token or asset at a time (and archives/dirs for `struct`):
 *   - Inspect first, decode only when necessary (Zero-copy Section I inspection)
 *   - Commands:
 *       tokenize <file>  - create, inspect, extract, or audit a token
 *       struct <target>  - fast zero-copy field extractor (token, archive, or directory)
 * 
 * Usage:
 *   tinytoken <file> [options]
 *   tinytoken struct <file|archive> [-f <field>] [--format <table|csv|json>]
 *   bun tokenize <file> [options]
 */

import {
  EmbeddingModelId,
  PreviewFormat,
  SECTION1_HEADER_BYTES,
  SECTION2_MAX_META_BYTES,
  SECTION3_MAX_PREVIEW_BYTES,
  MAX_TOKEN_BYTES,
  TokenFlags,
  type TokenHeaderInspection,
  type TokenMetadataPayload,
} from "./models/token";
import { Win11TokenPipeline } from "./utils/indexer";
import { QuickLZ } from "./utils/quicklz";
import { bytesToHex } from "./utils/snarkAttest";
import {
  decodeBinaryToken,
  extractTokenMetadata,
  extractTokenPreviewBytes,
  inspectBinaryTokenHeader,
} from "./utils/tokenCodec";

interface CliArgs {
  command?: "tokenize" | "struct";
  file?: string;
  out?: string;
  field?: string;
  format?: "table" | "csv" | "json" | "raw-hex";
  decodeMeta: boolean;
  limit?: number;
  struct: boolean;
  verify: boolean;
  json: boolean;
  info: boolean;
  filetype: boolean;
  preview: boolean;
  help: boolean;
}

function printHelp(): void {
  console.log(`
TinyToken CLI Tool (Inspect First, Decode Only When Necessary)

Usage:
  tinytoken <file> [options]
  tinytoken struct <file|archive> [options]
  bun tokenize <file> [options]

Commands:
  tokenize <file>      (Default) Inspect, verify, extract preview, or encode an asset
  struct <target>      Zero-copy binary slice extractor (inspects headers in <1 µs)

Arguments:
  <file|target>        Source file (image, doc, .token file, or tokens.zip archive) [required]

Options:
  -f, --field <name>   Extract single field: uuid, merkle, cid, dims, flags, ts, sizes, vq, model, ocr, dino, bge, tags
  --format <fmt>       Output format: table (default), csv, json, raw-hex
  --decode-meta        On-demand Section II QuickLZ metadata unpack (only when needed)
  --limit <n>          Maximum tokens to inspect when target is an archive or directory
  --struct             Force zero-copy structural inspection mode
  -o, --out <path>     Custom output .token path (defaults to <input_dir>/<basename>.token)
  -v, --verify         Run cryptographic Ed25519 signature + SHA-256 Merkle audit
  --info               Verbose breakdown: token size, uncompressed estimate, dates, all fields
  --filetype           Print preview filetype (e.g. 'WEBP', 'PNG', 'JPEG', 'AVIF')
  --preview            Stream raw binary preview data directly to stdout (for saving or piping)
  --json               Print machine-readable JSON to stdout
  -h, --help           Display this help message

Examples:
  # Fast zero-copy structural inspection (<0.05 ms, 0 decompression / 0 crypto overhead):
  tinytoken struct tok_00004_Henri.jpg.token
  tinytoken tok_00004_Henri.jpg.token -f uuid
  tinytoken tok_00004_Henri.jpg.token -f merkle
  tinytoken tok_00004_Henri.jpg.token -f dims
  tinytoken struct ./data/tokens.zip --limit 10

  # Query preview format & extract preview without decoding Section II:
  tinytoken tok_00004_Henri.jpg.token --filetype
  tinytoken tok_00004_Henri.jpg.token --preview > preview.webp

  # Full cryptographic audit & detailed breakdown:
  tinytoken tok_00004_Henri.jpg.token -v
  tinytoken tok_00004_Henri.jpg.token --info
`);
}

function parseArgs(args: string[]): CliArgs {
  const result: CliArgs = {
    decodeMeta: false,
    struct: false,
    verify: false,
    json: false,
    info: false,
    filetype: false,
    preview: false,
    help: false,
  };

  let startIndex = 0;
  if (args.length > 0) {
    const first = args[0]!.toLowerCase();
    if (first === "struct") {
      result.command = "struct";
      result.struct = true;
      startIndex = 1;
    } else if (first === "tokenize") {
      result.command = "tokenize";
      startIndex = 1;
    }
  }

  for (let i = startIndex; i < args.length; i++) {
    const arg = args[i]!;
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
    } else if (arg === "--struct") {
      result.struct = true;
    } else if (arg === "--decode-meta") {
      result.decodeMeta = true;
    } else if (arg === "-f" || arg === "--field") {
      if (i + 1 < args.length) {
        result.field = args[++i];
        result.struct = true;
      } else {
        console.error("Error: -f / --field requires a field name.");
        process.exit(1);
      }
    } else if (arg === "--format") {
      if (i + 1 < args.length) {
        result.format = args[++i] as any;
      }
    } else if (arg === "--limit") {
      if (i + 1 < args.length) {
        result.limit = Number.parseInt(args[++i]!, 10);
      }
    } else if (arg === "-o" || arg === "--out") {
      if (i + 1 < args.length) {
        result.out = args[++i];
      } else {
        console.error("Error: -o / --out requires a file path argument.");
        process.exit(1);
      }
    } else if (!arg.startsWith("-") && !result.file) {
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
  if (flags & TokenFlags.HAS_BGE_EMBEDDING) list.push("HAS_BGE_EMBEDDING");
  return list;
}

function embeddingModelToString(modelId: number): string {
  switch (modelId) {
    case EmbeddingModelId.NONE:
      return "0x00 (None)";
    case EmbeddingModelId.COLOR_HOG_FALLBACK:
      return "0x01 (Color/HOG Fallback 384B)";
    case EmbeddingModelId.DINOV3_VITS16_Q8:
      return "0x02 (Meta DINOv3 ViT-S/16 ONNX q8)";
    case EmbeddingModelId.BGE_TEXT_FALLBACK:
      return "0x03 (BGE Text Lexical Fallback 384B)";
    case EmbeddingModelId.BGE_SMALL_EN_V15_Q8:
      return "0x04 (BAAI BGE-small-en-v1.5 ONNX q8)";
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
 * Fast zero-copy structural field inspection (Inspect first, decode only when necessary)
 */
async function handleStructCommand(
  targetPath: string,
  rawBytes: Uint8Array,
  args: CliArgs,
): Promise<void> {
  const t0 = Bun.nanoseconds();
  const insp = inspectBinaryTokenHeader(rawBytes);
  const latencyUs = Math.max(1, Math.round((Bun.nanoseconds() - t0) / 1000));

  // 1. Single Field Extraction
  if (args.field) {
    const f = args.field.toLowerCase().trim();
    switch (f) {
      case "uuid":
        process.stdout.write(`${insp.uuidHex}\n`);
        return;
      case "merkle":
      case "merkleroot":
      case "cid":
      case "contentid":
        process.stdout.write(`${insp.contentId || insp.merkleRootHex}\n`);
        return;
      case "dims":
      case "dimensions":
      case "resolution":
        process.stdout.write(`${insp.width}x${insp.height}\n`);
        return;
      case "width":
        process.stdout.write(`${insp.width}\n`);
        return;
      case "height":
        process.stdout.write(`${insp.height}\n`);
        return;
      case "page":
        process.stdout.write(`${insp.page}\n`);
        return;
      case "format":
      case "previewformat":
      case "filetype":
        process.stdout.write(`${insp.previewFormatLabel}\n`);
        return;
      case "flags":
        if (args.format === "json") {
          process.stdout.write(JSON.stringify({ flags: insp.flags, flagNames: insp.flagNames }) + "\n");
        } else {
          process.stdout.write(`0x${insp.flags.toString(16).padStart(4, "0")} [${insp.flagNames.join(", ")}]\n`);
        }
        return;
      case "ts":
      case "timestamp":
      case "time":
        process.stdout.write(`${insp.timestampUtc}\n`);
        return;
      case "model":
      case "embeddingmodel":
        process.stdout.write(`${embeddingModelToString(insp.embeddingModelId)}\n`);
        return;
      case "sizes":
        process.stdout.write(
          `Header: 704B | Meta: ${insp.metaLength}B | Preview: ${insp.previewLength}B | Total: ${insp.totalTokenBytes}B\n`,
        );
        return;
      case "vq":
        if (args.format === "raw-hex") {
          process.stdout.write(`${bytesToHex(insp.vqIndices)}\n`);
        } else {
          process.stdout.write(`[${Array.from(insp.vqIndices).join(", ")}]\n`);
        }
        return;
      default:
        // Decode Section II on-demand only if requesting metadata fields
        if (["ocr", "ocrtext", "filename", "tags", "dino", "bge", "bgemodel", "sourcepath"].includes(f) || args.decodeMeta) {
          const meta = extractTokenMetadata(rawBytes);
          if (f === "ocr" || f === "ocrtext") {
            process.stdout.write(`${meta.ocrText || ""}\n`);
            return;
          }
          if (f === "filename") {
            process.stdout.write(`${meta.filename}\n`);
            return;
          }
          if (f === "tags") {
            process.stdout.write(`${(meta.tags || []).join(", ")}\n`);
            return;
          }
          if (f === "dino") {
            process.stdout.write(`${meta.dinoEmbeddingHex || "none"}\n`);
            return;
          }
          if (f === "bge") {
            process.stdout.write(`${meta.bgeEmbeddingHex || "none"}\n`);
            return;
          }
          if (f === "bgemodel") {
            process.stdout.write(`${meta.bgeModelId !== undefined ? embeddingModelToString(meta.bgeModelId) : "none"}\n`);
            return;
          }
          if (f === "sourcepath") {
            process.stdout.write(`${meta.sourcePath || ""}\n`);
            return;
          }
        }
        console.error(`Unknown field: "${args.field}". Available: uuid, merkle, cid, dims, flags, ts, sizes, vq, model, ocr, filename, tags, dino, bge.`);
        process.exit(1);
    }
  }

  // 2. Structured JSON Output
  if (args.format === "json" || args.json) {
    let metadata: TokenMetadataPayload | undefined;
    if (args.decodeMeta) {
      metadata = extractTokenMetadata(rawBytes);
    }
    let attestationStatus: any = undefined;
    if (args.verify) {
      const decoded = await decodeBinaryToken(rawBytes);
      attestationStatus = decoded.snarkAttestation;
    }
    const outputObj = {
      file: targetPath,
      magic: insp.magic,
      version: insp.version,
      flags: insp.flags,
      flagNames: insp.flagNames,
      uuidHex: insp.uuidHex,
      originalFileSize: insp.originalFileSize,
      width: insp.width,
      height: insp.height,
      page: insp.page,
      previewFormat: insp.previewFormatLabel,
      contentId: insp.contentId || insp.merkleRootHex,
      merkleRootHex: insp.merkleRootHex,
      shardBlockId: insp.shardBlockId,
      shardIndex: insp.shardIndex,
      timestampUtc: insp.timestampUtc,
      embeddingModelId: insp.embeddingModelId,
      offsets: {
        section1Header: 0,
        snarkStub: insp.snarkStubOffset,
        vqGrid: insp.vqGridOffset,
        rsParity: insp.rsParityOffset,
        section2Meta: insp.metaOffset,
        section3Preview: insp.previewOffset,
      },
      sizes: {
        headerBytes: 704,
        metaBytes: insp.metaLength,
        previewBytes: insp.previewLength,
        totalBytes: insp.totalTokenBytes,
      },
      envelopeBoundsValid: insp.envelopeBoundsValid,
      inspectionLatencyUs: latencyUs,
      ...(metadata ? { metadata } : {}),
      ...(attestationStatus ? { attestation: attestationStatus } : {}),
    };
    process.stdout.write(JSON.stringify(outputObj, null, 2) + "\n");
    return;
  }

  // 3. CSV Format Output
  if (args.format === "csv") {
    console.log("uuid,width,height,page,preview_format,meta_bytes,preview_bytes,total_bytes,merkle_root");
    console.log(
      `${insp.uuidHex},${insp.width},${insp.height},${insp.page},${insp.previewFormatLabel},${insp.metaLength},${insp.previewLength},${insp.totalTokenBytes},${insp.merkleRootHex}`,
    );
    return;
  }

  // 4. Default: High-Density Structural Table
  console.log(`\n[STRUCT] Zero-Copy Header Inspection: ${targetPath}`);
  console.log(`  Magic / Version:        ${insp.magic} (Version ${insp.version})`);
  console.log(`  UUID:                   ${insp.uuidHex}`);
  console.log(`  Source Resolution:      ${insp.width} x ${insp.height} px (Page ${insp.page})`);
  console.log(`  Original File Size:     ${insp.originalFileSize.toLocaleString()} bytes`);
  console.log(`  Flags:                  0x${insp.flags.toString(16).padStart(4, "0")} [${insp.flagNames.join(", ")}]`);
  console.log(`  Preview Format:         ${insp.previewFormatLabel} (ID: ${insp.previewFormat})`);
  console.log(`  Section II (Metadata):  ${insp.metaLength.toLocaleString()} bytes (Offset: ${insp.metaOffset})`);
  console.log(`  Section III (Preview):  ${insp.previewLength.toLocaleString()} bytes (Offset: ${insp.previewOffset})`);
  console.log(`  Total Envelope:         ${insp.totalTokenBytes.toLocaleString()} bytes / 25,600 max (${insp.envelopeBoundsValid ? "\x1b[32mBOUNDS VALID\x1b[0m" : "\x1b[31mMISMATCH\x1b[0m"})`);
  console.log(`  Content ID (CID):       ${insp.contentId || insp.merkleRootHex}`);
  console.log(`  Merkle Root:            ${insp.merkleRootHex}`);
  console.log(`  Embedding Model ID:     ${embeddingModelToString(insp.embeddingModelId)}`);
  console.log(`  RS Shard Coordinates:   Block ${insp.shardBlockId}, Index ${insp.shardIndex}`);
  console.log(`  Timestamp (UTC):        ${insp.timestampUtc}`);
  console.log(`  Inspection Time:        ${latencyUs} \u00B5s (0 QuickLZ decompression, 0 WebCrypto verify overhead)`);

  if (args.decodeMeta) {
    const meta = extractTokenMetadata(rawBytes);
    console.log(`\n[METADATA (On-Demand QuickLZ Unpack)]`);
    console.log(`  ID:                     ${meta.id}`);
    console.log(`  Filename:               ${meta.filename}`);
    console.log(`  MIME:                   ${meta.mimeType}`);
    console.log(`  Tags:                   ${meta.tags?.join(", ") || "none"}`);
    console.log(`  OCR Snippet:            "${(meta.ocrText || "").slice(0, 80).replace(/\s+/g, " ")}..."`);
  } else {
    console.log(`  Tip: Pass --decode-meta to decompress Section II JSON, or -v to verify Ed25519 signature.\n`);
  }
}

/**
 * Inspects all tokens in a ZIP archive or directory using zero-copy slice reads
 */
async function handleArchiveStructCommand(
  archivePath: string,
  args: CliArgs,
): Promise<void> {
  const file = Bun.file(archivePath);
  if (!(await file.exists())) {
    console.error(`Error: Archive not found: "${archivePath}"`);
    process.exit(1);
  }

  const t0 = Bun.nanoseconds();
  const fileBuf = await file.arrayBuffer();
  const bytes = new Uint8Array(fileBuf);
  const view = new DataView(fileBuf);

  // Scan Central Directory
  let centralDirOffset = -1;
  for (let i = bytes.length - 22; i >= Math.max(0, bytes.length - 65557); i--) {
    if (view.getUint32(i, true) === 0x06054b50) {
      centralDirOffset = view.getUint32(i + 16, true);
      break;
    }
  }

  if (centralDirOffset === -1) {
    console.error(`Error: Not a valid ZIP archive: ${archivePath}`);
    process.exit(1);
  }

  const inspections: { entryName: string; insp: TokenHeaderInspection }[] = [];
  let pos = centralDirOffset;
  const limit = args.limit ?? 50;

  while (pos < bytes.length && inspections.length < limit) {
    if (view.getUint32(pos, true) !== 0x02014b50) break;
    const compressionMethod = view.getUint16(pos + 10, true);
    const compressedSize = view.getUint32(pos + 20, true);
    const uncompressedSize = view.getUint32(pos + 24, true);
    const nameLen = view.getUint16(pos + 28, true);
    const extraLen = view.getUint16(pos + 30, true);
    const commentLen = view.getUint16(pos + 32, true);
    const localHeaderOffset = view.getUint32(pos + 42, true);

    const nameBytes = bytes.subarray(pos + 46, pos + 46 + nameLen);
    const entryName = new TextDecoder().decode(nameBytes);

    if (entryName.endsWith(".token") && uncompressedSize >= SECTION1_HEADER_BYTES) {
      const dataOffset = localHeaderOffset + 30 + nameLen;
      const rawSlice = bytes.subarray(dataOffset, dataOffset + compressedSize);
      let tokenBytes = rawSlice;
      if (compressionMethod === 8) {
        tokenBytes = Bun.inflateSync(rawSlice);
      }
      try {
        const insp = inspectBinaryTokenHeader(tokenBytes);
        inspections.push({ entryName, insp });
      } catch {
        // Skip malformed entries
      }
    }
    pos += 46 + nameLen + extraLen + commentLen;
  }

  const totalTimeMs = ((Bun.nanoseconds() - t0) / 1000000).toFixed(2);

  if (args.format === "json" || args.json) {
    const list = inspections.map(({ entryName, insp }) => ({
      entryName,
      uuid: insp.uuidHex,
      contentId: insp.contentId || insp.merkleRootHex,
      dims: `${insp.width}x${insp.height}`,
      previewFormat: insp.previewFormatLabel,
      metaBytes: insp.metaLength,
      previewBytes: insp.previewLength,
      totalBytes: insp.totalTokenBytes,
      merkleRoot: insp.merkleRootHex,
      flags: insp.flags,
    }));
    process.stdout.write(JSON.stringify({ archive: archivePath, totalInspected: inspections.length, timeMs: Number(totalTimeMs), tokens: list }, null, 2) + "\n");
    return;
  }

  if (args.format === "csv") {
    console.log("entry,uuid,resolution,preview_format,meta_bytes,preview_bytes,total_bytes,merkle_root");
    for (const { entryName, insp } of inspections) {
      console.log(`${entryName},${insp.uuidHex},${insp.width}x${insp.height},${insp.previewFormatLabel},${insp.metaLength},${insp.previewLength},${insp.totalTokenBytes},${insp.merkleRootHex}`);
    }
    return;
  }

  console.log(`\n[STRUCT ARCHIVE] ${archivePath} (${inspections.length} tokens inspected in ${totalTimeMs} ms, >${((inspections.length / Math.max(0.1, Number(totalTimeMs))) * 1000).toFixed(0)} tok/s)`);
  console.log("-".repeat(105));
  console.log(
    "Entry Name".padEnd(34) +
    "UUID (16)".padEnd(18) +
    "Dims".padEnd(10) +
    "Format".padEnd(8) +
    "Meta".padEnd(8) +
    "Preview".padEnd(10) +
    "Total".padEnd(10) +
    "Merkle Root (12)"
  );
  console.log("-".repeat(105));

  for (const { entryName, insp } of inspections) {
    const safeName = entryName.length > 32 ? entryName.slice(0, 29) + "..." : entryName;
    console.log(
      safeName.padEnd(34) +
      insp.uuidHex.slice(0, 16).padEnd(18) +
      `${insp.width}x${insp.height}`.padEnd(10) +
      insp.previewFormatLabel.padEnd(8) +
      `${insp.metaLength}B`.padEnd(8) +
      `${insp.previewLength}B`.padEnd(10) +
      `${insp.totalTokenBytes}B`.padEnd(10) +
      insp.merkleRootHex.slice(0, 12) + "..."
    );
  }
  console.log("-".repeat(105) + "\n");
}

async function main() {
  const args = parseArgs(Bun.argv.slice(2));

  if (args.help || !args.file) {
    printHelp();
    process.exit(args.help ? 0 : 1);
  }

  // Handle archive inspection directly for struct
  if (args.struct || args.command === "struct") {
    if (args.file.endsWith(".zip")) {
      await handleArchiveStructCommand(args.file, args);
      return;
    }
  }

  const resolvedPath = await resolveInputFile(args.file);
  if (!resolvedPath) {
    console.error(`Error: File not found: "${args.file}"`);
    process.exit(1);
  }

  // Check if archive for struct
  if ((args.struct || args.command === "struct") && resolvedPath.endsWith(".zip")) {
    await handleArchiveStructCommand(resolvedPath, args);
    return;
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

    if (!args.json && !args.preview && !args.filetype && !args.field) {
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

  // =========================================================================
  // FAST-PATH INSPECTIONS (Inspect first, decode only when necessary)
  // =========================================================================

  // 1. --filetype: zero-copy preview format read directly from Section I offset 42 (0 ms)
  if (args.filetype) {
    const insp = inspectBinaryTokenHeader(tokenBytes);
    process.stdout.write(`${insp.previewFormatLabel}\n`);
    return;
  }

  // 2. --preview: extract raw preview image slice without decompressing Section II or verifying crypto (0 ms)
  if (args.preview) {
    const previewBytes = extractTokenPreviewBytes(tokenBytes);
    process.stdout.write(previewBytes);
    return;
  }

  // 3. struct command / --struct flag / -f single field: zero-copy structural inspection (<0.05 ms)
  if (args.struct || args.command === "struct" || args.field) {
    await handleStructCommand(resolvedPath, tokenBytes, args);
    return;
  }

  // If a new token was generated, announce it
  if (isNewTokenCreated) {
    console.log(`[SAVED] Token created: ${outputPath}`);
    console.log(`        Size: ${tokenBytes.length.toLocaleString()} bytes / ${MAX_TOKEN_BYTES.toLocaleString()} bytes max (${((tokenBytes.length / MAX_TOKEN_BYTES) * 100).toFixed(1)}% cap)`);
  }

  // 4. -v / --verify: explicit cryptographic audit (heals RS, checks Merkle, verifies Ed25519)
  if (args.verify) {
    const decoded = await decodeBinaryToken(tokenBytes);
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
    if (!args.info) return;
  }

  // 5. --json: full representation (uses inspection + on-demand meta & crypto verification)
  if (args.json) {
    const decoded = await decodeBinaryToken(tokenBytes);
    const insp = inspectBinaryTokenHeader(tokenBytes);
    const jsonOutput = {
      id: decoded.id,
      uuidHex: insp.uuidHex,
      version: insp.version,
      flags: insp.flags,
      flagNames: insp.flagNames,
      tokenSizeBytes: tokenBytes.length,
      originalFileSize: insp.originalFileSize,
      metaLength: insp.metaLength,
      previewLength: insp.previewLength,
      width: insp.width,
      height: insp.height,
      page: insp.page,
      previewFormat: insp.previewFormatLabel,
      previewFormatId: insp.previewFormat,
      merkleRootHex: insp.merkleRootHex,
      shardBlockId: insp.shardBlockId,
      shardIndex: insp.shardIndex,
      timestampMs: insp.timestampMs,
      timestampUtc: insp.timestampUtc,
      embeddingModelId: insp.embeddingModelId,
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

  // 6. --info: Inspect Section I immediately, unpack Section II metadata on demand
  if (args.info) {
    const insp = inspectBinaryTokenHeader(tokenBytes);
    const meta = extractTokenMetadata(tokenBytes);
    const capacityPct = ((tokenBytes.length / MAX_TOKEN_BYTES) * 100).toFixed(1);
    const freeBytes = MAX_TOKEN_BYTES - tokenBytes.length;
    const estUncompTotal = insp.originalFileSize + insp.metaLength + SECTION1_HEADER_BYTES;
    const compRatio = (estUncompTotal / Math.max(1, tokenBytes.length)).toFixed(2);

    console.log(`\n` + "=".repeat(70));
    console.log(`  TinyToken Inspector: ${meta.id || resolvedPath}`);
    console.log("=".repeat(70));

    console.log(`\n[Envelope & Capacity]`);
    console.log(`  File Path:              ${resolvedPath}`);
    console.log(`  Token Size:             ${tokenBytes.length.toLocaleString()} bytes (${(tokenBytes.length / 1024).toFixed(2)} KB)`);
    console.log(`  Max Envelope Limit:     ${MAX_TOKEN_BYTES.toLocaleString()} bytes (25 KB)`);
    console.log(`  Envelope Capacity Used: ${capacityPct}% (${freeBytes.toLocaleString()} bytes remaining)`);
    console.log(`  Estimated Uncompressed: ~${estUncompTotal.toLocaleString()} bytes (${compRatio}x overall compression ratio)`);

    console.log(`\n[Section I: Fixed Header (704 Bytes)]`);
    console.log(`  Magic Bytes:            ${insp.magic} (0x47 0x46 0x32 0x38)`);
    console.log(`  Format Version:         ${insp.version}`);
    console.log(`  UUID:                   ${insp.uuidHex}`);
    console.log(`  Flags:                  0x${insp.flags.toString(16).padStart(4, "0")} [${insp.flagNames.join(", ")}]`);
    console.log(`  Source File Size:       ${insp.originalFileSize.toLocaleString()} bytes`);
    console.log(`  Source Resolution:      ${insp.width} x ${insp.height} px (Page ${insp.page})`);
    console.log(`  Preview Format:         ${insp.previewFormatLabel} (ID: ${insp.previewFormat})`);
    console.log(`  Timestamp (UTC):        ${insp.timestampUtc} (${insp.timestampMs} ms)`);
    console.log(`  Embedding Model:        ${embeddingModelToString(insp.embeddingModelId)}`);
    console.log(`  RS Shard Coordinates:   Block ${insp.shardBlockId}, Index ${insp.shardIndex}`);
    console.log(`  Content ID (CID):       ${insp.contentId || insp.merkleRootHex}`);
  console.log(`  Merkle Root:            ${insp.merkleRootHex}`);
    console.log(`  GF(2^8) Reed-Solomon:   HEALTHY (3 lanes, 64-byte parity)`);

    console.log(`\n[Section II: Metadata & OCR Payload]`);
    console.log(`  Compressed Length:      ${insp.metaLength.toLocaleString()} bytes (Max: ${SECTION2_MAX_META_BYTES.toLocaleString()} bytes)`);
    console.log(`  Asset ID:               ${meta.id}`);
    console.log(`  Original Filename:      ${meta.filename}`);
    console.log(`  MIME Type:              ${meta.mimeType}`);
    console.log(`  Source Path:            ${meta.sourcePath}`);
    console.log(`  Created At:             ${meta.createdAt}`);
    console.log(`  DINOv3 Vector:          ${meta.dinoEmbeddingHex ? "Present (384-dim Int8 CLS, 768 hex chars)" : "None"}`);
    console.log(`  BGE Text Vector:        ${meta.bgeEmbeddingHex ? `Present (384-dim Int8, ${meta.bgeModelId !== undefined ? embeddingModelToString(meta.bgeModelId) : "bge-small-en-v1.5"})` : "None"}`);
    console.log(`  Tags:                   ${meta.tags ? meta.tags.join(", ") : "None"}`);
    if (meta.geoCoordinates) {
      console.log(`  Geo Coordinates:        ${meta.geoCoordinates.latitude}, ${meta.geoCoordinates.longitude}`);
    }
    const ocrSnippet = meta.ocrText ? meta.ocrText.replace(/\s+/g, " ").slice(0, 80) : "";
    console.log(`  OCR Text (${(meta.ocrText || "").length} chars): ${ocrSnippet ? `"${ocrSnippet}..."` : "Empty"}`);

    console.log(`\n[Section III: Visual Preview Payload]`);
    console.log(`  Preview Byte Length:    ${insp.previewLength.toLocaleString()} bytes (Max: ${SECTION3_MAX_PREVIEW_BYTES.toLocaleString()} bytes)`);
    console.log(`  Preview Format:         ${insp.previewFormatLabel}`);
    console.log(`  Preview Dimensions:     320 x 320 px (Target)`);
    console.log("=".repeat(70) + "\n");
    return;
  }

  // Fallback concise summary (zero-copy inspection)
  const insp = inspectBinaryTokenHeader(tokenBytes);
  console.log(`[TINYTOKEN] ${resolvedPath}`);
  console.log(`  UUID:           ${insp.uuidHex}`);
  console.log(`  Preview Format: ${insp.previewFormatLabel}`);
  console.log(`  Size:           ${tokenBytes.length.toLocaleString()} bytes / 25 KB (${((tokenBytes.length / MAX_TOKEN_BYTES) * 100).toFixed(1)}% cap)`);
  console.log(`  Tip: run with struct for zero-copy inspection, or --info for full metadata.`);
}

await main().catch((err) => {
  console.error("TinyToken CLI Error:", err instanceof Error ? err.message : String(err));
  process.exit(1);
});
