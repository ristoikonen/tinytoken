/**
 * Bun Native Image Header Metadata Sniffer, WebP Raster Preview Generator & DINOv3/VQ Tensor Extractor
 * (utils/imagePreview.ts)
 *
 * Uses Bun's native `Blob.image().metadata()` for fast header extraction and `sharp` (libvips) to decode
 * PNG, JPEG, WebP, AVIF, BMP, SVG, and PDF cover layouts into:
 * 1. A lean 320x320 WebP raster preview (~2.5 - 9 KB) for Section III (leaving 15+ KB free in 25 KB token)
 * 2. A true decoded 16x16 RGB pixel grid (256 B VQ codebook indices) for Section I
 * 3. A 224x224 ViT-16 RGB tensor -> 384-byte Int8 DINOv3 semantic embedding for Section II
 */

import sharp from "sharp";
import { EmbeddingModelId, PreviewFormat } from "../models/token";
import {
  extractColorHogFallbackFromRgb224,
  extractRealDinoEmbedding,
} from "./dinoEmbedding";
import {
  VQ_CODEBOOK,
  encodeBytesToVqGrid,
  encodeRgb16x16ToVqGrid,
} from "./vqCodebook";

export interface ImageHeaderInfo {
  width: number;
  height: number;
  colorDepth: number;
  format: PreviewFormat;
  mimeType: string;
}

export interface RasterizedAssetBundle {
  width: number;
  height: number;
  colorDepth: number;
  previewBytes: Uint8Array<ArrayBuffer>;
  previewFormat: PreviewFormat;
  vqIndices: Uint8Array<ArrayBuffer>;
  dinoEmbedding: Int8Array<ArrayBuffer>;
  embeddingModelId: EmbeddingModelId;
  svgExtractedText?: string;
  timing?: {
    sharpNs: number;
    vqNs: number;
    dinoNs: number;
  };
}

export function previewFormatToMime(format: PreviewFormat): string {
  switch (format) {
    case PreviewFormat.AVIF:
      return "image/avif";
    case PreviewFormat.WEBP:
      return "image/webp";
    case PreviewFormat.JPEG:
      return "image/jpeg";
    case PreviewFormat.PNG:
      return "image/png";
    case PreviewFormat.BMP:
      return "image/bmp";
    case PreviewFormat.SVG:
      return "image/svg+xml";
    default:
      return "application/octet-stream";
  }
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
      return "SVG-VQ";
    default:
      return "RAW";
  }
}

/**
 * Extracts SVG dimensions (`viewBox` or `width`/`height`) and any embedded `<text>`/`<title>` strings.
 */
export function parseSvgMetadataAndText(svgBytes: Uint8Array): {
  width: number;
  height: number;
  extractedText: string;
} {
  const text = new TextDecoder("utf-8", { fatal: false }).decode(
    svgBytes.subarray(0, Math.min(svgBytes.length, 65536)),
  );

  let width = 220;
  let height = 220;
  const viewBoxMatch = text.match(
    /viewBox\s*=\s*["']\s*([-\d.]+)\s+([-\d.]+)\s+([\d.]+)\s+([\d.]+)\s*["']/i,
  );
  if (viewBoxMatch) {
    const vw = Number.parseFloat(viewBoxMatch[3] ?? "220");
    const vh = Number.parseFloat(viewBoxMatch[4] ?? "220");
    if (vw > 0 && vh > 0) {
      width = Math.round(vw);
      height = Math.round(vh);
    }
  } else {
    const wMatch = text.match(/\bwidth\s*=\s*["']\s*([\d.]+)/i);
    const hMatch = text.match(/\bheight\s*=\s*["']\s*([\d.]+)/i);
    if (wMatch?.[1] && hMatch?.[1]) {
      width = Math.max(1, Math.round(Number.parseFloat(wMatch[1])));
      height = Math.max(1, Math.round(Number.parseFloat(hMatch[1])));
    }
  }

  const textNodes: string[] = [];
  const tagRegex = /<(?:text|tspan|title|desc)\b[^>]*>([^<]+)<\/(?:text|tspan|title|desc)>/gi;
  let m: RegExpExecArray | null = null;
  while ((m = tagRegex.exec(text)) !== null && textNodes.length < 40) {
    const cleaned = (m[1] ?? "").replace(/\s+/g, " ").trim();
    if (cleaned.length > 0) textNodes.push(cleaned);
  }

  return {
    width,
    height,
    extractedText: textNodes.join(" ").slice(0, 400),
  };
}

/**
 * Maps a file extension to its standard MIME type.
 * Covers all Bun metadata-recognizable image formats (JPEG, PNG, WebP, AVIF, BMP, GIF, TIFF, HEIC) plus SVG and documents.
 */
export function extensionToMimeType(ext: string): string {
  const clean = ext.startsWith(".") ? ext.toLowerCase() : `.${ext.toLowerCase()}`;
  switch (clean) {
    case ".jpg":
    case ".jpeg":
      return "image/jpeg";
    case ".png":
      return "image/png";
    case ".webp":
      return "image/webp";
    case ".avif":
      return "image/avif";
    case ".bmp":
      return "image/bmp";
    case ".gif":
      return "image/gif";
    case ".tiff":
    case ".tif":
      return "image/tiff";
    case ".heic":
    case ".heif":
      return "image/heic";
    case ".svg":
      return "image/svg+xml";
    case ".pdf":
      return "application/pdf";
    case ".docx":
      return "application/vnd.openxmlformats-officedocument.wordprocessingml.document";
    case ".doc":
      return "application/msword";
    case ".txt":
      return "text/plain";
    case ".md":
      return "text/markdown";
    default:
      return "application/octet-stream";
  }
}

/**
 * Maps Bun's `image.metadata().format` string or a PreviewFormat enum to its standard MIME type.
 */
export function formatToMimeType(format: string | PreviewFormat): string {
  if (typeof format === "number") {
    switch (format) {
      case PreviewFormat.PNG:
        return "image/png";
      case PreviewFormat.JPEG:
        return "image/jpeg";
      case PreviewFormat.WEBP:
        return "image/webp";
      case PreviewFormat.AVIF:
        return "image/avif";
      case PreviewFormat.BMP:
        return "image/bmp";
      case PreviewFormat.SVG:
        return "image/svg+xml";
      default:
        return "application/octet-stream";
    }
  }
  const f = String(format).toLowerCase().trim();
  switch (f) {
    case "jpg":
    case "jpeg":
      return "image/jpeg";
    case "png":
      return "image/png";
    case "webp":
      return "image/webp";
    case "avif":
      return "image/avif";
    case "bmp":
      return "image/bmp";
    case "gif":
      return "image/gif";
    case "tiff":
      return "image/tiff";
    case "heic":
      return "image/heic";
    case "svg":
      return "image/svg+xml";
    default:
      return "image/jpeg";
  }
}

/**
 * Inspects image dimensions and format using Bun's native `Blob.image().metadata()`
 * for all supported binary formats (JPEG, PNG, WebP, AVIF, BMP, GIF) and SVG XML parsing for SVG.
 */
export async function sniffImageHeader(
  bytes: Uint8Array,
  extHint = "",
): Promise<ImageHeaderInfo> {
  const lowerExt = extHint.toLowerCase();
  if (lowerExt === ".svg") {
    const svgInfo = parseSvgMetadataAndText(bytes);
    return {
      width: svgInfo.width,
      height: svgInfo.height,
      colorDepth: 24,
      format: PreviewFormat.SVG,
      mimeType: "image/svg+xml",
    };
  }

  try {
    const blob = new Blob([bytes]);
    if (typeof (blob as any).image === "function") {
      const meta = await (blob as any).image().metadata();
      if (meta && meta.width && meta.height) {
        const fmtStr = String(meta.format || "").toLowerCase();
        let format = PreviewFormat.JPEG;
        if (fmtStr === "png") format = PreviewFormat.PNG;
        else if (fmtStr === "webp") format = PreviewFormat.WEBP;
        else if (fmtStr === "avif") format = PreviewFormat.AVIF;
        else if (fmtStr === "bmp") format = PreviewFormat.BMP;

        return {
          width: meta.width,
          height: meta.height,
          colorDepth: 24,
          format,
          mimeType: formatToMimeType(fmtStr || extHint),
        };
      }
    }
  } catch {
    // Edge-case fallback if Bun native image parser throws on corrupted payload
  }

  return {
    width: 1024,
    height: 768,
    colorDepth: 24,
    format: PreviewFormat.JPEG,
    mimeType: extensionToMimeType(extHint),
  };
}

/**
 * Downsamples any raw RGB image buffer (e.g. 384x384 or proportional) to a 16x16 raw RGB grid (768 bytes)
 * using pure Bun typed-array 24x24 box averaging without invoking Sharp or libvips.
 */
export function boxDownsampleRgbTo16x16(
  rawRgb: Uint8Array,
  width: number,
  height: number,
): Uint8Array {
  const out = new Uint8Array(16 * 16 * 3);
  const blockW = width / 16;
  const blockH = height / 16;

  for (let gy = 0; gy < 16; gy++) {
    const yStart = Math.floor(gy * blockH);
    const yEnd = Math.max(yStart + 1, Math.floor((gy + 1) * blockH));
    for (let gx = 0; gx < 16; gx++) {
      const xStart = Math.floor(gx * blockW);
      const xEnd = Math.max(xStart + 1, Math.floor((gx + 1) * blockW));

      let rSum = 0;
      let gSum = 0;
      let bSum = 0;
      let count = 0;

      for (let y = yStart; y < yEnd; y++) {
        const rowOffset = y * width * 3;
        for (let x = xStart; x < xEnd; x++) {
          const idx = rowOffset + x * 3;
          rSum += rawRgb[idx]!;
          gSum += rawRgb[idx + 1]!;
          bSum += rawRgb[idx + 2]!;
          count++;
        }
      }

      const outIdx = (gy * 16 + gx) * 3;
      out[outIdx] = Math.round(rSum / count);
      out[outIdx + 1] = Math.round(gSum / count);
      out[outIdx + 2] = Math.round(bSum / count);
    }
  }
  return out;
}

/**
 * Decodes any image or SVG (or synthesized document cover) via `sharp` into:
 * - A lean 220x220 WebP preview (`PreviewFormat.WEBP`, typically 2.5 - 9 KB)
 * - A true decoded 16x16 RGB VQ codebook grid (`256 B`)
 * - A 224x224 RGB ViT-16 tensor -> 384-byte `Int8Array(384)` DINOv3 embedding
 */
export async function rasterizeAssetToWebpAndTensors(
  sourceBytes: Uint8Array,
  ext: string,
  label: string,
  page = 1,
  totalPages = 1,
  docDimensions?: { width: number; height: number; excerpt?: string },
  maxPreviewBytes = 16384,
): Promise<RasterizedAssetBundle> {
  const lowerExt = ext.toLowerCase();
  const isText =
    lowerExt === ".txt" ||
    lowerExt === ".text" ||
    lowerExt === ".md" ||
    lowerExt === ".markdown" ||
    lowerExt === ".csv" ||
    lowerExt === ".json" ||
    lowerExt === ".log" ||
    lowerExt === ".yaml" ||
    lowerExt === ".yml";
  const isBinaryDoc = lowerExt === ".pdf" || lowerExt === ".docx" || lowerExt === ".doc";
  const isDocument = isBinaryDoc || isText;

  let svgExtractedText: string | undefined;
  let inputBuffer: Uint8Array = sourceBytes;
  let fallbackHeader = await sniffImageHeader(sourceBytes, lowerExt);

  if (lowerExt === ".svg") {
    const svgMeta = parseSvgMetadataAndText(sourceBytes);
    svgExtractedText = svgMeta.extractedText;
    fallbackHeader = {
      width: svgMeta.width,
      height: svgMeta.height,
      colorDepth: 24,
      format: PreviewFormat.WEBP,
      mimeType: "image/svg+xml",
    };
  } else if (isText) {
    const rawText =
      docDimensions?.excerpt ||
      new TextDecoder("utf-8", { fatal: false }).decode(sourceBytes.subarray(0, 4096));
    const lines = rawText.split(/\r?\n/).slice(0, 13);
    const escapedLines = lines.map((l) =>
      l
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .slice(0, 36),
    );

    const safeTitle = label.replace(/[<>&"']/g, "").slice(0, 26);
    const lineSvg = escapedLines
      .map(
        (l, i) =>
          `<text x="48" y="${98 + i * 14}" fill="#334155" font-family="monospace" font-size="10">${l || " "}</text>`,
      )
      .join("\n");

    const textDocSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">
      <defs>
        <linearGradient id="textBg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#0f172a"/>
          <stop offset="100%" stop-color="#1e293b"/>
        </linearGradient>
      </defs>
      <rect width="320" height="320" fill="url(#textBg)"/>
      <rect x="36" y="16" width="248" height="288" rx="6" fill="#f8fafc" stroke="#94a3b8" stroke-width="1.5"/>
      <rect x="36" y="16" width="248" height="34" rx="6" fill="#0284c7"/>
      <text x="48" y="38" fill="#ffffff" font-family="sans-serif" font-size="11" font-weight="bold">${lowerExt.replace(".", "").toUpperCase()} • ${totalPages} PAGE${totalPages > 1 ? "S" : ""}</text>
      <text x="48" y="70" fill="#0f172a" font-family="sans-serif" font-size="11.5" font-weight="bold">${safeTitle}</text>
      <line x1="48" y1="80" x2="270" y2="80" stroke="#cbd5e1" stroke-width="1"/>
      ${lineSvg}
    </svg>`;

    inputBuffer = new TextEncoder().encode(textDocSvg);
    fallbackHeader = {
      width: docDimensions?.width ?? 1240,
      height: docDimensions?.height ?? 1754,
      colorDepth: 24,
      format: PreviewFormat.WEBP,
      mimeType: extensionToMimeType(lowerExt),
    };
  } else if (isBinaryDoc) {
    const w = docDimensions?.width ?? 1240;
    const h = docDimensions?.height ?? 1754;
    const hash = Bun.hash.crc32(sourceBytes);
    const hueA = hash % 360;
    const hueB = (hash >> 8) % 360;
    const safeTitle = label.replace(/[<>&"']/g, "").slice(0, 26);
    const safeSub = (docDimensions?.excerpt ?? "PDF Document")
      .replace(/[<>&"']/g, "")
      .slice(0, 32);

    const docCoverSvg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 320 320" width="320" height="320">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="hsl(${hueA}, 55%, 18%)"/>
          <stop offset="100%" stop-color="hsl(${hueB}, 65%, 28%)"/>
        </linearGradient>
      </defs>
      <rect width="320" height="320" fill="url(#bg)"/>
      <rect x="38" y="20" width="244" height="280" rx="8" fill="#f8fafc" stroke="#38bdf8" stroke-width="2.5"/>
      <rect x="38" y="20" width="244" height="52" rx="8" fill="hsl(${hueA}, 70%, 35%)"/>
      <text x="52" y="52" fill="#ffffff" font-family="sans-serif" font-size="15" font-weight="bold">DOC • ${totalPages} PAGES</text>
      <rect x="54" y="92" width="174" height="10" rx="4" fill="#334155"/>
      <rect x="54" y="114" width="206" height="7" rx="3" fill="#64748b"/>
      <rect x="54" y="132" width="194" height="7" rx="3" fill="#64748b"/>
      <rect x="54" y="150" width="160" height="7" rx="3" fill="#94a3b8"/>
      <rect x="54" y="178" width="206" height="52" rx="6" fill="hsl(${hueB}, 45%, 88%)"/>
      <text x="52" y="256" fill="#0f172a" font-family="monospace" font-size="12.5" font-weight="bold">${safeTitle}</text>
      <text x="52" y="278" fill="#475569" font-family="sans-serif" font-size="11.5">${safeSub}</text>
    </svg>`;
    inputBuffer = new TextEncoder().encode(docCoverSvg);
    fallbackHeader = {
      width: w,
      height: h,
      colorDepth: 24,
      format: PreviewFormat.WEBP,
      mimeType: "application/pdf",
    };
  }

  try {
    const pipeline = sharp(inputBuffer, { failOn: "none" });
    const meta = await pipeline.metadata();
    const isRotated90or270 = (meta.orientation ?? 1) >= 5 && (meta.orientation ?? 1) <= 8;
    const width = isDocument
      ? (docDimensions?.width ?? 1240)
      : (isRotated90or270 ? (meta.height ?? fallbackHeader.height) : (meta.width ?? fallbackHeader.width));
    const height = isDocument
      ? (docDimensions?.height ?? 1754)
      : (isRotated90or270 ? (meta.width ?? fallbackHeader.width) : (meta.height ?? fallbackHeader.height));

    const tSharp0 = Bun.nanoseconds();

    // 1. Single decode: decode source image once to 384x384 raw RGB buffer with cubic resampling
    const dinoRgbResult = await sharp(inputBuffer, { failOn: "none" })
      .rotate()
      .flatten({ background: { r: 255, g: 255, b: 255 } })
      .resize(384, 384, { fit: "inside", withoutEnlargement: true, kernel: "cubic" })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    // 2. WebP preview encoded directly from the 384x384 raw RGB buffer (effort: 2, smartSubsample: false)
    let finalWebp = await sharp(dinoRgbResult.data, {
      raw: {
        width: dinoRgbResult.info.width,
        height: dinoRgbResult.info.height,
        channels: 3,
      },
    })
      .resize(320, 320, { fit: "inside" })
      .webp({ quality: 78, smartSubsample: false, effort: 2 })
      .toBuffer();

    if (finalWebp.byteLength > maxPreviewBytes) {
      finalWebp = await sharp(dinoRgbResult.data, {
        raw: {
          width: dinoRgbResult.info.width,
          height: dinoRgbResult.info.height,
          channels: 3,
        },
      })
        .resize(240, 240, { fit: "inside", withoutEnlargement: true })
        .webp({ quality: 68, smartSubsample: false, effort: 2 })
        .toBuffer();
    }
    const sharpNs = Bun.nanoseconds() - tSharp0;

    const previewBytes = new Uint8Array(new ArrayBuffer(finalWebp.byteLength));
    previewBytes.set(new Uint8Array(finalWebp.buffer, finalWebp.byteOffset, finalWebp.byteLength));

    // 3. 16x16 extraction via pure Bun typed-array 24x24 box average (no Sharp enc/dec)
    const tVq0 = Bun.nanoseconds();
    const rgb16Buf = boxDownsampleRgbTo16x16(
      dinoRgbResult.data,
      dinoRgbResult.info.width,
      dinoRgbResult.info.height,
    );
    const vqIndices = encodeRgb16x16ToVqGrid(rgb16Buf);
    const vqNs = Bun.nanoseconds() - tVq0;

    const tDino0 = Bun.nanoseconds();
    const dinoRes = await extractRealDinoEmbedding(
      new Uint8Array(
        dinoRgbResult.data.buffer,
        dinoRgbResult.data.byteOffset,
        dinoRgbResult.data.byteLength,
      ),
      dinoRgbResult.info.width,
      dinoRgbResult.info.height,
    );
    const dinoNs = Bun.nanoseconds() - tDino0;

    return {
      width,
      height,
      colorDepth: fallbackHeader.colorDepth,
      previewBytes,
      previewFormat: PreviewFormat.WEBP,
      vqIndices,
      dinoEmbedding: dinoRes.embedding,
      embeddingModelId: dinoRes.modelId,
      svgExtractedText,
      timing: {
        sharpNs,
        vqNs,
        dinoNs,
      },
    };
  } catch {
    // Fallback if input is non-image raw buffer
    const tVq0 = Bun.nanoseconds();
    const vqIndices = encodeBytesToVqGrid(sourceBytes, fallbackHeader.width, fallbackHeader.height, page);
    const vqNs = Bun.nanoseconds() - tVq0;

    const tSharp0 = Bun.nanoseconds();
    const { previewBytes, previewFormat } = buildCompactPreviewPayload(
      sourceBytes,
      fallbackHeader,
      vqIndices,
      label,
      page,
      maxPreviewBytes,
    );
    const sharpNs = Bun.nanoseconds() - tSharp0;

    const tDino0 = Bun.nanoseconds();
    const dinoEmbedding = extractColorHogFallbackFromRgb224(sourceBytes);
    const dinoNs = Bun.nanoseconds() - tDino0;

    return {
      width: fallbackHeader.width,
      height: fallbackHeader.height,
      colorDepth: fallbackHeader.colorDepth,
      previewBytes,
      previewFormat,
      vqIndices,
      dinoEmbedding,
      embeddingModelId: EmbeddingModelId.COLOR_HOG_FALLBACK,
      svgExtractedText,
      timing: {
        sharpNs,
        vqNs,
        dinoNs,
      },
    };
  }
}

/**
 * Synchronous fallback preview builder used by unit tests and mock token generators.
 */
export function buildCompactPreviewPayload(
  sourceBytes: Uint8Array,
  headerInfo: ImageHeaderInfo,
  vqGrid: Uint8Array,
  label: string,
  page: number,
  maxBytes = 16384,
): { previewBytes: Uint8Array<ArrayBuffer>; previewFormat: PreviewFormat } {
  if (
    sourceBytes.length > 0 &&
    sourceBytes.length <= maxBytes &&
    (headerInfo.format === PreviewFormat.AVIF ||
      headerInfo.format === PreviewFormat.WEBP ||
      headerInfo.format === PreviewFormat.JPEG ||
      headerInfo.format === PreviewFormat.PNG)
  ) {
    const out = new Uint8Array(new ArrayBuffer(sourceBytes.byteLength));
    out.set(sourceBytes);
    return {
      previewBytes: out,
      previewFormat: headerInfo.format,
    };
  }

  const c0 = VQ_CODEBOOK[vqGrid[0] ?? 0] ?? { r: 20, g: 33, b: 61 };
  const c1 = VQ_CODEBOOK[vqGrid[85] ?? 0] ?? { r: 18, g: 78, b: 120 };
  const c2 = VQ_CODEBOOK[vqGrid[170] ?? 0] ?? { r: 56, g: 189, b: 248 };
  const c3 = VQ_CODEBOOK[vqGrid[255] ?? 0] ?? { r: 247, g: 133, b: 34 };

  const safeLabel = label.replace(/[<>&"']/g, "").slice(0, 22);

  let cellsSvg = "";
  for (let gy = 0; gy < 8; gy++) {
    for (let gx = 0; gx < 8; gx++) {
      const vqIdx = vqGrid[gy * 2 * 16 + gx * 2] ?? 0;
      const c = VQ_CODEBOOK[vqIdx] ?? { r: 20, g: 33, b: 61 };
      const hex = `#${c.r.toString(16).padStart(2, "0")}${c.g.toString(16).padStart(2, "0")}${c.b.toString(16).padStart(2, "0")}`;
      cellsSvg += `<rect x="${gx * 27.5}" y="${gy * 27.5}" width="28" height="28" fill="${hex}" fill-opacity="0.72"/>`;
    }
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 220 220" width="220" height="220">
  <defs>
    <linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="rgb(${c0.r},${c0.g},${c0.b})"/>
      <stop offset="50%" stop-color="rgb(${c1.r},${c1.g},${c1.b})"/>
      <stop offset="100%" stop-color="rgb(${c2.r},${c2.g},${c2.b})"/>
    </linearGradient>
  </defs>
  <rect width="220" height="220" fill="url(#g)"/>
  <g>${cellsSvg}</g>
  <rect x="8" y="8" width="204" height="204" rx="10" fill="none" stroke="rgb(${c3.r},${c3.g},${c3.b})" stroke-opacity="0.55" stroke-width="1.5"/>
  <rect x="12" y="164" width="196" height="44" rx="6" fill="#14213D" fill-opacity="0.86" stroke="#38bdf8" stroke-opacity="0.4"/>
  <text x="20" y="183" fill="#E5E5E5" font-family="monospace" font-size="11" font-weight="bold">${safeLabel}</text>
  <text x="20" y="199" fill="#38bdf8" font-family="monospace" font-size="9.5">${headerInfo.width}x${headerInfo.height} • P.${page} • GF28+VQ</text>
</svg>`;

  const encoded = new TextEncoder().encode(svg);
  const sliceLen = Math.min(encoded.byteLength, maxBytes);
  const out = new Uint8Array(new ArrayBuffer(sliceLen));
  out.set(encoded.subarray(0, sliceLen));
  return {
    previewBytes: out,
    previewFormat: PreviewFormat.SVG,
  };
}
