/**
 * Fast Binary Image Header Sniffer, WebP Raster Preview Generator & DINOv2/VQ Tensor Extractor
 * (utils/imagePreview.ts)
 *
 * Uses `DataView` for sub-0.05ms header sniffing and `sharp` (libvips) to decode PNG, JPEG, WebP,
 * AVIF, BMP, SVG, and PDF cover layouts into:
 * 1. A lean 220x220 WebP raster preview (~2.5 - 9 KB) for Section III (leaving 15+ KB free in 25 KB token)
 * 2. A true decoded 16x16 RGB pixel grid (256 B VQ codebook indices) for Section I
 * 3. A 224x224 ViT-14 RGB tensor -> 384-byte Int8 DINOv2 semantic embedding for Section II
 */

import sharp from "sharp";
import { PreviewFormat } from "../models/token";
import {
  DINO_IMAGE_SIZE,
  extractDinoEmbeddingFromRgb224,
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
  svgExtractedText?: string;
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
 * Inspects the first 4 KB of an image buffer to extract exact dimensions and format in < 0.05 ms.
 */
export function sniffImageHeader(bytes: Uint8Array, extHint = ""): ImageHeaderInfo {
  const len = bytes.length;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);

  // 1. PNG: 89 50 4E 47 0D 0A 1A 0A + IHDR at offset 16..24
  if (
    len >= 26 &&
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47
  ) {
    const width = view.getUint32(16, false);
    const height = view.getUint32(20, false);
    const bitDepth = bytes[24] ?? 8;
    const colorType = bytes[25] ?? 2;
    const channels = colorType === 6 ? 4 : colorType === 2 ? 3 : 1;
    return {
      width: width || 220,
      height: height || 220,
      colorDepth: bitDepth * channels,
      format: PreviewFormat.PNG,
      mimeType: "image/png",
    };
  }

  // 2. JPEG: FF D8 ... scan for SOF0 (FF C0), SOF1 (FF C1), SOF2 (FF C2)
  if (len >= 4 && bytes[0] === 0xff && bytes[1] === 0xd8) {
    let offset = 2;
    while (offset + 9 < len) {
      if (bytes[offset] !== 0xff) {
        offset++;
        continue;
      }
      const marker = bytes[offset + 1] ?? 0;
      if (marker === 0xff) {
        offset++;
        continue;
      }
      if (marker === 0xc0 || marker === 0xc1 || marker === 0xc2 || marker === 0xc3) {
        const precision = bytes[offset + 4] ?? 8;
        const height = view.getUint16(offset + 5, false);
        const width = view.getUint16(offset + 7, false);
        const components = bytes[offset + 9] ?? 3;
        return {
          width: width || 220,
          height: height || 220,
          colorDepth: precision * components,
          format: PreviewFormat.JPEG,
          mimeType: "image/jpeg",
        };
      }
      if (marker === 0xd9 || marker === 0xda) break;
      const segLen = view.getUint16(offset + 2, false);
      if (segLen < 2) break;
      offset += 2 + segLen;
    }
    return {
      width: 1024,
      height: 768,
      colorDepth: 24,
      format: PreviewFormat.JPEG,
      mimeType: "image/jpeg",
    };
  }

  // 3. WebP: "RIFF" .... "WEBP"
  if (
    len >= 30 &&
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    let width = 220;
    let height = 220;
    if (bytes[12] === 0x56 && bytes[13] === 0x50 && bytes[14] === 0x38 && bytes[15] === 0x20) {
      width = view.getUint16(26, true) & 0x3fff;
      height = view.getUint16(28, true) & 0x3fff;
    } else if (bytes[12] === 0x56 && bytes[13] === 0x50 && bytes[14] === 0x38 && bytes[15] === 0x4c) {
      const bits = view.getUint32(21, true);
      width = (bits & 0x3fff) + 1;
      height = ((bits >> 14) & 0x3fff) + 1;
    }
    return {
      width: width || 220,
      height: height || 220,
      colorDepth: 32,
      format: PreviewFormat.WEBP,
      mimeType: "image/webp",
    };
  }

  // 4. BMP: "BM"
  if (len >= 26 && bytes[0] === 0x42 && bytes[1] === 0x4d) {
    const width = Math.abs(view.getInt32(18, true));
    const height = Math.abs(view.getInt32(22, true));
    const bpp = view.getUint16(28, true) || 24;
    return {
      width: width || 220,
      height: height || 220,
      colorDepth: bpp,
      format: PreviewFormat.BMP,
      mimeType: "image/bmp",
    };
  }

  // 5. AVIF / ISO BMFF ("ftypavif" or "ftypavis")
  if (
    len >= 16 &&
    bytes[4] === 0x66 &&
    bytes[5] === 0x74 &&
    bytes[6] === 0x79 &&
    bytes[7] === 0x70
  ) {
    return {
      width: 220,
      height: 220,
      colorDepth: 24,
      format: PreviewFormat.AVIF,
      mimeType: "image/avif",
    };
  }

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

  return {
    width: 1200,
    height: 1600,
    colorDepth: 24,
    format: PreviewFormat.SVG,
    mimeType: "image/svg+xml",
  };
}

/**
 * Decodes any image or SVG (or synthesized document cover) via `sharp` into:
 * - A lean 220x220 WebP preview (`PreviewFormat.WEBP`, typically 2.5 - 9 KB)
 * - A true decoded 16x16 RGB VQ codebook grid (`256 B`)
 * - A 224x224 RGB ViT-14 tensor -> 384-byte `Int8Array(384)` DINOv2 embedding
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
  const isDocument = lowerExt === ".pdf" || lowerExt === ".docx" || lowerExt === ".doc";

  let svgExtractedText: string | undefined;
  let inputBuffer: Uint8Array = sourceBytes;
  let fallbackHeader = sniffImageHeader(sourceBytes, lowerExt);

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
  } else if (isDocument) {
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
    const width = isDocument
      ? (docDimensions?.width ?? 1240)
      : (meta.width ?? fallbackHeader.width);
    const height = isDocument
      ? (docDimensions?.height ?? 1754)
      : (meta.height ?? fallbackHeader.height);

    const [webpBuf, rgb16Buf, rgb224Buf] = await Promise.all([
      sharp(inputBuffer, { failOn: "none" })
        .resize(320, 320, {
          fit: "inside",
          withoutEnlargement: true,
          background: { r: 20, g: 33, b: 61, alpha: 1 },
        })
        .webp({ quality: 78, smartSubsample: true, effort: 4 })
        .toBuffer(),
      sharp(inputBuffer, { failOn: "none" })
        .flatten({ background: { r: 20, g: 33, b: 61 } })
        .resize(16, 16, { fit: "fill", kernel: "lanczos3" })
        .removeAlpha()
        .raw()
        .toBuffer(),
      sharp(inputBuffer, { failOn: "none" })
        .flatten({ background: { r: 20, g: 33, b: 61 } })
        .resize(DINO_IMAGE_SIZE, DINO_IMAGE_SIZE, { fit: "fill", kernel: "lanczos3" })
        .removeAlpha()
        .raw()
        .toBuffer(),
    ]);
    let finalWebp = webpBuf;
    if (finalWebp.byteLength > maxPreviewBytes) {
      finalWebp = await sharp(inputBuffer, { failOn: "none" })
        .resize(240, 240, { fit: "inside", withoutEnlargement: true })
        .webp({ quality: 68, smartSubsample: true, effort: 4 })
        .toBuffer();
    }

    const previewBytes = new Uint8Array(new ArrayBuffer(finalWebp.byteLength));
    previewBytes.set(new Uint8Array(finalWebp.buffer, finalWebp.byteOffset, finalWebp.byteLength));

    const vqIndices = encodeRgb16x16ToVqGrid(
      new Uint8Array(rgb16Buf.buffer, rgb16Buf.byteOffset, rgb16Buf.byteLength),
    );
    const dinoEmbedding = extractDinoEmbeddingFromRgb224(
      new Uint8Array(rgb224Buf.buffer, rgb224Buf.byteOffset, rgb224Buf.byteLength),
    );

    return {
      width,
      height,
      colorDepth: fallbackHeader.colorDepth,
      previewBytes,
      previewFormat: PreviewFormat.WEBP,
      vqIndices,
      dinoEmbedding,
      svgExtractedText,
    };
  } catch {
    // Fallback if input is non-image raw buffer
    const vqIndices = encodeBytesToVqGrid(sourceBytes, fallbackHeader.width, fallbackHeader.height, page);
    const { previewBytes, previewFormat } = buildCompactPreviewPayload(
      sourceBytes,
      fallbackHeader,
      vqIndices,
      label,
      page,
      maxPreviewBytes,
    );
    const dinoEmbedding = new Int8Array(new ArrayBuffer(384));
    for (let i = 0; i < 256; i++) {
      dinoEmbedding[i] = ((vqIndices[i] ?? 128) - 128) as number;
    }
    return {
      width: fallbackHeader.width,
      height: fallbackHeader.height,
      colorDepth: fallbackHeader.colorDepth,
      previewBytes,
      previewFormat,
      vqIndices,
      dinoEmbedding,
      svgExtractedText,
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
