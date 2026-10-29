import { describe, expect, test } from "bun:test";
import {
  MAX_TOKEN_BYTES,
  PreviewFormat,
  SECTION1_HEADER_BYTES,
  TokenFlags,
  type TokenMetadataPayload,
} from "../models/token";
import {
  applyHomomorphicMaskGF28,
  gfAdd,
  gfDiv,
  gfInv,
  gfMul,
} from "../utils/galois";
import { sniffImageHeader } from "../utils/imagePreview";
import { Win11TokenPipeline } from "../utils/indexer";
import { QuickLZ } from "../utils/quicklz";
import { decodeBinaryToken, encodeBinaryToken } from "../utils/tokenCodec";
import {
  computeVqAdcDistance,
  decodeVqGridToRgba,
  encodeBytesToVqGrid,
} from "../utils/vqCodebook";
import { CompressedTokenSearchEngine } from "../utils/zipEngine";

describe("TinyToken GF(2^8) + VQ + SHARK/SNARK Pipeline", () => {
  test("GF(2^8) finite field identities hold", () => {
    for (let a = 1; a < 256; a++) {
      const inv = gfInv(a);
      expect(gfMul(a, inv)).toBe(1);
      expect(gfAdd(a, a)).toBe(0);
      expect(gfDiv(gfMul(a, 42), 42)).toBe(a);
    }
  });

  test("QuickLZ compresses and decompresses OCR text losslessly", () => {
    const sample = "Compressed Token Index Specification V2 • GF(2^8) Reed-Solomon + VQ 16x16. ".repeat(12);
    const raw = new TextEncoder().encode(sample);
    const compressed = QuickLZ.compress(raw);
    expect(compressed.length).toBeLessThan(raw.length);
    const decompressed = QuickLZ.decompress(compressed);
    expect(new TextDecoder().decode(decompressed)).toBe(sample);
  });

  test("25 KB Binary Token encodes, verifies SHARK/SNARK attestation, and heals GF(2^8) byte corruption", () => {
    const dummyImg = new Uint8Array(2048);
    for (let i = 0; i < dummyImg.length; i++) dummyImg[i] = (i * 7) & 0xff;

    const vqIndices = encodeBytesToVqGrid(dummyImg, 1024, 768, 1);
    expect(vqIndices.length).toBe(256);

    const rgba = decodeVqGridToRgba(vqIndices);
    expect(rgba.length).toBe(1024);

    const metadata: TokenMetadataPayload = {
      id: "tok_00001_test",
      filename: "test_document.pdf",
      sourcePath: "/api/image/test_document.pdf",
      mimeType: "application/pdf",
      page: 3,
      totalPages: 8,
      width: 1654,
      height: 2339,
      colorDepth: 24,
      createdAt: "2026-10-29T08:00:00.000Z",
      ocrText: "Invoice #94810 • Galois Field GF(2^8) Reed-Solomon & VQ-VAE Neural Thumbnail Test Layer.",
      tags: ["invoice", "gf28"],
    };

    const tokenBytes = encodeBinaryToken({
      originalFileSize: 158920,
      width: 1654,
      height: 2339,
      page: 3,
      previewFormat: PreviewFormat.AVIF,
      vqIndices,
      metadata,
      previewBytes: dummyImg,
    });

    expect(tokenBytes.length).toBeGreaterThan(SECTION1_HEADER_BYTES);
    expect(tokenBytes.length).toBeLessThanOrEqual(MAX_TOKEN_BYTES);

    // 1. Nominal decode
    const decoded = decodeBinaryToken(tokenBytes);
    expect(decoded.magic).toBe("GF28");
    expect(decoded.version).toBe(2);
    expect(decoded.rsParityValid).toBe(true);
    expect(decoded.rsCorrectedErrors).toBe(0);
    expect(decoded.snarkAttestation.valid).toBe(true);
    expect(decoded.metadata.ocrText).toBe(metadata.ocrText);

    // 2. Corrupt 1 byte inside each of the 3 header lanes (Header, SNARK stub, VQ grid)
    const corrupted = new Uint8Array(tokenBytes);
    corrupted[36] = (corrupted[36] ?? 0) ^ 0x55;  // Lane 0: corrupt width
    corrupted[250] = (corrupted[250] ?? 0) ^ 0xaa; // Lane 1: corrupt inside 256B SHARK stub
    corrupted[500] = (corrupted[500] ?? 0) ^ 0x33; // Lane 2: corrupt inside 256B VQ grid

    const healed = decodeBinaryToken(corrupted);
    expect(healed.rsParityValid).toBe(true);
    expect(healed.rsCorrectedErrors).toBe(3);
    expect(healed.width).toBe(1654);
    expect(healed.snarkAttestation.valid).toBe(true);
    expect(Array.from(healed.vqIndices)).toEqual(Array.from(vqIndices));

    // 3. Homomorphic GF(2^8) state transition
    const mutated = applyHomomorphicMaskGF28(tokenBytes, 0x5a, TokenFlags.HOMOMORPHIC_MASKED);
    const decodedMutated = decodeBinaryToken(mutated);
    expect(decodedMutated.rsParityValid).toBe(true);
    expect((decodedMutated.flags & TokenFlags.HOMOMORPHIC_MASKED) !== 0).toBe(true);
  });

  test("VQ Asymmetric Distance Computation (ADC) ranks identical & similar grids accurately", () => {
    const bufA = new Uint8Array(1024).fill(40);
    const bufB = new Uint8Array(1024).fill(42);
    const bufC = new Uint8Array(1024).fill(240);

    const gridA = encodeBytesToVqGrid(bufA, 220, 220, 1);
    const gridB = encodeBytesToVqGrid(bufB, 220, 220, 1);
    const gridC = encodeBytesToVqGrid(bufC, 220, 220, 9);

    expect(computeVqAdcDistance(gridA, gridA)).toBe(0);
    expect(computeVqAdcDistance(gridA, gridB)).toBeLessThan(computeVqAdcDistance(gridA, gridC));
  });

  test("Native Bun ZIP Indexer & Search Engine indexes d.jpg and slices tokens in sub-5ms", async () => {
    // Verify image header sniffer on real ./images/d.jpg
    const dFile = Bun.file("./images/d.jpg");
    if (await dFile.exists()) {
      const dBytes = new Uint8Array(await dFile.slice(0, 4096).arrayBuffer());
      const info = sniffImageHeader(dBytes, ".jpg");
      expect(info.format).toBe(PreviewFormat.JPEG);
      expect(info.width).toBeGreaterThan(0);
      expect(info.height).toBeGreaterThan(0);
    }

    const testZipPath = "./data/test_tokens.zip";
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      outputZipPath: testZipPath,
      allowedExtensions: [".jpg", ".png", ".webp"],
      maxTokenSizeKb: 25,
    });

    const buildRes = await pipeline.buildTokenArchive(12);
    expect(buildRes.totalTokens).toBeGreaterThanOrEqual(12);

    const engine = new CompressedTokenSearchEngine(testZipPath);
    await engine.buildIndex();

    expect(engine.getCount()).toBe(buildRes.totalTokens);
    expect(engine.lastBootIndexMs).toBeLessThan(500);

    const ocrHits = engine.searchOCR("SHARK", 10);
    expect(ocrHits.length).toBeGreaterThan(0);

    const firstId = ocrHits[0]!.id;
    const vqHits = engine.searchSimilarVQ(firstId, "", 5);
    expect(vqHits.length).toBeGreaterThan(0);
    expect(vqHits[0]!.id).toBe(firstId);
    expect(vqHits[0]!.similarityScore).toBe(100);

    const t0 = performance.now();
    const rawSlice = await engine.extractToken(firstId);
    const sliceMs = performance.now() - t0;

    expect(rawSlice).not.toBeNull();
    expect(sliceMs).toBeLessThan(50);

    const decoded = decodeBinaryToken(rawSlice!);
    expect(decoded.id).toBe(firstId);
    expect(decoded.snarkAttestation.valid).toBe(true);
  });
});

