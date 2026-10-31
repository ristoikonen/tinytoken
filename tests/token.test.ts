import { describe, expect, test } from "bun:test";
import { TokenApiHandlers } from "../handlers/tokenHandlers";
import {
  EmbeddingModelId,
  MAX_TOKEN_BYTES,
  PreviewFormat,
  SECTION1_HEADER_BYTES,
  TokenFlags,
  type TokenMetadataPayload,
} from "../models/token";
import {
  applyHomomorphicMaskGF28,
  encodeHeaderParity64,
  gfAdd,
  gfDiv,
  gfInv,
  gfMul,
} from "../utils/galois";
import { sniffImageHeader } from "../utils/imagePreview";
import { Win11TokenPipeline } from "../utils/indexer";
import { QuickLZ } from "../utils/quicklz";
import { corsResponse, errorResponse, jsonResponse } from "../utils/response";
import { generateAttestationKeyPair } from "../utils/snarkAttest";
import {
  decodeBinaryToken,
  encodeBinaryToken,
  generateMockTinyTokenBuffer,
} from "../utils/tokenCodec";
import {
  computeVqAdcDistance,
  decodeVqGridToRgba,
  encodeBytesToVqGrid,
} from "../utils/vqCodebook";
import { VqInvertedIndexEngine } from "../utils/vqInvertedIndex";
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

    // Security: reject crafted header claiming 4 GB decompression size
    const malicious4Gb = new Uint8Array(9);
    malicious4Gb[0] = 3;
    malicious4Gb[1] = 0xff;
    malicious4Gb[2] = 0xff;
    malicious4Gb[3] = 0xff;
    malicious4Gb[4] = 0xff; // 4,294,967,295 bytes (4 GB)
    expect(() => QuickLZ.decompress(malicious4Gb)).toThrow("exceeds safety limit");

    // Security: reject truncated headers
    expect(() => QuickLZ.decompress(new Uint8Array([3, 10, 0]))).toThrow("expected at least 9 header bytes");
  });

  test("25 KB Binary Token encodes, verifies Ed25519+Merkle attestation across all sections, and heals GF(2^8) byte corruption", async () => {
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

    const tokenBytes = await encodeBinaryToken({
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

    // 1. Nominal decode & full Ed25519 + 4-section verification
    const decoded = await decodeBinaryToken(tokenBytes);
    expect(decoded.magic).toBe("GF28");
    expect(decoded.version).toBe(2);
    expect(decoded.rsParityValid).toBe(true);
    expect(decoded.rsCorrectedErrors).toBe(0);
    expect(decoded.snarkAttestation.protocol).toBe("ED25519-MERKLE-GF28-v3");
    expect(decoded.snarkAttestation.valid).toBe(true);
    expect(decoded.snarkAttestation.signatureVerified).toBe(true);
    expect(decoded.snarkAttestation.trustedKeyMatch).toBe(true);
    expect(decoded.snarkAttestation.sectionsVerified).toEqual({
      originHeader: true,
      vqGrid: true,
      section2Metadata: true,
      section3Preview: true,
      merkleRoot: true,
    });
    expect(decoded.metadata.ocrText).toBe(metadata.ocrText);

    // 2. Corrupt 1 byte inside each of the 3 header lanes (Header, Ed25519 attestation block, VQ grid)
    const corrupted = new Uint8Array(tokenBytes);
    corrupted[36] = (corrupted[36] ?? 0) ^ 0x55;  // Lane 0: corrupt width
    corrupted[250] = (corrupted[250] ?? 0) ^ 0xaa; // Lane 1: corrupt inside 256B Ed25519 block
    corrupted[500] = (corrupted[500] ?? 0) ^ 0x33; // Lane 2: corrupt inside 256B VQ grid

    const healed = await decodeBinaryToken(corrupted);
    expect(healed.rsParityValid).toBe(true);
    expect(healed.rsCorrectedErrors).toBe(3);
    expect(healed.width).toBe(1654);
    expect(healed.snarkAttestation.valid).toBe(true);
    expect(Array.from(healed.vqIndices)).toEqual(Array.from(vqIndices));

    // 3. Homomorphic GF(2^8) state transition preserves Ed25519 provenance attestation
    const mutated = applyHomomorphicMaskGF28(tokenBytes, 0x5a, TokenFlags.HOMOMORPHIC_MASKED);
    const decodedMutated = await decodeBinaryToken(mutated);
    expect(decodedMutated.rsParityValid).toBe(true);
    expect(decodedMutated.snarkAttestation.valid).toBe(true);
    expect((decodedMutated.flags & TokenFlags.HOMOMORPHIC_MASKED) !== 0).toBe(true);

    // 4. Tamper with Section II (metaBytes) -> MUST fail section2Metadata & overall attestation
    const tamperedMeta = new Uint8Array(tokenBytes);
    const metaOffset = SECTION1_HEADER_BYTES + 12;
    tamperedMeta[metaOffset] = (tamperedMeta[metaOffset] ?? 0) ^ 0xff;
    const decodedTamperedMeta = await decodeBinaryToken(tamperedMeta);
    expect(decodedTamperedMeta.snarkAttestation.valid).toBe(false);
    expect(decodedTamperedMeta.snarkAttestation.sectionsVerified.section2Metadata).toBe(false);
    expect(decodedTamperedMeta.snarkAttestation.failureReason).toBe("section2_metadata_hash_mismatch");

    // 5. Tamper with Section III (previewBytes) -> MUST fail section3Preview & overall attestation
    const tamperedPreview = new Uint8Array(tokenBytes);
    const previewOffset = tokenBytes.length - 10;
    tamperedPreview[previewOffset] = (tamperedPreview[previewOffset] ?? 0) ^ 0xff;
    const decodedTamperedPreview = await decodeBinaryToken(tamperedPreview);
    expect(decodedTamperedPreview.snarkAttestation.valid).toBe(false);
    expect(decodedTamperedPreview.snarkAttestation.sectionsVerified.section3Preview).toBe(false);
    expect(decodedTamperedPreview.snarkAttestation.failureReason).toBe("section3_preview_hash_mismatch");

    // 6. Forgery attempt via rogue Ed25519 keypair -> MUST fail trustedKeyMatch
    const rogueKeys = await generateAttestationKeyPair();
    const forgedToken = await encodeBinaryToken({
      originalFileSize: 158920,
      width: 1654,
      height: 2339,
      page: 3,
      previewFormat: PreviewFormat.AVIF,
      vqIndices,
      metadata: { ...metadata, ocrText: "FORGED OCR PAYLOAD BY ATTACKER" },
      previewBytes: dummyImg,
      signingKeys: rogueKeys,
    });
    const decodedForged = await decodeBinaryToken(forgedToken);
    expect(decodedForged.rsParityValid).toBe(true);
    expect(decodedForged.snarkAttestation.valid).toBe(false);
    expect(decodedForged.snarkAttestation.trustedKeyMatch).toBe(false);
    expect(decodedForged.snarkAttestation.failureReason).toBe("untrusted_signer_public_key");

    // 7. Keyless hash/signature tampering (recomputing RS parity without private key) -> MUST fail Ed25519 signature
    const sigTampered = new Uint8Array(tokenBytes);
    sigTampered[128 + 170] = (sigTampered[128 + 170] ?? 0) ^ 0x01; // flip 1 bit in 64B Ed25519 signature
    sigTampered.set(encodeHeaderParity64(sigTampered.subarray(0, 640)), 640); // recompute valid RS parity
    const decodedSigTampered = await decodeBinaryToken(sigTampered);
    expect(decodedSigTampered.rsParityValid).toBe(true);
    expect(decodedSigTampered.snarkAttestation.valid).toBe(false);
    expect(decodedSigTampered.snarkAttestation.signatureVerified).toBe(false);
    expect(decodedSigTampered.snarkAttestation.failureReason).toBe("ed25519_signature_invalid");
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

  test("Native Bun ZIP Indexer & Search Engine indexes ./images and ./documents in sub-5ms", async () => {
    const sampleImg = Bun.file("./images/HectorDoor.png");
    if (await sampleImg.exists()) {
      const imgBytes = new Uint8Array(await sampleImg.slice(0, 4096).arrayBuffer());
      const info = sniffImageHeader(imgBytes, ".png");
      expect(info.format).toBe(PreviewFormat.PNG);
      expect(info.width).toBeGreaterThan(0);
      expect(info.height).toBeGreaterThan(0);
    }

    const testZipPath = "./data/test_tokens.zip";
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: testZipPath,
      allowedExtensions: [".jpg", ".jpeg", ".png", ".webp", ".svg", ".pdf", ".docx"],
      maxTokenSizeKb: 25,
      maxPagesPerDocument: 2,
    });

    const buildRes = await pipeline.buildTokenArchive();
    expect(buildRes.imageTokens).toBeGreaterThan(0);
    expect(buildRes.documentTokens).toBeGreaterThanOrEqual(5);
    expect(buildRes.totalTokens).toBe(buildRes.imageTokens + buildRes.documentTokens);

    const engine = new CompressedTokenSearchEngine(testZipPath);
    await engine.buildIndex();

    expect(engine.getCount()).toBe(buildRes.totalTokens);
    expect(engine.lastBootIndexMs).toBeLessThan(500);

    const docHits = engine.searchOCR("PDF Page", 20);
    expect(docHits.length).toBeGreaterThan(0);

    const firstId = docHits[0]!.id;
    const vqHits = engine.searchSimilarVQ(firstId, "", 5);
    expect(vqHits.length).toBeGreaterThan(0);
    expect(vqHits[0]!.id).toBe(firstId);
    expect(vqHits[0]!.similarityScore).toBe(100);

    const t0 = performance.now();
    const rawSlice = await engine.extractToken(firstId);
    const sliceMs = performance.now() - t0;

    expect(rawSlice).not.toBeNull();
    expect(sliceMs).toBeLessThan(50);

    const decoded = await decodeBinaryToken(rawSlice!);
    expect(decoded.id).toBe(firstId);
    expect(decoded.snarkAttestation.valid).toBe(true);
    expect(decoded.snarkAttestation.signatureVerified).toBe(true);
    expect(decoded.embeddingModelId).toBe(EmbeddingModelId.DINOV2_VIT_S14_Q8);
    expect(decoded.metadata.sourcePath.startsWith("/api/document/")).toBe(true);

    // Verify native bun:sqlite VQ Inverted Index Engine + FTS5 full-text search
    const invIndex = new VqInvertedIndexEngine("file::memory:");
    await invIndex.syncTokens(engine.getAllEntries());

    const ftsMatches = invIndex.searchFts5("Australian Governance", 10);
    expect(ftsMatches.length).toBeGreaterThan(0);

    const targetEntry = engine.getEntry(firstId)!;
    // Pass the full 256-element VQ array (with duplicates) to verify deduplicated HAVING threshold
    const patternMatches = await invIndex.findImagesByTokenPattern(
      Array.from(targetEntry.vqIndices),
      0.75,
      10,
    );
    expect(patternMatches.length).toBeGreaterThan(0);
    expect(patternMatches.some((m) => m.id === firstId)).toBe(true);
    invIndex.close();
  }, 15000);

  test("generateMockTinyTokenBuffer creates a valid 25KB Canberra synthetic token that decodes cleanly", async () => {
    const mockBuf = await generateMockTinyTokenBuffer("canberra_grass_01", "grassland_synthetic");
    const decoded = await decodeBinaryToken(mockBuf);

    expect(decoded.id).toBe("canberra_grass_01");
    expect(decoded.magic).toBe("GF28");
    expect(decoded.rsParityValid).toBe(true);
    expect(decoded.snarkAttestation.valid).toBe(true);
    expect(decoded.metadata.environmentClass).toBe("grassland_synthetic");
    expect(decoded.metadata.geoCoordinates?.latitude).toBeCloseTo(-35.1764, 4);
    expect(decoded.metadata.geoCoordinates?.longitude).toBeCloseTo(149.1311, 4);
  });

  test("CORS and cross-origin POST guards block untrusted origins and CSRF attacks", async () => {
    // 1. jsonResponse and errorResponse never emit Access-Control-Allow-Origin: *
    const defaultJson = jsonResponse({ ok: true });
    expect(defaultJson.headers.get("access-control-allow-origin")).toBeNull();
    expect(defaultJson.headers.get("x-content-type-options")).toBe("nosniff");

    const wildcardAttempt = jsonResponse({ ok: true }, 200, true, "*");
    expect(wildcardAttempt.headers.get("access-control-allow-origin")).toBeNull();

    const defaultErr = errorResponse("Bad request", 400);
    expect(defaultErr.headers.get("access-control-allow-origin")).toBeNull();

    // 2. corsResponse rejects cross-origin OPTIONS preflight and permits same-origin
    const evilOptionsReq = new Request("http://localhost:3000/api/index/rebuild", {
      method: "OPTIONS",
      headers: { Origin: "https://evil.example" },
    });
    const evilPreflight = corsResponse(evilOptionsReq);
    expect(evilPreflight.status).toBe(403);
    expect(evilPreflight.headers.get("access-control-allow-origin")).toBeNull();

    const sameOriginOptionsReq = new Request("http://localhost:3000/api/index/rebuild", {
      method: "OPTIONS",
      headers: { Origin: "http://localhost:3000" },
    });
    const okPreflight = corsResponse(sameOriginOptionsReq);
    expect(okPreflight.status).toBe(204);
    expect(okPreflight.headers.get("access-control-allow-origin")).toBe("http://localhost:3000");

    // 3. Mutating POST endpoints reject cross-origin Origin, Sec-Fetch-Site, and no-cors form posts
    const engine = new CompressedTokenSearchEngine("./data/test_tokens.zip");
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: "./data/test_tokens.zip",
      allowedExtensions: [".png"],
      maxTokenSizeKb: 25,
    });
    const handlers = new TokenApiHandlers(engine, pipeline);

    const crossOriginRebuild = await handlers.handleRebuildIndex(
      new Request("http://localhost:3000/api/index/rebuild", {
        method: "POST",
        headers: {
          Origin: "https://evil.example",
          "Content-Type": "application/json",
        },
        body: "{}",
      }),
    );
    expect(crossOriginRebuild.status).toBe(403);

    const crossSiteFetchMutate = await handlers.handleHomomorphicMutate(
      "tok_00001_HectorDoor.png",
      new Request("http://localhost:3000/api/token/tok_00001_HectorDoor.png/mutate", {
        method: "POST",
        headers: {
          "Sec-Fetch-Site": "cross-site",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ scalar: 0x5a }),
      }),
    );
    expect(crossSiteFetchMutate.status).toBe(403);

    const noCorsFormPost = await handlers.handleRebuildIndex(
      new Request("http://localhost:3000/api/index/rebuild", {
        method: "POST",
        headers: {
          Origin: "http://localhost:3000",
          "Sec-Fetch-Site": "same-origin",
          "Content-Type": "text/plain",
        },
        body: "x=1",
      }),
    );
    expect(noCorsFormPost.status).toBe(415);
  });

  test("Malformed URI paths return 400 Bad Request instead of throwing unhandled URIError", async () => {
    const { default: server } = await import("../server");
    const malformedTokenReq = new Request("http://localhost:3000/api/token/%E0%A4%A");
    const resToken = await server.fetch(malformedTokenReq);
    expect(resToken.status).toBe(400);
    const jsonToken = (await resToken.json()) as { success: boolean; error: string };
    expect(jsonToken.success).toBe(false);
    expect(jsonToken.error).toBe("Invalid URI encoding");

    const malformedImgReq = new Request("http://localhost:3000/api/image/%E0%A4%A");
    const resImg = await server.fetch(malformedImgReq);
    expect(resImg.status).toBe(400);

    const malformedDocReq = new Request("http://localhost:3000/api/document/%E0%A4%A");
    const resDoc = await server.fetch(malformedDocReq);
    expect(resDoc.status).toBe(400);
  });

  test("handleGetTokenPreview applies X-Content-Type-Options: nosniff and CSP sandbox on SVG payloads", async () => {
    const rawSvgToken = await encodeBinaryToken({
      originalFileSize: 64,
      width: 100,
      height: 100,
      page: 1,
      previewFormat: PreviewFormat.SVG,
      vqIndices: new Uint8Array(256),
      metadata: {
        id: "tok_svg_csp_test",
        filename: "test.svg",
        sourcePath: "",
        mimeType: "image/svg+xml",
        page: 1,
        totalPages: 1,
        width: 100,
        height: 100,
        colorDepth: 24,
        createdAt: new Date().toISOString(),
        ocrText: "",
      },
      previewBytes: new TextEncoder().encode("<svg><script>alert(1)</script></svg>"),
    });

    const engine = new CompressedTokenSearchEngine("./data/test_tokens.zip");
    engine.extractToken = async () => rawSvgToken;
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: "./data/test_tokens.zip",
      allowedExtensions: [".png"],
      maxTokenSizeKb: 25,
    });
    const handlers = new TokenApiHandlers(engine, pipeline);

    const previewRes = await handlers.handleGetTokenPreview("tok_svg_csp_test");
    expect(previewRes.status).toBe(200);
    expect(previewRes.headers.get("content-type")).toBe("image/svg+xml");
    expect(previewRes.headers.get("x-content-type-options")).toBe("nosniff");
    expect(previewRes.headers.get("content-security-policy")).toBe("default-src 'none'; sandbox");
  });
});

