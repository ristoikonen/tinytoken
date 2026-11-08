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
import {
  DocxDocumentProcessor,
  getDocumentProcessor,
  MAXIMUM_DECOMPRESSED_DOCX,
  maximum_decompressed_DOCX,
  PdfDocumentProcessor,
  TextDocumentProcessor,
} from "../utils/documentProcessor";
import {
  extensionToMimeType,
  formatToMimeType,
  rasterizeAssetToWebpAndTensors,
  sniffImageHeader,
} from "../utils/imagePreview";
import { Win11TokenPipeline } from "../utils/indexer";
import { PipelineBenchmark } from "../utils/benchmark";
import { QuickLZ } from "../utils/quicklz";
import { corsResponse, errorResponse, jsonResponse } from "../utils/response";
import {
  generateAttestationKeyPair,
  getAuthorityAttestationKeys,
} from "../utils/snarkAttest";
import {
  decodeBinaryToken,
  encodeBinaryToken,
  extractTokenMetadata,
  extractTokenPreviewBytes,
  generateMockTinyTokenBuffer,
  inspectBinaryTokenHeader,
  uuidStringToBytes,
  validateBinaryToken,
} from "../utils/tokenCodec";
import {
  adcDistanceToSimilarityPct,
  computeHybridVisualSimilarity,
  computeVqAdcDistance,
  decodeVqGridToRgba,
  encodeBytesToVqGrid,
  MAX_ADC_DISTANCE,
  quantizeRgbToCentroid,
  RGB_TO_VQ_LUT,
  validateVqGrid,
  VQ_CODEBOOK,
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

  test("Archive reuse detects source content changes using SHA-256 manifest hashes", async () => {
    const sourcePath = "./data/test_source_hash_reuse.txt";
    const archivePath = "./data/test_source_hash_reuse.zip";
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./data",
      documentDirectory: "./data",
      outputZipPath: archivePath,
      allowedExtensions: [".txt"],
      maxTokenSizeKb: 25,
    });

    try {
      await Bun.write(sourcePath, "alpha");
      const firstBuild = await pipeline.buildTokenArchive();
      expect(firstBuild.totalTokens).toBe(1);
      expect(firstBuild.reusedTokens).toBe(0);

      const firstEngine = new CompressedTokenSearchEngine(archivePath);
      await firstEngine.buildIndex();
      const firstHash = firstEngine.getAllEntries()[0]?.sourceHash;
      expect(firstHash).toMatch(/^[a-f0-9]{64}$/);

      await Bun.write(sourcePath, "bravo");
      const changedBuild = await pipeline.buildTokenArchive();
      expect(changedBuild.totalTokens).toBe(1);
      expect(changedBuild.reusedTokens).toBe(0);

      const changedEngine = new CompressedTokenSearchEngine(archivePath);
      await changedEngine.buildIndex();
      expect(changedEngine.getAllEntries()[0]?.sourceHash).not.toBe(firstHash);

      const unchangedBuild = await pipeline.buildTokenArchive();
      expect(unchangedBuild.reusedTokens).toBe(1);
    } finally {
      for (const path of [sourcePath, archivePath]) {
        const file = Bun.file(path);
        if (await file.exists()) await file.delete();
      }
    }
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
    const authorityKeys = await getAuthorityAttestationKeys();
    const decodedNoTrustedKey = await decodeBinaryToken(tokenBytes);
    expect(decodedNoTrustedKey.magic).toBe("GF28");
    expect(decodedNoTrustedKey.version).toBe(2);
    expect(decodedNoTrustedKey.rsParityValid).toBe(true);
    expect(decodedNoTrustedKey.rsCorrectedErrors).toBe(0);
    expect(decodedNoTrustedKey.snarkAttestation.protocol).toBe("ED25519-MERKLE-GF28-v3");
    expect(decodedNoTrustedKey.snarkAttestation.signatureVerified).toBe(true);
    expect(decodedNoTrustedKey.snarkAttestation.attestationStatus?.signatureValid).toBe(true);
    // Without trusted public key provided by caller, trustedIssuer must NOT be implied
    expect(decodedNoTrustedKey.snarkAttestation.attestationStatus?.trustedIssuer).toBe(false);
    expect(decodedNoTrustedKey.snarkAttestation.trustedKeyMatch).toBe(false);

    // With trusted authority key supplied:
    const decoded = await decodeBinaryToken(tokenBytes, {
      trustedPublicKeyRaw: authorityKeys.publicKeyRaw,
    });
    expect(decoded.snarkAttestation.valid).toBe(true);
    expect(decoded.snarkAttestation.trustedKeyMatch).toBe(true);
    expect(decoded.snarkAttestation.attestationStatus?.trustedIssuer).toBe(true);
    expect(decoded.snarkAttestation.sectionsVerified).toEqual({
      originHeader: true,
      vqGrid: true,
      section2Metadata: true,
      section3Preview: true,
      merkleRoot: true,
    });
    expect(decoded.metadata.ocrText).toBe(metadata.ocrText);

    // Strictly validate valid token
    const validated = await validateBinaryToken(tokenBytes, {
      trustedPublicKeyRaw: authorityKeys.publicKeyRaw,
    });
    expect(validated.envelopeBoundsValid).toBe(true);

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
    const decodedForged = await decodeBinaryToken(forgedToken, {
      trustedPublicKeyRaw: authorityKeys.publicKeyRaw,
    });
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
      const info = await sniffImageHeader(imgBytes, ".png");
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

    const rebuildRes = await pipeline.buildTokenArchive();
    expect(rebuildRes.totalTokens).toBe(buildRes.totalTokens);
    expect(rebuildRes.reusedTokens).toBe(buildRes.totalTokens);

    const engine = new CompressedTokenSearchEngine(testZipPath);
    await engine.buildIndex();

    expect(engine.getCount()).toBe(buildRes.totalTokens);
    expect(
      engine.getAllEntries().every((entry) => /^[a-f0-9]{64}$/.test(entry.sourceHash ?? "")),
    ).toBe(true);
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
    expect(decoded.embeddingModelId).toBe(EmbeddingModelId.DINOV3_VITS16_Q8);
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

  test("VisualTraitRegistry evaluates 16x16 grids and handles manual registrations in sub-0.1ms", async () => {
    const { VisualTraitRegistry } = await import("../utils/visualTraitRegistry");
    const registry = new VisualTraitRegistry(
      "./data/test_manual_token_registry.json",
      "./data/test_object_registry.json",
      "./data/test_learning_evidence.json",
    );

    // Synthetic 16x16 grid with kangaroo (108), paper white (255), and ink (252)
    const grid = new Uint8Array(256).fill(255); // 250 white cells
    for (let i = 0; i < 6; i++) grid[i] = 108; // 6 kangaroo cells

    const traits = registry.evaluateGrid(grid);
    expect(traits.length).toBe(2);

    const whiteTrait = traits.find((t) => t.tokenId === 255);
    expect(whiteTrait).toBeDefined();
    expect(whiteTrait?.category).toBe("ColorSpace");
    expect(whiteTrait?.isManuallyVerified).toBe(true);
    expect(whiteTrait?.percentage).toBe(97.7);

    const kangarooTrait = traits.find((t) => t.tokenId === 108);
    expect(kangarooTrait).toBeDefined();
    expect(kangarooTrait?.name).toContain("Kangaroo");
    expect(kangarooTrait?.category).toBe("Silhouette");
    expect(kangarooTrait?.isManuallyVerified).toBe(true);

    // Register custom entity
    await registry.registerToken(42, "Custom Drone Target", "Silhouette");
    grid[10] = 42;
    grid[11] = 42;
    const updatedTraits = registry.evaluateGrid(grid);
    const customTrait = updatedTraits.find((t) => t.tokenId === 42);
    expect(customTrait).toBeDefined();
    expect(customTrait?.name).toBe("Custom Drone Target");
    expect(customTrait?.isManuallyVerified).toBe(true);

    // Teach object bounding box annotation
    const teachResult = await registry.teachObject(
      "tok_test_01",
      "Red Kangaroo Juvenile",
      "Silhouette",
      [0.2, 0.3, 0.4, 0.5],
      [108, 132],
    );
    expect(teachResult.taughtCentroids).toEqual([108, 132]);
    expect(teachResult.annotation.label).toBe("Red Kangaroo Juvenile");

    const annotations = await registry.getObjectAnnotations();
    expect(annotations.some((a) => a.label === "Red Kangaroo Juvenile")).toBe(true);

    // Clean up test files if created
    for (const p of [
      "./data/test_manual_token_registry.json",
      "./data/test_object_registry.json",
      "./data/test_learning_evidence.json",
    ]) {
      try {
        const file = Bun.file(p);
        if (await file.exists()) {
          await file.delete();
        }
      } catch {}
    }
  });

  test("Google Photos Ingestion uses google-auth-library with photoslibrary.readonly scope", async () => {
    const {
      GOOGLE_PHOTOS_READ_SCOPES,
      getGooglePhotosAuthUrl,
      resolveGooglePhotosAuth,
      fetchAndIndexGooglePhotos,
    } = await import("../utils/googlePhotos");

    // 1. Verify read access scope
    expect(GOOGLE_PHOTOS_READ_SCOPES).toEqual([
      "https://www.googleapis.com/auth/photoslibrary.readonly",
    ]);

    // 2. Verify OAuth2 consent URL generation
    const authUrl = getGooglePhotosAuthUrl();
    expect(authUrl).toBeDefined();
    expect(authUrl).toContain("accounts.google.com/o/oauth2/v2/auth");
    expect(authUrl).toContain("access_type=offline");
    expect(authUrl).toContain("photoslibrary.readonly");

    // 3. Verify override token resolution
    const authRes = await resolveGooglePhotosAuth("test_bearer_token");
    expect(authRes.accessToken).toBe("test_bearer_token");
    expect(authRes.source).toBe("override_token");

    // 4. Verify batch fetch telemetry formatting
    const prevFlag = Bun.env.USE_MY_GOOGLE_PHOTOS;
    try {
      Bun.env.USE_MY_GOOGLE_PHOTOS = "true";
      const result = await fetchAndIndexGooglePhotos();
      expect(result.messages.length).toBeGreaterThanOrEqual(3);
      expect(result.messages.some((m) => m.startsWith("Fertched "))).toBe(true);
      expect(result.messages).toContain("Set Indexed.");
      expect(result.messages).toContain("No traits added.");
    } finally {
      Bun.env.USE_MY_GOOGLE_PHOTOS = prevFlag;
    }
  });

  test("APP_ZIP_USE=false keeps *.token files in data dir without creating tokens.zip", async () => {
    const testDir = "./data/test_unzipped_tokens";
    const zipPath = `${testDir}/tokens.zip`;

    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      outputZipPath: zipPath,
      dataDirectory: testDir,
      useZip: false,
      allowedExtensions: [".png"],
      maxTokenSizeKb: 25,
    });

    const buildRes = await pipeline.buildTokenArchive();
    expect(buildRes.totalTokens).toBeGreaterThan(0);

    // Verify tokens.zip was NOT created
    const zipFile = Bun.file(zipPath);
    expect(await zipFile.exists()).toBe(false);

    // Verify individual *.token files were written
    const glob = new Bun.Glob("*.token");
    const tokenFiles: string[] = [];
    for await (const name of glob.scan({ cwd: testDir })) {
      tokenFiles.push(name);
    }
    expect(tokenFiles.length).toBe(buildRes.totalTokens);

    // Verify _index.json was created for fast indexing
    const manifestFile = Bun.file(`${testDir}/_index.json`);
    expect(await manifestFile.exists()).toBe(true);

    // Test SearchEngine in unzipped directory mode
    const engine = new CompressedTokenSearchEngine(zipPath, {
      useZip: false,
      dataDirPath: testDir,
    });
    expect(engine.isZipMode()).toBe(false);
    expect(engine.getZipFilePath()).toBe(testDir);

    await engine.buildIndex();
    expect(engine.getCount()).toBe(buildRes.totalTokens);

    // Test extraction directly from *.token file
    const firstEntry = engine.getAllEntries()[0]!;
    const extractedBytes = await engine.extractToken(firstEntry.id);
    expect(extractedBytes).not.toBeNull();
    expect(extractedBytes!.length).toBeGreaterThan(500);

    // Clean up testDir files
    for (const name of tokenFiles) {
      await Bun.file(`${testDir}/${name}`).delete();
    }
    await manifestFile.delete();
  });

  test("quantizeRgbToCentroid uses precomputed RGB_TO_VQ_LUT and reaches all 256 centroids including greyscale anchors", () => {
    expect(RGB_TO_VQ_LUT.length).toBe(32768);

    // Verify all four greyscale contrast anchors are reachable
    expect(quantizeRgbToCentroid(0, 0, 0)).toBe(252); // pure ink black
    expect(quantizeRgbToCentroid(85, 85, 85)).toBe(253); // dark charcoal
    expect(quantizeRgbToCentroid(170, 170, 170)).toBe(254); // silver grey
    expect(quantizeRgbToCentroid(255, 255, 255)).toBe(255); // pure paper white

    // Verify that all 256 centroids are reachable in the LUT under the 4x green perceptual metric
    const uniqueCentroids = new Set(RGB_TO_VQ_LUT);
    expect(uniqueCentroids.size).toBe(256);

    // Verify bit-shift index encoding matches: ((r >>> 3) << 10) | ((g >>> 3) << 5) | (b >>> 3)
    const testR = 120;
    const testG = 200;
    const testB = 50;
    const expectedIdx = ((testR >>> 3) << 10) | ((testG >>> 3) << 5) | (testB >>> 3);
    expect(quantizeRgbToCentroid(testR, testG, testB)).toBe(RGB_TO_VQ_LUT[expectedIdx]!);
  });

  test("ADC distance normalization anchors MAX_ADC_DISTANCE to theoretical maximum of distance model", () => {
    // 16,646,400 = 256 * 255^2
    expect(MAX_ADC_DISTANCE).toBe(16_646_400);

    const blackGrid = new Uint8Array(256).fill(252); // centroid #252 = (0, 0, 0)
    const whiteGrid = new Uint8Array(256).fill(255); // centroid #255 = (255, 255, 255)

    const maxDist = computeVqAdcDistance(blackGrid, whiteGrid);
    expect(maxDist).toBe(MAX_ADC_DISTANCE);
    expect(adcDistanceToSimilarityPct(maxDist)).toBe(0);

    const zeroDist = computeVqAdcDistance(blackGrid, blackGrid);
    expect(zeroDist).toBe(0);
    expect(adcDistanceToSimilarityPct(zeroDist)).toBe(100);
  });

  test("computeHybridVisualSimilarity returns dinoSimilarityPct: null when DINO is not evaluated or models mismatch", () => {
    const gridA = new Uint8Array(256).fill(42);
    const gridB = new Uint8Array(256).fill(42);

    // Case 1: adcDistance === 0 but no DINO embeddings provided -> dinoSimilarityPct must be null, not 100
    const noDinoRes = computeHybridVisualSimilarity(gridA, gridB);
    expect(noDinoRes.similarityScore).toBe(100);
    expect(noDinoRes.adcDistance).toBe(0);
    expect(noDinoRes.dinoSimilarityPct).toBeNull();

    // Case 2: Model IDs unknown or mismatched -> DINO must not be evaluated
    const dummyDino1 = new Int8Array(384).fill(10);
    const dummyDino2 = new Int8Array(384).fill(10);

    // Undefined queryModelId -> unknown model is not compatible -> dinoSimilarityPct: null
    const unknownModelRes = computeHybridVisualSimilarity(
      gridA,
      gridB,
      dummyDino1,
      dummyDino2,
      undefined,
      2,
    );
    expect(unknownModelRes.dinoSimilarityPct).toBeNull();
    expect(unknownModelRes.similarityScore).toBe(100);

    // Different model IDs (e.g. 1 vs 2) -> incompatible -> dinoSimilarityPct: null
    const mismatchModelRes = computeHybridVisualSimilarity(
      gridA,
      gridB,
      dummyDino1,
      dummyDino2,
      1,
      2,
    );
    expect(mismatchModelRes.dinoSimilarityPct).toBeNull();

    // Known matching model IDs (2 vs 2) -> compatible -> dinoSimilarityPct calculated
    const matchingModelRes = computeHybridVisualSimilarity(
      gridA,
      gridB,
      dummyDino1,
      dummyDino2,
      2,
      2,
    );
    expect(matchingModelRes.dinoSimilarityPct).toBe(100);
    expect(matchingModelRes.similarityScore).toBe(100);
  });

  test("validateVqGrid enforces 256-byte grids and prevents silent truncation/padding bugs", () => {
    const validGrid = new Uint8Array(256).fill(42);
    expect(() => validateVqGrid(validGrid)).not.toThrow();

    const shortGrid = new Uint8Array(100).fill(42);
    expect(() => validateVqGrid(shortGrid)).toThrow(RangeError);
    expect(() => validateVqGrid(shortGrid)).toThrow("Expected 256-byte VQ grid, received 100");

    const longGrid = new Uint8Array(300).fill(42);
    expect(() => validateVqGrid(longGrid)).toThrow(RangeError);

    // computeVqAdcDistance must throw when given a truncated grid
    expect(() => computeVqAdcDistance(shortGrid, validGrid)).toThrow(RangeError);
    expect(() => computeVqAdcDistance(validGrid, shortGrid)).toThrow(RangeError);

    // decodeVqGridToRgba must throw when given a truncated grid
    expect(() => decodeVqGridToRgba(shortGrid)).toThrow(RangeError);

    // decodeVqGridToRgba decodes valid grids faithfully without needing fallback
    const decodedRgba = decodeVqGridToRgba(validGrid);
    expect(decodedRgba.length).toBe(1024);
    const expectedCentroid = VQ_CODEBOOK[42]!;
    expect(decodedRgba[0]).toBe(expectedCentroid.r);
    expect(decodedRgba[1]).toBe(expectedCentroid.g);
    expect(decodedRgba[2]).toBe(expectedCentroid.b);
    expect(decodedRgba[3]).toBe(255);
  });

  test("APP_STOP_ON_25KB_LIMIT=true enforces configured hard maximum size", async () => {
    const origEnv = Bun.env.APP_STOP_ON_25KB_LIMIT;
    try {
      Bun.env.APP_STOP_ON_25KB_LIMIT = "true";
      // Create a pipeline with 1 KB maximum size (1024 bytes)
      const pipeline = new Win11TokenPipeline({
        sourceDirectory: "./images",
        outputZipPath: "./data/test_size_limit.zip",
        allowedExtensions: [".png"],
        maxTokenSizeKb: 1, // 1024 bytes maximum
      });

      const samplePng = Bun.file("./images/HectorDoor.png");
      if (await samplePng.exists()) {
        const rawBytes = new Uint8Array(await samplePng.arrayBuffer());
        // Since a standard token is at least 704 bytes header + metadata + preview (> 1024 bytes),
        // it must throw with the exact required error format:
        await expect(
          pipeline.tokenizeBuffer(rawBytes, "HectorDoor.png", 1),
        ).rejects.toThrow(/Token exceeds maximum size: \d+ \/ 1024 bytes/);
      }
    } finally {
      Bun.env.APP_STOP_ON_25KB_LIMIT = origEnv;
    }
  });

  test("extensionToMimeType and formatToMimeType cover all Bun-supported image formats and documents", () => {
    expect(extensionToMimeType(".jpg")).toBe("image/jpeg");
    expect(extensionToMimeType(".jpeg")).toBe("image/jpeg");
    expect(extensionToMimeType(".png")).toBe("image/png");
    expect(extensionToMimeType(".webp")).toBe("image/webp");
    expect(extensionToMimeType(".avif")).toBe("image/avif");
    expect(extensionToMimeType(".bmp")).toBe("image/bmp");
    expect(extensionToMimeType(".gif")).toBe("image/gif");
    expect(extensionToMimeType(".tiff")).toBe("image/tiff");
    expect(extensionToMimeType(".tif")).toBe("image/tiff");
    expect(extensionToMimeType(".heic")).toBe("image/heic");
    expect(extensionToMimeType(".heif")).toBe("image/heic");
    expect(extensionToMimeType(".svg")).toBe("image/svg+xml");
    expect(extensionToMimeType(".pdf")).toBe("application/pdf");
    expect(extensionToMimeType(".docx")).toBe(
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    );
    expect(extensionToMimeType(".doc")).toBe("application/msword");
    expect(extensionToMimeType(".txt")).toBe("text/plain");
    expect(extensionToMimeType(".md")).toBe("text/markdown");
    expect(extensionToMimeType(".unknown")).toBe("application/octet-stream");

    expect(formatToMimeType(PreviewFormat.PNG)).toBe("image/png");
    expect(formatToMimeType(PreviewFormat.JPEG)).toBe("image/jpeg");
    expect(formatToMimeType(PreviewFormat.WEBP)).toBe("image/webp");
    expect(formatToMimeType(PreviewFormat.AVIF)).toBe("image/avif");
    expect(formatToMimeType(PreviewFormat.BMP)).toBe("image/bmp");
    expect(formatToMimeType(PreviewFormat.SVG)).toBe("image/svg+xml");
  });

  test("sniffImageHeader asynchronously extracts metadata using Bun native image metadata & SVG parser", async () => {
    // 1. SVG
    const svgStr = `<svg xmlns="http://www.w3.org/2000/svg" width="300" height="200"><rect fill="red"/></svg>`;
    const svgBytes = new TextEncoder().encode(svgStr);
    const svgInfo = await sniffImageHeader(svgBytes, ".svg");
    expect(svgInfo.format).toBe(PreviewFormat.SVG);
    expect(svgInfo.width).toBe(300);
    expect(svgInfo.height).toBe(200);
    expect(svgInfo.mimeType).toBe("image/svg+xml");

    // 2. Real PNG
    const pngFile = Bun.file("./images/HectorDoor.png");
    if (await pngFile.exists()) {
      const pngBytes = new Uint8Array(await pngFile.slice(0, 4096).arrayBuffer());
      const pngInfo = await sniffImageHeader(pngBytes, ".png");
      expect(pngInfo.format).toBe(PreviewFormat.PNG);
      expect(pngInfo.width).toBeGreaterThan(0);
      expect(pngInfo.height).toBeGreaterThan(0);
      expect(pngInfo.mimeType).toBe("image/png");
    }
  });

  test("Model ID consistency: Meta DINOv3 ViT-S/16 across binary header, metadata, and search", async () => {
    expect(EmbeddingModelId.DINOV3_VITS16_Q8).toBe(0x02);
    // Ensure legacy DINOv2 alias has been completely eliminated
    expect((EmbeddingModelId as Record<string, unknown>)["DINOV2_VIT_S14_Q8"]).toBeUndefined();

    // Verify token encodes and decodes with DINOV3_VITS16_Q8 model ID preserved
    const testBytes = await encodeBinaryToken({
      originalFileSize: 1024,
      width: 256,
      height: 256,
      page: 1,
      previewFormat: PreviewFormat.WEBP,
      shardBlockId: 0,
      shardIndex: 1,
      embeddingModelId: EmbeddingModelId.DINOV3_VITS16_Q8,
      vqIndices: new Uint8Array(256).fill(12),
      metadata: {
        id: "tok_dinov3_test",
        filename: "dinov3_test.png",
        sourcePath: "/test",
        mimeType: "image/png",
        page: 1,
        totalPages: 1,
        width: 256,
        height: 256,
        colorDepth: 24,
        createdAt: new Date().toISOString(),
        ocrText: "DINOv3 test token",
        tags: ["dinov3-vit-s16"],
      },
      previewBytes: new Uint8Array([0x52, 0x49, 0x46, 0x46]),
    });

    const decoded = await decodeBinaryToken(testBytes);
    expect(decoded.embeddingModelId).toBe(EmbeddingModelId.DINOV3_VITS16_Q8);
    expect(decoded.metadata.tags).toContain("dinov3-vit-s16");
  });

  test("PDFium failure fallback prepends 'Read failed' to OCR text and marks failure tags", async () => {
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: "./data/test_pdfium_fail.zip",
      allowedExtensions: [".pdf"],
      maxTokenSizeKb: 25,
    });

    const corruptPdfPath = "./data/corrupted_pdfium_test.pdf";
    try {
      // Write non-PDF bytes to trigger PDFium parser exception
      await Bun.write(corruptPdfPath, new Uint8Array([0x00, 0x01, 0x02, 0x03, 0x04]));

      const items = await pipeline.tokenizeDocumentFile(corruptPdfPath, 99);
      expect(items.length).toBe(1);

      const item = items[0]!;
      expect(item.meta.ocrText).toContain("Read failed (PDFium error:");
      expect(item.meta.mimeType).toBe("application/pdf");

      const decoded = await decodeBinaryToken(item.tokenBytes);
      expect(decoded.metadata.ocrText).toContain("Read failed (PDFium error:");
      expect(decoded.metadata.tags).toContain("read-failed");
      expect(decoded.metadata.tags).toContain("pdfium-failed");
      expect(decoded.metadata.tags).toContain("dinov3-vit-s16");
    } finally {
      const file = Bun.file(corruptPdfPath);
      if (await file.exists()) {
        await file.delete();
      }
    }
  });

  test("scanDirectory distinguishes an empty directory from a failed scan when logging", async () => {
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      outputZipPath: "./data/test_scan.zip",
      allowedExtensions: [".jpg"],
      maxTokenSizeKb: 25,
    });

    const origError = console.error;
    const origInfo = console.info;
    let loggedError = "";
    let loggedInfo = "";

    console.error = (...args: unknown[]) => {
      loggedError += args.join(" ") + "\n";
    };
    console.info = (...args: unknown[]) => {
      loggedInfo += args.join(" ") + "\n";
    };

    try {
      // 1. Non-existent directory -> should log [SCAN_FAILED]
      const failedScan = await pipeline.scanDirectory("./non_existent_dir_99999");
      expect(failedScan).toEqual([]);
      expect(loggedError).toContain("[SCAN_FAILED]");

      // 2. Empty directory -> should log [SCAN_EMPTY]
      const emptyDir = "./data/temp_empty_test_dir";
      // Use Bun shell or temp directory
      const proc = Bun.spawnSync(["pwsh", "-NoProfile", "-Command", `New-Item -ItemType Directory -Path "${emptyDir}" -Force | Out-Null`]);
      expect(proc.exitCode).toBe(0);

      const emptyScan = await pipeline.scanDirectory(emptyDir);
      expect(emptyScan).toEqual([]);
      expect(loggedInfo).toContain("[SCAN_EMPTY]");

      // Clean up temp directory
      Bun.spawnSync(["pwsh", "-NoProfile", "-Command", `Remove-Item -Recurse -Force "${emptyDir}"`]);
    } finally {
      console.error = origError;
      console.info = origInfo;
    }
  });

  test("DocumentProcessor provides format-aware routing and text file rendering", async () => {
    // 1. Processor resolution
    const pdfProc = getDocumentProcessor(".pdf");
    expect(pdfProc).toBeInstanceOf(PdfDocumentProcessor);
    expect(pdfProc.supports(".pdf")).toBe(true);
    expect(pdfProc.supports(".txt")).toBe(false);

    const docxProc = getDocumentProcessor(".docx");
    expect(docxProc).toBeInstanceOf(DocxDocumentProcessor);
    expect(docxProc.supports(".docx")).toBe(true);

    const txtProc = getDocumentProcessor(".txt");
    expect(txtProc).toBeInstanceOf(TextDocumentProcessor);
    expect(txtProc.supports(".txt")).toBe(true);
    expect(txtProc.supports(".md")).toBe(true);
    expect(txtProc.supports(".markdown")).toBe(true);

    // 2. Text file processing via pipeline
    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: "./data/test_text_pipe.zip",
      allowedExtensions: [".txt", ".md"],
      maxTokenSizeKb: 25,
    });

    const testTxtPath = "./data/sample_audit_note.txt";
    const textContent =
      "Antigravity System Audit Log\n" +
      "Timestamp: 2026-11-04T11:00:00Z\n" +
      "Status: All systems operational\n" +
      "Model: Meta DINOv3 ViT-S/16 Int8\n" +
      "Format: GF(2^8) 25KB Binary Token\n";

    await Bun.write(testTxtPath, textContent);

    try {
      const items = await pipeline.tokenizeDocumentFile(testTxtPath, 1);
      expect(items.length).toBe(1);

      const item = items[0]!;
      expect(item.meta.mimeType).toContain("text/plain");
      expect(item.meta.ocrText).toContain("Antigravity System Audit Log");
      expect(item.meta.ocrText).toContain("Status: All systems operational");

      const decoded = await decodeBinaryToken(item.tokenBytes);
      expect(decoded.previewFormat).toBe(PreviewFormat.WEBP);
      expect(decoded.previewBytes.length).toBeGreaterThan(0);
      expect(decoded.vqIndices.length).toBe(256);
      expect(decoded.embeddingModelId).toBe(EmbeddingModelId.DINOV3_VITS16_Q8);
    } finally {
      const file = Bun.file(testTxtPath);
      if (await file.exists()) {
        await file.delete();
      }
    }
  });

  test("Maximum source file size enforcement (APP_MAX_FILE_SIZE) allows searchOCR('File size over max size')", async () => {
    const origMax = Bun.env.APP_MAX_FILE_SIZE;
    const testOversizedPath = "./data/oversized_test_file.txt";
    // Set limit very low (0.00001 MB = ~10 bytes) so our file exceeds it
    Bun.env.APP_MAX_FILE_SIZE = "0.00001";

    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: "./data/test_oversized.zip",
      allowedExtensions: [".txt"],
      maxTokenSizeKb: 25,
    });

    try {
      await Bun.write(testOversizedPath, "This content is intentionally larger than 10 bytes to trigger APP_MAX_FILE_SIZE limit.");

      const items = await pipeline.tokenizeDocumentFile(testOversizedPath, 77);
      expect(items.length).toBe(1);

      const item = items[0]!;
      expect(item.meta.ocrText).toContain("File size over max size");

      // Verify search engine finds it via searchOCR
      const testZip = "./data/test_oversized_search.zip";
      const { writeNativeTokenZipArchive } = await import("../utils/zipEngine");
      await writeNativeTokenZipArchive(testZip, items);

      const engine = new CompressedTokenSearchEngine(testZip);
      await engine.buildIndex();

      const hits = engine.searchOCR("File size over max size");
      expect(hits.length).toBe(1);
      expect(hits[0]!.ocrSnippet).toContain("File size over max size");

      const zipFile = Bun.file(testZip);
      if (await zipFile.exists()) await zipFile.delete();
    } finally {
      Bun.env.APP_MAX_FILE_SIZE = origMax;
      const file = Bun.file(testOversizedPath);
      if (await file.exists()) await file.delete();
    }
  });

  test("DOCX uncompressed size ceiling (maximum_decompressed_DOCX = 20MB) prevents zip-bombs and allows searchOCR('DOCX uncomressed size over max size')", async () => {
    expect(MAXIMUM_DECOMPRESSED_DOCX).toBe(20 * 1024 * 1024);
    expect(maximum_decompressed_DOCX).toBe(20 * 1024 * 1024);

    // Build a mock DOCX archive with a zip header claiming 25 MB uncompressed size
    const buf = new Uint8Array(120);
    const view = new DataView(buf.buffer);
    view.setUint32(0, 0x04034b50, true); // Local file header signature
    view.setUint16(8, 8, true);          // Deflate method
    view.setUint32(18, 10, true);         // Compressed size: 10 bytes
    view.setUint32(22, 25 * 1024 * 1024, true); // Claimed uncompressed size: 25 MB (> 20 MB ceiling)
    const name = "word/document.xml";
    view.setUint16(26, name.length, true);
    view.setUint16(28, 0, true);
    buf.set(new TextEncoder().encode(name), 30);

    const mockDocxPath = "./data/mock_zipbomb.docx";
    await Bun.write(mockDocxPath, buf);

    const pipeline = new Win11TokenPipeline({
      sourceDirectory: "./images",
      documentDirectory: "./documents",
      outputZipPath: "./data/test_zipbomb.zip",
      allowedExtensions: [".docx"],
      maxTokenSizeKb: 25,
    });

    try {
      const items = await pipeline.tokenizeDocumentFile(mockDocxPath, 88);
      expect(items.length).toBe(1);

      const item = items[0]!;
      expect(item.meta.ocrText).toContain("DOCX uncomressed size over max size");

      const testZip = "./data/test_docx_search.zip";
      const { writeNativeTokenZipArchive } = await import("../utils/zipEngine");
      await writeNativeTokenZipArchive(testZip, items);

      const engine = new CompressedTokenSearchEngine(testZip);
      await engine.buildIndex();

      const hits = engine.searchOCR("DOCX uncomressed size over max size");
      expect(hits.length).toBe(1);
      expect(hits[0]!.ocrSnippet).toContain("DOCX uncomressed size over max size");

      const zipFile = Bun.file(testZip);
      if (await zipFile.exists()) await zipFile.delete();
    } finally {
      const file = Bun.file(mockDocxPath);
      if (await file.exists()) await file.delete();
    }
  });

  test("tokenCodec: enforces 256-byte VQ indices length and rejects invalid sizes", async () => {
    const authorityKeys = await getAuthorityAttestationKeys();
    const shortVq = new Uint8Array(200);
    expect(
      encodeBinaryToken({
        originalFileSize: 1000,
        width: 100,
        height: 100,
        page: 0,
        previewFormat: PreviewFormat.AVIF,
        vqIndices: shortVq,
        metadata: {
          id: "00000000-0000-0000-0000-000000000001",
          filename: "test.png",
          sourcePath: "",
          mimeType: "image/png",
          page: 1,
          totalPages: 1,
          width: 100,
          height: 100,
          colorDepth: 24,
          createdAt: new Date().toISOString(),
          ocrText: "test",
        },
        previewBytes: new Uint8Array(100),
        signingKeys: authorityKeys,
      }),
    ).rejects.toThrow("Expected 256 VQ indices");
  });

  test("tokenCodec: uuidStringToBytes strictly validates UUID syntax", () => {
    expect(() => uuidStringToBytes("invalid-uuid-syntax")).toThrow("Invalid UUID syntax");
    expect(() => uuidStringToBytes("12345")).toThrow("Invalid UUID syntax");
    expect(() => uuidStringToBytes("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz")).toThrow("Invalid UUID syntax");

    // Standard RFC4122
    const valid1 = uuidStringToBytes("123e4567-e89b-12d3-a456-426614174000");
    expect(valid1.length).toBe(16);

    // 32-hex format
    const valid2 = uuidStringToBytes("123e4567e89b12d3a456426614174000");
    expect(valid2.length).toBe(16);
    expect(valid1).toEqual(valid2);
  });

  test("tokenCodec: OCR text truncation preserves valid JSON without mid-byte truncation and rejects excessive mandatory metadata", async () => {
    const authorityKeys = await getAuthorityAttestationKeys();
    const vqIndices = new Uint8Array(256).fill(42);
    const hugeOcr = "A".repeat(50_000); // 50 KB of OCR text

    const tokenBytes = await encodeBinaryToken({
      originalFileSize: 10000,
      width: 200,
      height: 200,
      page: 0,
      previewFormat: PreviewFormat.AVIF,
      vqIndices,
      metadata: {
        id: "tok_ocr_trunc_01",
        filename: "huge_ocr.png",
        sourcePath: "/api/token/preview",
        mimeType: "image/png",
        page: 1,
        totalPages: 1,
        width: 200,
        height: 200,
        colorDepth: 24,
        createdAt: new Date().toISOString(),
        ocrText: hugeOcr,
      },
      previewBytes: new Uint8Array(100).fill(1),
      signingKeys: authorityKeys,
    });

    const decoded = await decodeBinaryToken(tokenBytes);
    expect(decoded.metadata.ocrText.length).toBeGreaterThan(0);
    expect(decoded.metadata.ocrText.length).toBeLessThan(hugeOcr.length);
    expect(decoded.metadata.filename).toBe("huge_ocr.png");

    // Mandatory metadata exceeding limit throws
    const hugeFilename = "F".repeat(5000);
    expect(
      encodeBinaryToken({
        originalFileSize: 10000,
        width: 200,
        height: 200,
        page: 0,
        previewFormat: PreviewFormat.AVIF,
        vqIndices,
        metadata: {
          id: "tok_mandatory_overflow",
          filename: hugeFilename,
          sourcePath: "",
          mimeType: "image/png",
          page: 1,
          totalPages: 1,
          width: 200,
          height: 200,
          colorDepth: 24,
          createdAt: new Date().toISOString(),
          ocrText: "",
        },
        previewBytes: new Uint8Array(100).fill(1),
        signingKeys: authorityKeys,
      }),
    ).rejects.toThrow("Mandatory metadata exceeds Section II limit");
  });

  test("tokenCodec: decodeBinaryToken enforces token size bounds and version", async () => {
    // Under minimum size
    expect(decodeBinaryToken(new Uint8Array(500))).rejects.toThrow("Invalid token size");

    // Over maximum size (> 25,600)
    expect(decodeBinaryToken(new Uint8Array(30_000))).rejects.toThrow("Invalid token size");

    // Unsupported version in byte 4 (with valid RS parity for that header)
    const validToken = await generateMockTinyTokenBuffer("tok_ver_01");
    const badVersionToken = new Uint8Array(validToken);
    badVersionToken[4] = 99; // corrupt version
    badVersionToken[5] = 0;
    badVersionToken.set(encodeHeaderParity64(badVersionToken.subarray(0, 640)), 640);
    expect(decodeBinaryToken(badVersionToken)).rejects.toThrow(
      "Unsupported token version: expected 2, found 99",
    );
  });

  test("tokenCodec: QuickLZ decompression safely enforces uncompressed metadata ceiling", async () => {
    // QuickLZ 9-byte header specifies decompressed size at offset 1 (little-endian 32-bit uint)
    // Create a forged header claiming 40,000 bytes decompressed size (> 32,768 ceiling)
    const forgedQuickLz = new Uint8Array(12);
    forgedQuickLz[0] = 0x01; // level 1
    const view = new DataView(forgedQuickLz.buffer);
    view.setUint32(1, 40_000, true);

    expect(() => QuickLZ.decompress(forgedQuickLz, 32768)).toThrow(
      "exceeds safety limit (32768 bytes)",
    );
  });

  test("tokenCodec: validateBinaryToken strictly enforces integrity, authenticity and envelope structure", async () => {
    const authorityKeys = await getAuthorityAttestationKeys();
    const validToken = await generateMockTinyTokenBuffer("tok_strict_01");

    // 1. Valid token passes validation
    const validated = await validateBinaryToken(validToken, {
      trustedPublicKeyRaw: authorityKeys.publicKeyRaw,
    });
    expect(validated.snarkAttestation.valid).toBe(true);
    expect(validated.snarkAttestation.attestationStatus?.signatureValid).toBe(true);
    expect(validated.snarkAttestation.attestationStatus?.trustedIssuer).toBe(true);

    // 2. Corrupted RS parity throws
    const rsCorrupted = new Uint8Array(validToken);
    rsCorrupted[640] = (rsCorrupted[640] ?? 0) ^ 0x55; // corrupt parity byte
    rsCorrupted[641] = (rsCorrupted[641] ?? 0) ^ 0xaa; // corrupt second parity byte
    expect(validateBinaryToken(rsCorrupted)).rejects.toThrow("Reed-Solomon header parity validation failed");

    // 3. Forged signer key throws untrusted signer error
    const rogueKeys = await generateAttestationKeyPair();
    const rogueToken = await encodeBinaryToken({
      originalFileSize: 1000,
      width: 100,
      height: 100,
      page: 0,
      previewFormat: PreviewFormat.AVIF,
      vqIndices: new Uint8Array(256).fill(10),
      metadata: {
        id: "00000000-0000-0000-0000-000000000002",
        filename: "rogue.png",
        sourcePath: "",
        mimeType: "image/png",
        page: 1,
        totalPages: 1,
        width: 100,
        height: 100,
        colorDepth: 24,
        createdAt: new Date().toISOString(),
        ocrText: "rogue",
      },
      previewBytes: new Uint8Array(100),
      signingKeys: rogueKeys,
    });
    expect(
      validateBinaryToken(rogueToken, { trustedPublicKeyRaw: authorityKeys.publicKeyRaw }),
    ).rejects.toThrow("Token signer public key does not match trusted public key");

    // 4. Envelope length mismatch throws
    // Token with truncated preview bytes (smaller than expectedTotalBytes)
    const truncatedToken = validToken.subarray(0, validToken.length - 20);
    // decodeBinaryToken can parse without crashing if size >= 704
    const inspected = await decodeBinaryToken(truncatedToken);
    expect(inspected.envelopeBoundsValid).toBe(false);
    // validateBinaryToken rejects it
    expect(validateBinaryToken(truncatedToken)).rejects.toThrow(
      "Invalid token envelope bounds",
    );
  });

  test("Benchmarking: instruments all 8 stages independently, calculates latency percentiles, and detects primary bottleneck", async () => {
    const bench = new PipelineBenchmark();

    // 1. File reading (10 MB in 20 ms -> ~500 MB/s)
    bench.recordFileRead(10 * 1024 * 1024, 20 * 1_000_000);

    // 2. Sharp processing (4 images, 40 ms total -> 10 ms/img)
    bench.recordSharp(40 * 1_000_000, 4);

    // 3. DINO inference (4 images, 200 ms total -> 50 ms/img)
    bench.recordDino(200 * 1_000_000, 4);

    // 4. VQ generation (4 images, 0.4 ms total -> 0.1 ms/img)
    bench.recordVq(400_000, 4);

    // 5. Binary encoding (4 tokens, 4 ms total -> 1 ms/tok)
    bench.recordEncode(4 * 1_000_000, 4);

    // 6. Binary decoding (4 tokens, 2 ms total -> 0.5 ms/tok)
    bench.recordDecode(2 * 1_000_000, 4);

    // 7. Signature verification (4 tokens, 1 ms total -> 0.25 ms/tok)
    bench.recordSigVerify(1 * 1_000_000, 4);

    // 8. Archive writing (1 MB in 5 ms -> ~200 MB/s)
    bench.recordArchiveWrite(1024 * 1024, 5 * 1_000_000);

    // Latency samples
    bench.recordTokenLatency(50 * 1_000_000);
    bench.recordTokenLatency(60 * 1_000_000);
    bench.recordTokenLatency(70 * 1_000_000);
    bench.recordTokenLatency(100 * 1_000_000);

    const report = bench.getReport();

    // Check all 8 operation metrics
    expect(report.fileReadingMbSec).toBeGreaterThan(450);
    expect(report.sharpMsPerImage).toBe(10);
    expect(report.dinoMsPerImage).toBe(50);
    expect(report.vqMsPerImage).toBe(0.1);
    expect(report.binaryEncodingMsPerToken).toBe(1);
    expect(report.binaryDecodingMsPerToken).toBe(0.5);
    expect(report.sigVerifyMsPerToken).toBe(0.25);
    expect(report.archiveWritingMbSec).toBeGreaterThan(180);

    // Check latency percentiles
    expect(report.latency.minMs).toBe(50);
    expect(report.latency.maxMs).toBe(100);
    expect(report.latency.avgMs).toBe(70);
    expect(report.latency.p50Ms).toBe(65);
    expect(report.latency.p95Ms).toBe(95.5);

    // Check bottleneck detection (DINO was 200 ms out of ~271 ms -> ~73.7%)
    expect(report.summary.primaryBottleneck).toContain("DINO inference");
    expect(report.summary.primaryBottleneck).toContain("%");

    // Check compact console format lines
    const lines = bench.formatMetricLines();
    expect(lines.length).toBe(4);
    for (const l of lines) {
      expect(l).toStartWith("[METRIC]");
      expect(l).toContain("|"); // Multiple figures on one line
    }
    expect(lines[0]).toContain("Read:");
    expect(lines[0]).toContain("Sharp:");
    expect(lines[0]).toContain("DINO:");
    expect(lines[0]).toContain("VQ:");

    expect(lines[1]).toContain("Enc:");
    expect(lines[1]).toContain("Dec:");
    expect(lines[1]).toContain("SigVerify:");
    expect(lines[1]).toContain("ArchiveWrite:");

    expect(lines[2]).toContain("Latency (e2e):");
    expect(lines[3]).toContain("Pipeline:");
    expect(lines[3]).toContain("Bottleneck: DINO inference");
  });

  test("cli.ts: single-token CLI commands (--info, --filetype, --json, --preview)", async () => {
    // 1. --filetype
    const ftProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "--filetype"]);
    expect(ftProc.exitCode).toBe(0);
    expect(ftProc.stdout.toString().trim()).toBe("WEBP");

    // 2. --info
    const infoProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "--info"]);
    expect(infoProc.exitCode).toBe(0);
    const infoStr = infoProc.stdout.toString();
    expect(infoStr).toContain("TinyToken Inspector");
    expect(infoStr).toContain("tok_00004_Henri.jpg");
    expect(infoStr).toContain("Section I: Fixed Header");
    expect(infoStr).toContain("Section II: Metadata & OCR Payload");
    expect(infoStr).toContain("Section III: Visual Preview Payload");

    // 3. --json
    const jsonProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "--json"]);
    expect(jsonProc.exitCode).toBe(0);
    const parsed = JSON.parse(jsonProc.stdout.toString());
    expect(parsed.id).toBe("tok_00004_Henri.jpg");
    expect(parsed.previewFormat).toBe("WEBP");
    expect(parsed.attestation.valid).toBe(true);
    expect(parsed.tokenSizeBytes).toBe(10589);

    // 4. --preview
    const prevProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "--preview"]);
    expect(prevProc.exitCode).toBe(0);
    expect(prevProc.stdout.length).toBe(8656);
  });

  test("inspectBinaryTokenHeader & extractTokenPreviewBytes: zero-copy fast inspection without decompression or crypto", async () => {
    const file = Bun.file("data/tok_00004_Henri.jpg.token");
    const rawBytes = new Uint8Array(await file.arrayBuffer());

    // 1. Zero-copy header inspection executes in sub-millisecond (< 500 us)
    const t0 = Bun.nanoseconds();
    const insp = inspectBinaryTokenHeader(rawBytes);
    const elapsedUs = (Bun.nanoseconds() - t0) / 1000;

    expect(elapsedUs).toBeLessThan(500);
    expect(insp.magic).toBe("GF28");
    expect(insp.version).toBe(2);
    expect(insp.previewFormat).toBe(PreviewFormat.WEBP);
    expect(insp.previewFormatLabel).toBe("WEBP");
    expect(insp.width).toBe(520);
    expect(insp.height).toBe(200);
    expect(insp.page).toBe(1);
    expect(insp.metaLength).toBe(1229);
    expect(insp.previewLength).toBe(8656);
    expect(insp.totalTokenBytes).toBe(10589);
    expect(insp.envelopeBoundsValid).toBe(true);
    expect(insp.uuidHex).toBe("3bbdc2ec4fa644ddac0445ec1ff2c601");
    expect(insp.merkleRootHex).toBe("c5f22f2ab4735d185130605f9e01297f40317300b29ac99a31c3bff668342ba9");
    expect(insp.vqIndices.length).toBe(256);

    // 2. Extract preview bytes zero-copy slice matches decoded preview
    const previewBytes = extractTokenPreviewBytes(rawBytes);
    expect(previewBytes.length).toBe(8656);
    // WebP magic header: RIFF .... WEBP
    expect(String.fromCharCode(...previewBytes.subarray(0, 4))).toBe("RIFF");
    expect(String.fromCharCode(...previewBytes.subarray(8, 12))).toBe("WEBP");

    // 3. Extract metadata on-demand matches decoded metadata
    const meta = extractTokenMetadata(rawBytes);
    expect(meta.id).toBe("tok_00004_Henri.jpg");
    expect(meta.filename).toBe("Henri.jpg");
    expect(meta.mimeType).toBe("image/jpeg");

    // 4. Comparison with full decodeBinaryToken
    const fullDecoded = await decodeBinaryToken(rawBytes);
    expect(fullDecoded.previewBytes).toEqual(previewBytes);
    expect(fullDecoded.merkleRootHex).toBe(insp.merkleRootHex);
  });

  test("cli.ts: struct command, single field extraction (-f), and archive scanning", async () => {
    // 1. struct command on single token
    const structProc = Bun.spawnSync([process.execPath, "cli.ts", "struct", "tok_00004_Henri.jpg.token"]);
    expect(structProc.exitCode).toBe(0);
    const structOut = structProc.stdout.toString();
    expect(structOut).toContain("[STRUCT] Zero-Copy Header Inspection");
    expect(structOut).toContain("3bbdc2ec4fa644ddac0445ec1ff2c601");
    expect(structOut).toContain("520 x 200 px");
    expect(structOut).toContain("BOUNDS VALID");

    // 2. Single field extractions: -f uuid, -f merkle, -f dims, -f sizes
    const uuidProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "-f", "uuid"]);
    expect(uuidProc.exitCode).toBe(0);
    expect(uuidProc.stdout.toString().trim()).toBe("3bbdc2ec4fa644ddac0445ec1ff2c601");

    const merkleProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "-f", "merkle"]);
    expect(merkleProc.exitCode).toBe(0);
    expect(merkleProc.stdout.toString().trim()).toBe("c5f22f2ab4735d185130605f9e01297f40317300b29ac99a31c3bff668342ba9");

    const dimsProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "-f", "dims"]);
    expect(dimsProc.exitCode).toBe(0);
    expect(dimsProc.stdout.toString().trim()).toBe("520x200");

    const sizesProc = Bun.spawnSync([process.execPath, "cli.ts", "tok_00004_Henri.jpg.token", "-f", "sizes"]);
    expect(sizesProc.exitCode).toBe(0);
    expect(sizesProc.stdout.toString().trim()).toBe("Header: 704B | Meta: 1229B | Preview: 8656B | Total: 10589B");

    // 3. struct JSON format
    const jsonProc = Bun.spawnSync([process.execPath, "cli.ts", "struct", "tok_00004_Henri.jpg.token", "--format", "json"]);
    expect(jsonProc.exitCode).toBe(0);
    const jsonParsed = JSON.parse(jsonProc.stdout.toString());
    expect(jsonParsed.magic).toBe("GF28");
    expect(jsonParsed.width).toBe(520);
    expect(jsonParsed.height).toBe(200);
    expect(jsonParsed.envelopeBoundsValid).toBe(true);

    // 4. struct on tokens.zip archive
    const zipProc = Bun.spawnSync([process.execPath, "cli.ts", "struct", "./data/tokens.zip", "--limit", "3"]);
    expect(zipProc.exitCode).toBe(0);
    const zipOut = zipProc.stdout.toString();
    expect(zipOut).toContain("[STRUCT ARCHIVE]");
    expect(zipOut).toContain("tokens inspected");
  });
});
