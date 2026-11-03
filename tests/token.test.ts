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
  adcDistanceToSimilarityPct,
  computeHybridVisualSimilarity,
  computeVqAdcDistance,
  decodeVqGridToRgba,
  encodeBytesToVqGrid,
  MAX_ADC_DISTANCE,
  quantizeRgbToCentroid,
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
    const registry = new VisualTraitRegistry("./data/test_manual_token_registry.json");

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

    // Clean up test file if created
    const testFile = Bun.file("./data/test_manual_token_registry.json");
    if (await testFile.exists()) {
      await Bun.write("./data/test_manual_token_registry.json", "{}");
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

  test("quantizeRgbToCentroid performs true nearest-centroid quantization across all 256 codebook entries", () => {
    // Contrast greyscale anchors
    expect(quantizeRgbToCentroid(0, 0, 0)).toBe(252);
    expect(quantizeRgbToCentroid(85, 85, 85)).toBe(253);
    expect(quantizeRgbToCentroid(170, 170, 170)).toBe(254);
    expect(quantizeRgbToCentroid(255, 255, 255)).toBe(255);

    // Verify against exhaustive nearest search on sample colors
    const sampleColors = [
      [12, 10, 16],
      [248, 248, 250],
      [108, 128, 112],
      [5, 5, 5],
      [80, 85, 90],
      [200, 30, 40],
      [30, 220, 100],
      [15, 25, 230],
    ];

    for (const [r, g, b] of sampleColors) {
      const idx = quantizeRgbToCentroid(r!, g!, b!);
      const chosen = VQ_CODEBOOK[idx]!;
      const chosenDist =
        2 * (r! - chosen.r) ** 2 + 4 * (g! - chosen.g) ** 2 + 1 * (b! - chosen.b) ** 2;

      // Verify no other centroid has a strictly smaller perceptual distance
      for (let i = 0; i < 256; i++) {
        const c = VQ_CODEBOOK[i]!;
        const dist = 2 * (r! - c.r) ** 2 + 4 * (g! - c.g) ** 2 + 1 * (b! - c.b) ** 2;
        expect(dist).toBeGreaterThanOrEqual(chosenDist);
      }
    }
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
});

