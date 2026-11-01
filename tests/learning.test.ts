import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { LearningHandlers } from "../handlers/learningHandlers";
import { ArchiveManager } from "../utils/archiveManager";
import { VisualTraitRegistry, DEFAULT_OBSERVATION_LATITUDE, DEFAULT_OBSERVATION_LONGITUDE } from "../utils/visualTraitRegistry";
import { CompressedTokenSearchEngine, writeNativeTokenZipArchive } from "../utils/zipEngine";
import { generateMockTinyTokenBuffer } from "../utils/tokenCodec";

describe("Persistent Knowledge Layer & Transactional Archive Ingestion", () => {
  const TEST_MANUAL_REGISTRY = "./data/test_manual_traits.json";
  const TEST_OBJECT_REGISTRY = "./data/test_object_recipes.json";
  const TEST_EVIDENCE_STORE = "./data/test_learning_evidence.json";
  const TEST_ACTIVE_ZIP = "./data/test_active_tokens.zip";

  let searchEngine: CompressedTokenSearchEngine;
  let archiveManager: ArchiveManager;
  let visualRegistry: VisualTraitRegistry;
  let learningHandlers: LearningHandlers;

  function createMockArchiveItem(id: string, tokenBytes: Uint8Array, centroid = 120) {
    return {
      entryName: `${id}.token`,
      tokenBytes,
      meta: {
        id,
        filename: `${id}.token`,
        sourcePath: `images/${id}.jpg`,
        mimeType: "image/jpeg",
        page: 1,
        totalPages: 1,
        width: 320,
        height: 320,
        colorDepth: 24,
        originalFileSize: tokenBytes.length,
        previewFormat: 2,
        flags: 0,
        ocrText: `${id} synthetic test`,
        createdAt: new Date().toISOString(),
        merkleRootHex: "00".repeat(32),
        snarkVerified: true,
        vqIndices: new Uint8Array(256).fill(centroid),
      },
    };
  }

  beforeAll(async () => {
    // 1. Create initial synthetic active archive (Gen 1)
    const tokenA = await generateMockTinyTokenBuffer("tok_alpha", "Alpha Test Asset", 120);
    const tokenB = await generateMockTinyTokenBuffer("tok_beta", "Beta Test Asset", 222);

    await writeNativeTokenZipArchive(
      TEST_ACTIVE_ZIP,
      [
        createMockArchiveItem("tok_alpha", tokenA, 120),
        createMockArchiveItem("tok_beta", tokenB, 222),
      ],
    );

    searchEngine = new CompressedTokenSearchEngine(TEST_ACTIVE_ZIP);
    await searchEngine.buildIndex();

    archiveManager = new ArchiveManager(searchEngine, TEST_ACTIVE_ZIP);
    visualRegistry = new VisualTraitRegistry(
      TEST_MANUAL_REGISTRY,
      TEST_OBJECT_REGISTRY,
      TEST_EVIDENCE_STORE,
    );
    learningHandlers = new LearningHandlers(archiveManager, visualRegistry);
  });

  afterAll(async () => {
    for (const f of [
      TEST_MANUAL_REGISTRY,
      TEST_OBJECT_REGISTRY,
      TEST_EVIDENCE_STORE,
      TEST_ACTIVE_ZIP,
      `${TEST_ACTIVE_ZIP}.staging`,
    ]) {
      try {
        await Bun.file(f).delete();
      } catch {}
    }
  });

  test("Teaching an object creates token-independent recipe and populates evidence with default lat/long & UTC time", async () => {
    const teachRes = await visualRegistry.teachObject(
      "tok_alpha",
      "Charred Pine Bark",
      "Texture",
      [0.1, 0.1, 0.5, 0.5],
      [222, 222, 180, 253],
    );

    expect(teachRes.taughtCentroids).toEqual([222, 180, 253]);
    expect(teachRes.annotation.label).toBe("Charred Pine Bark");

    // 1. Evidence record verification
    expect(teachRes.evidence).toBeDefined();
    expect(teachRes.evidence.evidenceId.startsWith("evi_")).toBe(true);
    expect(teachRes.evidence.observedTokenId).toBe("tok_alpha");
    expect(teachRes.evidence.latitude).toBe(DEFAULT_OBSERVATION_LATITUDE);
    expect(teachRes.evidence.longitude).toBe(DEFAULT_OBSERVATION_LONGITUDE);
    expect(teachRes.evidence.utcTimestamp).toBeDefined();
    expect(teachRes.evidence.observedAt).toBeDefined();
    expect(teachRes.evidence.source).toBe("human_verified");
    expect(teachRes.evidence.confidence).toBe(1.0);

    // 2. Recipe verification (INVARIANT: Never contains tokenId!)
    expect(teachRes.recipe).toBeDefined();
    expect(teachRes.recipe.label).toBe("Charred Pine Bark");
    expect((teachRes.recipe as any).tokenId).toBeUndefined();
    expect((teachRes.recipe as any).observedTokenId).toBeUndefined();
    expect(teachRes.recipe.version).toBe(1);
    expect(teachRes.recipe.source).toBe("human_verified");
    expect(teachRes.recipe.confidence).toBe(1.0);
    expect(teachRes.recipe.dominantCentroidDist[222]).toBeGreaterThan(0);

    // 3. Second observation increments recipe version while keeping it token-agnostic
    const teachRes2 = await visualRegistry.teachObject(
      "tok_beta",
      "Charred Pine Bark",
      "Texture",
      [0.2, 0.2, 0.4, 0.4],
      [222, 180],
      { confidence: 1.0 },
    );

    expect(teachRes2.recipe.version).toBe(2);
    expect((teachRes2.recipe as any).tokenId).toBeUndefined();
    expect(teachRes2.evidence.observedTokenId).toBe("tok_beta");

    // 4. Query recipes and evidence
    const recipes = await visualRegistry.getRecipes();
    expect(recipes.length).toBeGreaterThanOrEqual(1);
    expect(recipes[0]?.label).toBe("Charred Pine Bark");

    const evidence = await visualRegistry.getEvidence();
    expect(evidence.length).toBeGreaterThanOrEqual(2);
  });

  test("Transactional archive swap rejects malformed/corrupted archive and keeps active generation intact", async () => {
    const initialState = archiveManager.getGenerationState();
    expect(initialState.generation).toBe(1);
    expect(initialState.totalTokens).toBe(2);

    // Create corrupted dummy payload
    const garbageBytes = new TextEncoder().encode("THIS IS NOT A VALID ZIP ARCHIVE AT ALL!");

    const badSwapReq = new Request("http://localhost:3000/api/archive/swap", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: garbageBytes,
    });

    const res = await learningHandlers.handleArchiveSwap(badSwapReq);
    expect(res.status).toBe(400);

    const body = await res.json();
    expect(body.success).toBe(false);
    expect(body.activeGeneration).toBe(1);

    // Verify active archive is still Gen 1 and intact
    const currentState = archiveManager.getGenerationState();
    expect(currentState.generation).toBe(1);
    expect(currentState.totalTokens).toBe(2);

    // Verify staging file was cleaned up
    const stagingExists = await Bun.file(`${TEST_ACTIVE_ZIP}.staging`).exists();
    expect(stagingExists).toBe(false);
  });

  test("Transactional archive swap ingests valid candidate archive, increments generation, and preserves all learnings", async () => {
    // Create new Generation 2 archive with 3 different tokens
    const tok1 = await generateMockTinyTokenBuffer("tok_gamma", "Gamma Replacement 1", 50);
    const tok2 = await generateMockTinyTokenBuffer("tok_delta", "Delta Replacement 2", 80);
    const tok3 = await generateMockTinyTokenBuffer("tok_epsilon", "Epsilon Replacement 3", 110);

    const tempCandidatePath = "./data/temp_candidate.zip";
    await writeNativeTokenZipArchive(
      tempCandidatePath,
      [
        createMockArchiveItem("tok_gamma", tok1, 50),
        createMockArchiveItem("tok_delta", tok2, 80),
        createMockArchiveItem("tok_epsilon", tok3, 110),
      ],
    );

    const candidateBytes = new Uint8Array(await Bun.file(tempCandidatePath).arrayBuffer());
    await Bun.file(tempCandidatePath).delete();

    // Perform swap via HTTP handler
    const swapReq = new Request("http://localhost:3000/api/archive/swap", {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: candidateBytes,
    });

    const res = await learningHandlers.handleArchiveSwap(swapReq);
    expect(res.status).toBe(200);

    const body = await res.json();
    expect(body.success).toBe(true);
    expect(body.generation).toBe(2);
    expect(body.totalTokens).toBe(3);
    expect(body.learningsPreserved).toBe(true);

    // Active state updated
    const updatedState = archiveManager.getGenerationState();
    expect(updatedState.generation).toBe(2);
    expect(updatedState.totalTokens).toBe(3);

    // Search engine sees new tokens immediately
    expect(searchEngine.getCount()).toBe(3);
    expect(searchEngine.getEntry("tok_gamma")).toBeDefined();
    expect(searchEngine.getEntry("tok_delta")).toBeDefined();
    expect(searchEngine.getEntry("tok_epsilon")).toBeDefined();

    // Verify all learnings remain 100% intact
    const recipesAfterSwap = await visualRegistry.getRecipes();
    expect(recipesAfterSwap.some((r) => r.label === "Charred Pine Bark")).toBe(true);

    const evidenceAfterSwap = await visualRegistry.getEvidence();
    expect(evidenceAfterSwap.length).toBeGreaterThanOrEqual(2);
  });

  test("GET /api/archive/status reports accurate generation and knowledge counts", async () => {
    const res = await learningHandlers.handleArchiveStatus();
    expect(res.status).toBe(200);

    const data = await res.json();
    expect(data.success).toBe(true);
    expect(data.archive.generation).toBe(2);
    expect(data.archive.totalTokens).toBe(3);
    expect(data.knowledge.totalObjectRecipes).toBeGreaterThanOrEqual(1);
    expect(data.knowledge.totalEvidenceObservations).toBeGreaterThanOrEqual(2);
  });
});
