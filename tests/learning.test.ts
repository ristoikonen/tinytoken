import { afterAll, beforeAll, describe, expect, test } from "bun:test";
import { LearningHandlers } from "../handlers/learningHandlers";
import { ArchiveManager } from "../utils/archiveManager";
import {
  VisualTraitRegistry,
  DEFAULT_OBSERVATION_LATITUDE,
  DEFAULT_OBSERVATION_LONGITUDE,
} from "../utils/visualTraitRegistry";
import { CompressedTokenSearchEngine, writeNativeTokenZipArchive } from "../utils/zipEngine";
import { generateMockTinyTokenBuffer } from "../utils/tokenCodec";
import type { ArchiveTokenItem } from "../utils/indexer";

describe("Persistent Knowledge Layer & Transactional Archive Ingestion", () => {
  const TEST_MANUAL_REGISTRY = "./data/test_manual_traits.json";
  const TEST_OBJECT_REGISTRY = "./data/test_object_recipes.json";
  const TEST_EVIDENCE_STORE = "./data/test_learning_evidence.json";
  const TEST_ACTIVE_ZIP = "./data/test_active_tokens.zip";
  const TEST_CANDIDATE_ZIP = "./data/test_temp_candidate.zip";

  let searchEngine: CompressedTokenSearchEngine;
  let archiveManager: ArchiveManager;
  let visualRegistry: VisualTraitRegistry;
  let learningHandlers: LearningHandlers;

  function createMockArchiveItem(
    id: string,
    tokenBytes: Uint8Array<ArrayBuffer>,
    centroid = 120,
  ): ArchiveTokenItem {
    return {
      entryName: `${id}.token`,
      tokenBytes,
      meta: {
        id,
        entryName: `${id}.token`,
        filename: `${id}.jpg`,
        sourcePath: `images/${id}.jpg`,
        mimeType: "image/jpeg",
        page: 1,
        totalPages: 1,
        width: 320,
        height: 320,
        originalFileSize: tokenBytes.length,
        previewFormat: 2,
        flags: 0,
        ocrText: `${id} synthetic test fixture`,
        createdAt: new Date().toISOString(),
        merkleRootHex: "00".repeat(32),
        snarkVerified: true,
        vqIndices: new Uint8Array(256).fill(centroid),
      },
    };
  }

  beforeAll(async () => {
    // 1. Build initial synthetic active archive (Generation 1: tok_alpha + tok_beta)
    const tokenA = await generateMockTinyTokenBuffer("tok_alpha", "Alpha Test Asset");
    const tokenB = await generateMockTinyTokenBuffer("tok_beta", "Beta Test Asset");

    const m1 = createMockArchiveItem("tok_alpha", tokenA, 120);
    const m2 = createMockArchiveItem("tok_beta", tokenB, 222);

    await writeNativeTokenZipArchive(TEST_ACTIVE_ZIP, [m1, m2]);

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
      TEST_CANDIDATE_ZIP,
    ]) {
      try {
        const file = Bun.file(f);
        if (await file.exists()) {
          await file.delete();
        }
      } catch { }
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

    // 2. Recipe verification (CRITICAL INVARIANT: Never contains tokenId!)
    expect(teachRes.recipe).toBeDefined();
    expect(teachRes.recipe.label).toBe("Charred Pine Bark");
    expect("tokenId" in teachRes.recipe).toBe(false);
    expect("observedTokenId" in teachRes.recipe).toBe(false);
    expect(teachRes.recipe.version).toBe(1);
    expect(teachRes.recipe.observationCount).toBe(1);
    expect(teachRes.recipe.source).toBe("human_verified");
    expect(teachRes.recipe.confidence).toBe(1.0);
    expect(teachRes.recipe.dominantCentroidDist[222]).toBeGreaterThan(0);

    // 3. Second observation updates distribution and increments version
    const teachRes2 = await visualRegistry.teachObject(
      "tok_beta",
      "Charred Pine Bark",
      "Texture",
      [0.2, 0.2, 0.4, 0.4],
      [222, 180],
      { confidence: 1.0 },
    );

    expect(teachRes2.recipe.version).toBe(2);
    expect(teachRes2.recipe.observationCount).toBe(2);
    expect("tokenId" in teachRes2.recipe).toBe(false);
    expect(teachRes2.evidence.observedTokenId).toBe("tok_beta");

    // 4. Repeated observation without material distribution drift keeps version stable and increments observationCount
    const teachRes3 = await visualRegistry.teachObject(
      "tok_gamma",
      "Charred Pine Bark",
      "Texture",
      [0.2, 0.2, 0.4, 0.4],
      [222, 180],
      { confidence: 1.0 },
    );
    expect(teachRes3.recipe.version).toBe(2);
    expect(teachRes3.recipe.observationCount).toBe(3);

    // 5. Query recipes and evidence direct methods
    const recipes = await visualRegistry.getRecipes();
    expect(recipes.length).toBeGreaterThanOrEqual(1);
    expect(recipes[0]?.label).toBe("Charred Pine Bark");

    const evidence = await visualRegistry.getEvidence();
    expect(evidence.length).toBeGreaterThanOrEqual(3);
  });

  test("HTTP Learning Handlers return recipes and evidence and record new observations", async () => {
    // 1. GET /api/learning/recipes
    const getRecipesRes = await learningHandlers.handleGetRecipes();
    expect(getRecipesRes.status).toBe(200);
    const recipesBody = (await getRecipesRes.json()) as unknown as {
      success: boolean;
      count: number;
      recipes: Array<{ label: string }>;
    };
    expect(recipesBody.success).toBe(true);
    expect(recipesBody.count).toBeGreaterThanOrEqual(1);
    expect(recipesBody.recipes.some((r: { label: string }) => r.label === "Charred Pine Bark")).toBe(true);

    // 2. GET /api/learning/evidence
    const getEvidenceRes = await learningHandlers.handleGetEvidence();
    expect(getEvidenceRes.status).toBe(200);
    const evidenceBody = (await getEvidenceRes.json()) as unknown as {
      success: boolean;
      count: number;
    };
    expect(evidenceBody.success).toBe(true);
    expect(evidenceBody.count).toBeGreaterThanOrEqual(3);

    // 3. POST /api/learning/evidence
    const recordReq = new Request("http://localhost:3000/api/learning/evidence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tokenId: "tok_delta",
        label: "Red Kangaroo Fur",
        category: "Texture",
        boundingBox: [0.1, 0.1, 0.6, 0.6],
        centroidIds: [108, 132],
        confidence: 1.0,
      }),
    });

    const recordRes = await learningHandlers.handleRecordEvidence(recordReq);
    expect(recordRes.status).toBe(200);
    const recordBody = (await recordRes.json()) as unknown as {
      success: boolean;
      recipe: { label: string };
      evidence: { observedTokenId: string };
    };
    expect(recordBody.success).toBe(true);
    expect(recordBody.recipe.label).toBe("Red Kangaroo Fur");
    expect(recordBody.evidence.observedTokenId).toBe("tok_delta");
  });

  test("Transactional archive swap rejects malformed/corrupted archive and keeps active generation intact", async () => {
    const initialState = archiveManager.getGenerationState();
    expect(initialState.generation).toBe(1);
    expect(initialState.totalTokens).toBe(2);

    // Corrupted dummy payload
    const garbageBytes = new TextEncoder().encode("THIS IS NOT A VALID ZIP CONTAINER!");

    const badSwapReq = new Request("http://localhost:3000/api/archive/swap", {
      method: "POST",
      headers: { "Content-Type": "application/octet-stream" },
      body: garbageBytes,
    });

    const res = await learningHandlers.handleArchiveSwap(badSwapReq);
    expect(res.status).toBe(400);

    const body = (await res.json()) as unknown as {
      success: boolean;
      activeGeneration: number;
    };
    expect(body.success).toBe(false);
    expect(body.activeGeneration).toBe(1);

    // Active archive remains Generation 1 and untouched
    const currentState = archiveManager.getGenerationState();
    expect(currentState.generation).toBe(1);
    expect(currentState.totalTokens).toBe(2);

    // Staging file was unlinked and cleaned up
    const stagingExists = await Bun.file(`${TEST_ACTIVE_ZIP}.staging`).exists();
    expect(stagingExists).toBe(false);
  });

  test("Transactional archive swap ingests valid candidate archive, increments generation, and preserves all learnings", async () => {
    // Build Generation 2 candidate archive with 3 replacement tokens
    const tok1 = await generateMockTinyTokenBuffer("tok_gamma", "Gamma Replacement 1");
    const tok2 = await generateMockTinyTokenBuffer("tok_delta", "Delta Replacement 2");
    const tok3 = await generateMockTinyTokenBuffer("tok_epsilon", "Epsilon Replacement 3");

    const c1 = createMockArchiveItem("tok_gamma", tok1, 50);
    const c2 = createMockArchiveItem("tok_delta", tok2, 80);
    const c3 = createMockArchiveItem("tok_epsilon", tok3, 110);

    await writeNativeTokenZipArchive(TEST_CANDIDATE_ZIP, [c1, c2, c3]);

    const candidateBytes = new Uint8Array(await Bun.file(TEST_CANDIDATE_ZIP).arrayBuffer());
    await Bun.file(TEST_CANDIDATE_ZIP).delete();

    // Perform hot swap via HTTP handler
    const swapReq = new Request("http://localhost:3000/api/archive/swap", {
      method: "POST",
      headers: { "Content-Type": "application/zip" },
      body: candidateBytes,
    });

    const res = await learningHandlers.handleArchiveSwap(swapReq);
    expect(res.status).toBe(200);

    const body = (await res.json()) as unknown as {
      success: boolean;
      generation: number;
      totalTokens: number;
      learningsPreserved: boolean;
    };
    expect(body.success).toBe(true);
    expect(body.generation).toBe(2);
    expect(body.totalTokens).toBe(3);
    expect(body.learningsPreserved).toBe(true);

    // Active state updated to Generation 2
    const updatedState = archiveManager.getGenerationState();
    expect(updatedState.generation).toBe(2);
    expect(updatedState.totalTokens).toBe(3);

    // In-memory search engine immediately queries new tokens
    expect(searchEngine.getCount()).toBe(3);
    expect(searchEngine.getEntry("tok_gamma")).toBeDefined();
    expect(searchEngine.getEntry("tok_delta")).toBeDefined();
    expect(searchEngine.getEntry("tok_epsilon")).toBeDefined();

    // Verify all persistent learnings remain 100% intact across archive swap
    const recipesAfterSwap = await visualRegistry.getRecipes();
    expect(recipesAfterSwap.some((r) => r.label === "Charred Pine Bark")).toBe(true);
    expect(recipesAfterSwap.some((r) => r.label === "Red Kangaroo Fur")).toBe(true);

    const evidenceAfterSwap = await visualRegistry.getEvidence();
    expect(evidenceAfterSwap.length).toBeGreaterThanOrEqual(4);
  });

  test("GET /api/archive/status reports accurate generation and knowledge counts", async () => {
    const res = await learningHandlers.handleArchiveStatus();
    expect(res.status).toBe(200);

    const data = (await res.json()) as unknown as {
      success: boolean;
      archive: { generation: number; totalTokens: number };
      knowledge: { totalObjectRecipes: number; totalEvidenceObservations: number };
    };
    expect(data.success).toBe(true);
    expect(data.archive.generation).toBe(2);
    expect(data.archive.totalTokens).toBe(3);
    expect(data.knowledge.totalObjectRecipes).toBeGreaterThanOrEqual(2);
    expect(data.knowledge.totalEvidenceObservations).toBeGreaterThanOrEqual(4);
  });
});
