# TinyToken and Manual Registry

Evaluation pipeline, architectural analysis, and hybrid manual registry integration using modern TypeScript and native Bun I/O.

---

## 1. High-Performance Native Ingestion Engine

Maximizes processing speed by bypassing standard Node.js structures, leveraging Bun's native file system I/O, explicit hardware-aligned typed buffers (`Uint16Array`), and asynchronous iteration generators.

```typescript
import { file } from "bun";
import { TokenSimilarityEngine, type VisualTraitCategory, type TokenMetadata } from "./TokenSimilarityEngine";

// ============================================================================
// 1. HIGH-PERFORMANCE NATIVE INGESTION ENGINE
// ============================================================================

export class BunTokenIngestPipeline {
  /**
   * Reads a raw 25KB bit-packed tinytokens file from disk using Bun.file.
   * Leverages internal zero-copy ArrayBuffer extraction.
   */
  public static async readFileToTypedArray(filePath: string): Promise<Uint16Array> {
    const bunFile = file(filePath);
    
    // Fast path native buffer extraction
    const arrayBuffer = await bunFile.arrayBuffer();
    
    // Directly cast the raw memory segment into 16-bit discrete tokens
    return new Uint16Array(arrayBuffer);
  }

  /**
   * Memory-efficient Async Generator to stream and stream-parse 25KB token 
   * footprints across vast datasets without blocking the main event loop threads.
   */
  public static async *streamTokenFolder(filePaths: string[]): AsyncGenerator<{ path: string; tokens: Uint16Array }> {
    for (const path of filePaths) {
      if (await file(path).exists()) {
        const tokens = await this.readFileToTypedArray(path);
        yield { path, tokens };
      }
    }
  }
}

// ============================================================================
// 2. DICTIONARY DEFINITION GENERATOR (COMPUTE RUNTIME)
// ============================================================================

interface FrequencyAccumulator {
  globalCount: number;
  documentTracking: Set<string>; // Prevents memory bloat using string references
}

/**
 * Evaluates dense token occurrences using modern TS structures to build a highly targeted
 * visual vocabulary separating background matrix artifacts from distinct individual silhouettes.
 */
export async function buildDictionaryFromLocalWorkspace(
  filePaths: string[]
): Promise<Map<number, TokenMetadata>> {
  const analyticsMap = new Map<number, FrequencyAccumulator>();
  const totalDocuments = filePaths.length;

  // Stream-ingest data patterns using the TS Async Iterator Generator
  for await (const { path, tokens } of BunTokenIngestPipeline.streamTokenFolder(filePaths)) {
    // Process flat arrays using fast hardware-aligned iteration loops
    const length = tokens.length;
    for (let i = 0; i < length; i++) {
      const tokenId = tokens[i];

      if (!analyticsMap.has(tokenId)) {
        analyticsMap.set(tokenId, { globalCount: 0, documentTracking: new Set() });
      }

      const entry = analyticsMap.get(tokenId)!;
      entry.globalCount++;
      entry.documentTracking.add(path);
    }
  }

  const outputDictionary = new Map<number, TokenMetadata>();

  // Threshold categorization matrix leveraging TS mapping structures
  for (const [tokenId, data] of analyticsMap.entries()) {
    const documentRatio = data.documentTracking.size / totalDocuments;

    let category: VisualTraitCategory = "StructuralGrid";
    let name = `Structural Feature Component [ID: ${tokenId}]`;

    if (documentRatio > 0.80) {
      // Pervasive background matrix structures (Skies, open outback ground plane grids)
      category = "ColorSpace";
      name = `Global Background Matrix Envelope #${tokenId}`;
    } else if (documentRatio > 0.30 && documentRatio <= 0.80) {
      // Common intersecting layers (Vegetation bark grain, localized structural horizons)
      category = "Texture";
      name = `Recurring Surface Texture Signature #${tokenId}`;
    } else if (documentRatio <= 0.30 && data.globalCount > 3) {
      // Localized high-density clusters indicating distinct entities (Familiar animals / human silhouettes)
      category = "Silhouette";
      name = `Focal Entity Signature Artifact #${tokenId}`;
    }

    outputDictionary.set(tokenId, { name, category });
  }

  return outputDictionary;
}

// ============================================================================
// 3. COMPLETE PIPELINE RUNTIME EVALUATION
// ============================================================================

async function runEvaluationPipeline() {
  console.time("⚡ Bun Pipeline Execution Time");

  // Sample listing of locally cached 25KB token asset files
  const tokenFilePaths = ["frame_01.bin", "frame_02.bin", "frame_03.bin"];

  console.log("Reading memory buffers and analyzing token loops...");
  const runtimeDictionary = await buildDictionaryFromLocalWorkspace(tokenFilePaths);
  console.log(`Generated ${runtimeDictionary.size} persistent vocabulary mappings.`);

  // Write out the dictionary mapping directly to disk using fast native text streams
  const dictPayload = JSON.stringify(Object.fromEntries(runtimeDictionary), null, 2);
  await Bun.write("generated_codebook_dictionary.json", dictPayload);
  console.log("💾 Codebook dictionary maps successfully saved to disk via Bun.write.");

  // Spin up Similarity Engine with the fresh TS Map
  const similarityEngine = new TokenSimilarityEngine(runtimeDictionary);

  if (tokenFilePaths.length >= 2) {
    const targetA = await BunTokenIngestPipeline.readFileToTypedArray(tokenFilePaths[0]);
    const targetB = await BunTokenIngestPipeline.readFileToTypedArray(tokenFilePaths[1]);

    const report = similarityEngine.compareAndName(targetA, targetB);

    console.log("\n=============================================");
    console.log("        STRUCTURAL COGNITION EVAL REPORT      ");
    console.log("=============================================");
    console.log(`Global Content Alignment (Cosine): ${(report.cosineScore * 100).toFixed(2)}%`);
    console.log(`Category Intersect Ratio (Jaccard): ${(report.jaccardScore * 100).toFixed(2)}%`);
    console.log(`Dominant Signal Matrix Path     : ${report.dominantDrivers.join(" -> ")}`);
    console.log("---------------------------------------------");
    console.log("Top Identified Shared Elements:");
    report.sharedElements.slice(0, 3).forEach((el) => {
      console.log(` * [${el.category}] ${el.name} (Densities: ${el.occurrencesInTarget}x vs ${el.occurrencesInCache}x)`);
    });
  }

  console.timeEnd("⚡ Bun Pipeline Execution Time");
}

runEvaluationPipeline().catch(console.error);
```

### Native Bun & TypeScript Features Utilized
- **`Bun.file()` & `arrayBuffer()`**: Maximizes memory efficiency by interacting directly with low-level kernel system calls, avoiding intermediate string conversions or buffer overheads.
- **`AsyncGenerator (async function *)` & `for await`**: Streams file data sequentially without overloading the heap, keeping runtime fast and efficient even under continuous extraction loops.
- **`Bun.write()`**: Bypasses legacy streams for outputting structured JSON definitions, saving computed configuration tables to disk instantly.
- **TS `Map` and `Set` Serialization**: Leveraging built-in constructs like `Object.fromEntries(runtimeDictionary)` handles type transformations quickly and cleanly.

---

## 2. Capabilities & Constraints Analysis

### What the System CAN Do
- **Process Local Binary Data with Native Speed**: Eliminates Node.js layer overhead by using `Bun.file()` for zero-copy reads directly into hardware-aligned typed buffers (`Uint16Array`).
- **Handle Huge Image Footprints in Low Memory**: Streams and evaluates thousands of vision files sequentially using TypeScript `AsyncGenerator`, preventing memory bloat by processing files one at a time.
- **Perform Fast, Integer-Only Image Matching**: Maps visual grids into discrete codebook vector IDs (integers), skipping heavy floating-point vector calculations by running fast integer-overlap algorithms (Jaccard Set and Cosine Vector Histograms).
- **Discover "Familiar" Objects Without Manual Labels**: Analyzes token distributions across files to automatically build a dictionary. Isolates rare, high-density token groupings as **Silhouettes** while marking widespread patterns as **ColorSpaces** (background noise).
- **Generate Clear, Explainable Similarity Reports**: Instead of returning a single opaque numerical score, maps matched tokens to your custom dictionary (e.g., *“84% structural match driven by shared Texture signatures”*).

### What the System CANNOT Do
- **Cannot Generate Initial Raw Tokens Alone**: Downstream analysis tool; requires DINOv2 / VQ inference pass to turn raw `.jpg`/`.png` files into 25 KB matrices.
- **Cannot Match Rotated or Transformed Visual Layouts**: Because tokens map to an explicit 2D spatial grid (e.g., 16×16 patch matrix), severe rotations or crops change token positions and reduce similarity scores.
- **Cannot Know Semantic Object Names Without Examples**: Statistical distribution flags that an entity exists, but cannot infer human labels (e.g., *"Henri"* or *"Kangaroo"*) unless manually registered in the metadata map.
- **Cannot Persist State Interactively Without External Storage**: Dumps JSON configuration maps via `Bun.write`, but does not provide an interactive querying database across sessions without an integrated SQLite database or indexed cache.

---

## 3. Preparing TinyToken for Manual Registration

To support manually registering specific token IDs alongside automated discovery, TinyToken uses a **hybrid dictionary architecture**. High-confidence tokens (specific human silhouettes, known animal signatures, terrain markers) are explicitly mapped to human-readable names, while automated frequency loops handle unmapped background noise.

### 1. Updated `TokenSimilarityEngine.ts`

```typescript
export type VisualTraitCategory = "Silhouette" | "Texture" | "ColorSpace" | "StructuralGrid" | "Unknown";

export interface TokenMetadata {
  name: string;
  category: VisualTraitCategory;
  isManuallyVerified?: boolean; // Protects manual overrides from automated clobbering
}

export class TokenSimilarityEngine {
  public dictionary: Map<number, TokenMetadata> = new Map();

  constructor(initialRegistry?: Record<number, TokenMetadata>) {
    if (initialRegistry) {
      this.loadRegistry(initialRegistry);
    }
  }

  /**
   * Explicitly maps a specific token ID to a known object or person.
   */
  public registerToken(tokenId: number, name: string, category: VisualTraitCategory): void {
    this.dictionary.set(tokenId, { name, category, isManuallyVerified: true });
  }

  /**
   * Bulk loads a saved configuration registry profile.
   */
  public loadRegistry(registry: Record<number, TokenMetadata>): void {
    for (const [id, meta] of Object.entries(registry)) {
      this.dictionary.set(Number(id), meta);
    }
  }
}
```

### 2. Workspace Wiring Script (`app.ts`)

```typescript
import { file, write } from "bun";
import { TokenSimilarityEngine, type TokenMetadata } from "./TokenSimilarityEngine";
import { BunTokenIngestPipeline, buildDictionaryFromLocalWorkspace } from "./PipelineEngine";

const REGISTRY_PATH = "./manual_token_registry.json";

// ============================================================================
// 1. INITIALIZE & BOOTSTRAP YOUR MANUAL REGISTRY
// ============================================================================
async function loadManualRegistry(): Promise<Record<number, TokenMetadata>> {
  const registryFile = file(REGISTRY_PATH);
  
  if (await registryFile.exists()) {
    return await registryFile.json();
  }
  
  // Baseline manual mappings if no saved file exists yet
  return {
    108: { name: "Familiar Red Kangaroo Silhouette", category: "Silhouette", isManuallyVerified: true },
    215: { name: "Risto Ikonen (Facial Alignment Structure)", category: "Silhouette", isManuallyVerified: true },
    512: { name: "Sutton Arid Dirt Terrain Profile", category: "ColorSpace", isManuallyVerified: true }
  };
}

// ============================================================================
// 2. RUN THE COMBINED WORKFLOW PIPELINE
// ============================================================================
async function bootstrapTinytokensApp() {
  console.log("Loading manually registered tokens...");
  const manualRegistry = await loadManualRegistry();
  
  // Spin up engine pre-seeded with verified identities
  const similarityEngine = new TokenSimilarityEngine(manualRegistry);
  console.log(`Loaded ${similarityEngine.dictionary.size} manual verification vectors.`);

  // Define target frame targets to analyze
  const localBinFiles = ["frame_01.bin", "frame_02.bin", "frame_03.bin"];

  console.log("\nRunning automated BunTokenIngestPipeline over workspace...");
  const discoveredMap = await buildDictionaryFromLocalWorkspace(localBinFiles);

  // Merge discovered patterns into the engine WITHOUT overwriting manual records
  for (const [tokenId, discoveredMeta] of discoveredMap.entries()) {
    if (!similarityEngine.dictionary.has(tokenId)) {
      similarityEngine.dictionary.set(tokenId, discoveredMeta);
    }
  }
  console.log(`Pipeline complete. Total active dictionary entries: ${similarityEngine.dictionary.size}`);

  // ============================================================================
  // 3. PERSIST THE SYSTEM STATE
  // ============================================================================
  const fullDictionaryPayload = JSON.stringify(Object.fromEntries(similarityEngine.dictionary), null, 2);
  await write("./generated_complete_codebook.json", fullDictionaryPayload);
  console.log("💾 Master vocabulary saved to 'generated_complete_codebook.json'.");

  // ============================================================================
  // 4. RUN COGNITIVE COMPARISON LOOKUPS
  // ============================================================================
  if (localBinFiles.length >= 2) {
    const freshScan = await BunTokenIngestPipeline.readFileToTypedArray(localBinFiles[0]);
    const cachedScan = await BunTokenIngestPipeline.readFileToTypedArray(localBinFiles[1]);

    const report = similarityEngine.compareAndName(freshScan, cachedScan);
    
    console.log("\n=== EVALUATION REPORT ===");
    console.log(`Structural Match: ${(report.cosineScore * 100).toFixed(2)}%`);
    console.log("Identified Shared Features:");
    report.sharedElements.slice(0, 5).forEach(el => {
      const marker = el.isManuallyVerified ? "🌟 [VERIFIED]" : "🤖 [AUTO-DISCOVERED]";
      console.log(`  * ${marker} ${el.name} (Seen ${el.occurrencesInTarget}x)`);
    });
  }
}

bootstrapTinytokensApp().catch(console.error);
```

---

## 4. Step-by-Step App Workflow Strategy

1. **Define Signatures Manually**: Populate `manual_token_registry.json` with exact token IDs isolated during tests for core subjects (e.g., facial geometry or terrain markers).
2. **Execute Ingest Processing**: `BunTokenIngestPipeline` processes the flat binary packages via zero-copy buffering.
3. **Safe Memory Merging**: When evaluating shared tokens, if a token ID is already locked down by your manual registry, the engine respects the override flag (`isManuallyVerified: true`) and protects your human-defined names.
4. **Inspect & Refine**: Inspect `generated_complete_codebook.json` after an evaluation pass. When an auto-discovered element (*"Structural Pattern Element #44"*) recurs with high frequency alongside a specific subject, copy it into `manual_token_registry.json` and assign it a permanent human label.

