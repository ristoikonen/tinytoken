/**
 * Semantic Visual Traits & Multi-Tier Knowledge Registry (utils/visualTraitRegistry.ts)
 *
 * Implements human-in-the-loop explainable AI for TinyToken's 16x16 VQ latent space.
 * Enforces the 3-tier persistent intelligence model sitting above TinyToken's binary representation:
 * 1. Level 1: Centroid Intelligence (manual_token_registry.json) - Deterministic RGB traits
 * 2. Level 2 & 3: Object Intelligence (object_registry.json) - Token-ID agnostic versioned recipes
 * 3. Observation Ledger (learning_evidence.json) - Append-oriented groundings with lat/long & UTC time
 */

import type {
  AbstractObjectRecipe,
  CentroidTraitDefinition,
  LearningEvidenceRecord,
  LearningSource,
} from "../models/learning";
import { VQ_CODEBOOK } from "./vqCodebook";

export type {
  AbstractObjectRecipe,
  CentroidTraitDefinition,
  LearningEvidenceRecord,
  LearningSource,
};

export type VisualTraitCategory =
  | "Silhouette"
  | "Texture"
  | "ColorSpace"
  | "StructuralGrid"
  | "Unknown";

export interface TokenTraitMetadata {
  name: string;
  category: VisualTraitCategory;
  isManuallyVerified?: boolean;
  rgb?: [number, number, number];
  source?: LearningSource;
  confidence?: number;
  updatedAt?: string;
}

export interface ObjectAnnotation {
  id: string;
  tokenId: string;
  label: string;
  category: VisualTraitCategory;
  boundingBox: [number, number, number, number]; // [normX, normY, normW, normH] 0..1
  centroidIds: number[];
  createdAt: string;
}

export interface EvaluatedTrait {
  tokenId: number;
  name: string;
  category: VisualTraitCategory;
  isManuallyVerified: boolean;
  count: number;
  percentage: number;
  rgb: { r: number; g: number; b: number };
}

// Default observation coordinates (Sydney / NSW baseline)
export const DEFAULT_OBSERVATION_LATITUDE = -33.8688;
export const DEFAULT_OBSERVATION_LONGITUDE = 151.2093;

export class VisualTraitRegistry {
  private registry = new Map<number, TokenTraitMetadata>();
  private registryFilePath: string;
  private objectRegistryFilePath: string;
  private learningEvidenceFilePath: string;
  private isDefaultPaths: boolean;

  constructor(
    filePath?: string,
    objectRegistryFilePath?: string,
    learningEvidenceFilePath?: string,
  ) {
    const isVercel =
      (Bun.env.VERCEL === "1" || Bun.env.VERCEL === "true" || Bun.env.APP_VERCEL === "true") &&
      Bun.env.VERCEL !== "false";
    const dataDir = Bun.env.DATA_DIR ?? (isVercel ? "/tmp/data" : "./data");

    this.isDefaultPaths = !filePath && !objectRegistryFilePath && !learningEvidenceFilePath;
    this.registryFilePath = filePath ?? `${dataDir}/manual_token_registry.json`;
    this.objectRegistryFilePath = objectRegistryFilePath ?? `${dataDir}/object_registry.json`;
    this.learningEvidenceFilePath = learningEvidenceFilePath ?? `${dataDir}/learning_evidence.json`;

    // Seed in-memory map with baseline curated entries
    for (const [key, val] of Object.entries(VisualTraitRegistry.DEFAULT_CURATED_MAP)) {
      this.registry.set(Number(key), { ...val });
    }
    this.initRegistry();
  }

  /**
   * Baseline human-curated anchors seeding prominent color, entity, and document patterns.
   */
  private static readonly DEFAULT_CURATED_MAP: Record<number, TokenTraitMetadata> = {
    12: { name: "Dense Foliage Greenery", category: "Texture", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    29: { name: "Clear Atmospheric Sky", category: "ColorSpace", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    61: { name: "Eucalyptus Canopy Texture", category: "Texture", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    77: { name: "High Horizon Cyan Envelope", category: "ColorSpace", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    97: { name: "Terracotta Earth Ground Plane", category: "ColorSpace", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    108: { name: "Familiar Red Kangaroo Silhouette", category: "Silhouette", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    132: { name: "Outback Ochre Clay Signature", category: "Texture", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    146: { name: "Warm Subject Flesh Geometry", category: "Silhouette", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    194: { name: "Warm Sunlit Highlight", category: "ColorSpace", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    215: { name: "Risto Ikonen (Facial Alignment Structure)", category: "Silhouette", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    252: { name: "Pure Ink Line & Typography Anchor", category: "StructuralGrid", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    253: { name: "Dark Shadow & Boundary Horizon", category: "StructuralGrid", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    254: { name: "Silver Grey Document Intermediate", category: "ColorSpace", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
    255: { name: "Pure Paper White Canvas Field", category: "ColorSpace", isManuallyVerified: true, source: "human_verified", confidence: 1.0 },
  };

  /**
   * Initializes registries and migrates legacy object_registry format if necessary.
   */
  public async initRegistry(): Promise<void> {
    try {
      // 1. Load Level 1 manual token registry
      let file = Bun.file(this.registryFilePath);
      if (!(await file.exists()) && this.isDefaultPaths && this.registryFilePath !== "./data/manual_token_registry.json") {
        const fallback = Bun.file("./data/manual_token_registry.json");
        if (await fallback.exists()) {
          file = fallback;
        }
      }
      if (await file.exists() && file.size > 2) {
        try {
          const data = (await file.json()) as Record<string, TokenTraitMetadata>;
          for (const [key, val] of Object.entries(data)) {
            const id = Number(key);
            if (!Number.isNaN(id) && id >= 0 && id <= 255) {
              this.registry.set(id, { ...val, isManuallyVerified: val.isManuallyVerified ?? true });
            }
          }
        } catch {}
      } else {
        for (const [key, val] of Object.entries(VisualTraitRegistry.DEFAULT_CURATED_MAP)) {
          this.registry.set(Number(key), val);
        }
        try {
          await Bun.write(
            this.registryFilePath,
            JSON.stringify(VisualTraitRegistry.DEFAULT_CURATED_MAP, null, 2),
          );
        } catch {}
      }

      // 2. Migrate legacy object_registry.json if it contains token-dependent records
      await this.ensureStoresSeparated();
    } catch (err) {
      console.warn("VisualTraitRegistry: Using in-memory fallback:", err);
    }
  }

  /**
   * Transparently migrates any legacy object_registry.json with tokenIds
   * into clean AbstractObjectRecipe[] and learning_evidence.json records.
   */
  private async ensureStoresSeparated(): Promise<void> {
    try {
      let objFile = Bun.file(this.objectRegistryFilePath);
      if (!(await objFile.exists()) && this.isDefaultPaths && this.objectRegistryFilePath !== "./data/object_registry.json") {
        const fallback = Bun.file("./data/object_registry.json");
        if (await fallback.exists()) {
          objFile = fallback;
        }
      }
      if (!(await objFile.exists())) return;

      const raw = (await objFile.json()) as any[];
      if (!Array.isArray(raw) || raw.length === 0) return;

      // Check if file contains legacy records with `tokenId`
      const hasLegacy = raw.some((item) => typeof item.tokenId === "string");
      if (!hasLegacy) return;

      let evidenceFile = Bun.file(this.learningEvidenceFilePath);
      if (!(await evidenceFile.exists()) && this.isDefaultPaths && this.learningEvidenceFilePath !== "./data/learning_evidence.json") {
        const fallback = Bun.file("./data/learning_evidence.json");
        if (await fallback.exists()) {
          evidenceFile = fallback;
        }
      }
      let existingEvidence: LearningEvidenceRecord[] = [];
      if (await evidenceFile.exists()) {
        try {
          existingEvidence = (await evidenceFile.json()) as LearningEvidenceRecord[];
        } catch {}
      }

      const recipeMap = new Map<string, AbstractObjectRecipe>();

      for (const item of raw) {
        const label = String(item.label || "Unknown Object").trim();
        const recipeId = this.generateRecipeId(label);
        const category = (item.category || "Silhouette") as VisualTraitCategory;
        const centroids = Array.isArray(item.centroidIds) ? item.centroidIds : [];

        // Build normalized centroid frequency for recipe
        const dist = this.computeCentroidDistribution(centroids);

        if (!recipeMap.has(recipeId)) {
          recipeMap.set(recipeId, {
            recipeId,
            label,
            category,
            dominantCentroidDist: dist,
            source: "human_verified",
            confidence: 1.0,
            version: 1,
            observationCount: 1,
            createdAt: item.createdAt || new Date().toISOString(),
            updatedAt: item.createdAt || new Date().toISOString(),
          });
        } else {
          const rec = recipeMap.get(recipeId)!;
          rec.observationCount = (rec.observationCount || 1) + 1;
          rec.dominantCentroidDist = this.blendDistributions(rec.dominantCentroidDist, dist);
          rec.updatedAt = item.createdAt || new Date().toISOString();
        }

        // Add to evidence ledger if not already recorded
        const evidenceId = item.id || `evi_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
        if (!existingEvidence.some((e) => e.evidenceId === evidenceId)) {
          existingEvidence.push({
            evidenceId,
            recipeId,
            label,
            category,
            observedTokenId: item.tokenId,
            boundingBox: Array.isArray(item.boundingBox) ? item.boundingBox : [0, 0, 1, 1],
            observedCentroids: centroids,
            latitude: DEFAULT_OBSERVATION_LATITUDE,
            longitude: DEFAULT_OBSERVATION_LONGITUDE,
            utcTimestamp: item.createdAt || new Date().toISOString(),
            observedAt: item.createdAt || new Date().toISOString(),
            source: "human_verified",
            confidence: 1.0,
          });
        }
      }

      // Persist separated stores
      await Bun.write(this.objectRegistryFilePath, JSON.stringify(Array.from(recipeMap.values()), null, 2));
      await Bun.write(this.learningEvidenceFilePath, JSON.stringify(existingEvidence, null, 2));
    } catch (err) {
      console.warn("VisualTraitRegistry: Error separating stores:", err);
    }
  }

  /**
   * Explicitly maps a specific centroid ID to a known trait.
   */
  public async registerToken(
    tokenId: number,
    name: string,
    category: VisualTraitCategory,
    source: LearningSource = "human_verified",
    confidence = 1.0,
  ): Promise<void> {
    const rgb = VQ_CODEBOOK[tokenId] ? [VQ_CODEBOOK[tokenId].r, VQ_CODEBOOK[tokenId].g, VQ_CODEBOOK[tokenId].b] as [number, number, number] : undefined;
    this.registry.set(tokenId, {
      name,
      category,
      isManuallyVerified: true,
      rgb,
      source,
      confidence,
      updatedAt: new Date().toISOString(),
    });
    try {
      const payload = JSON.stringify(Object.fromEntries(this.registry), null, 2);
      await Bun.write(this.registryFilePath, payload);
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not persist registry to disk:", err);
    }
  }

  /**
   * Teaches an object annotation:
   * 1. Updates Level 1 centroid mappings in manual_token_registry.json
   * 2. Creates/updates versioned, token-independent AbstractObjectRecipe in object_registry.json
   * 3. Appends concrete grounded observation to learning_evidence.json with geo & UTC timestamps
   */
  public async teachObject(
    tokenId: string,
    label: string,
    category: VisualTraitCategory,
    boundingBox: [number, number, number, number],
    centroidIds: number[],
    options?: {
      source?: LearningSource;
      confidence?: number;
      latitude?: number;
      longitude?: number;
      utcTimestamp?: string;
      observedAt?: string;
      sourceArchiveHash?: string;
      dinoEmbedding?: number[];
    },
  ): Promise<{
    taughtCentroids: number[];
    annotation: ObjectAnnotation;
    evidence: LearningEvidenceRecord;
    recipe: AbstractObjectRecipe;
  }> {
    const taughtCentroids: number[] = [];
    const source: LearningSource = options?.source ?? "human_verified";
    const confidence: number = options?.confidence ?? (source === "human_verified" ? 1.0 : 0.3);
    const nowUtc = new Date().toISOString();
    const utcTimestamp = options?.utcTimestamp ?? nowUtc;
    const observedAt = options?.observedAt ?? nowUtc;
    const latitude = options?.latitude ?? DEFAULT_OBSERVATION_LATITUDE;
    const longitude = options?.longitude ?? DEFAULT_OBSERVATION_LONGITUDE;

    // 1. Update Level 1 manual token registry for covered centroids (deduplicated)
    const uniqueCentroids = Array.from(new Set(centroidIds));
    for (const cid of uniqueCentroids) {
      if (cid >= 0 && cid <= 255) {
        const rgb = VQ_CODEBOOK[cid] ? [VQ_CODEBOOK[cid].r, VQ_CODEBOOK[cid].g, VQ_CODEBOOK[cid].b] as [number, number, number] : undefined;
        this.registry.set(cid, {
          name: label,
          category,
          isManuallyVerified: true,
          rgb,
          source,
          confidence,
          updatedAt: utcTimestamp,
        });
        taughtCentroids.push(cid);
      }
    }

    try {
      await Bun.write(this.registryFilePath, JSON.stringify(Object.fromEntries(this.registry), null, 2));
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not write token registry:", err);
    }

    // 2. Manage Level 2 & 3 Object Recipe (Strictly Token-Agnostic)
    const recipeId = this.generateRecipeId(label);
    const currentDist = this.computeCentroidDistribution(taughtCentroids);
    let recipes: AbstractObjectRecipe[] = [];
    try {
      let objFile = Bun.file(this.objectRegistryFilePath);
      if (!(await objFile.exists()) && this.isDefaultPaths && this.objectRegistryFilePath !== "./data/object_registry.json") {
        const fallback = Bun.file("./data/object_registry.json");
        if (await fallback.exists()) {
          objFile = fallback;
        }
      }
      if (await objFile.exists()) {
        const raw = (await objFile.json()) as any[];
        recipes = raw.filter((r) => r && typeof r.recipeId === "string");
      }
    } catch {}

    let recipeIndex = recipes.findIndex((r) => r.recipeId === recipeId);
    let recipe: AbstractObjectRecipe;

    if (recipeIndex >= 0) {
      const existing = recipes[recipeIndex]!;
      const blendedDist = this.blendDistributions(existing.dominantCentroidDist, currentDist);
      const drift = this.computeDistributionDrift(existing.dominantCentroidDist, blendedDist);
      const isMaterialShift = drift > 0.10 || existing.category !== category;

      recipe = {
        ...existing,
        version: isMaterialShift ? existing.version + 1 : existing.version,
        observationCount: (existing.observationCount || 1) + 1,
        dominantCentroidDist: blendedDist,
        source,
        confidence,
        updatedAt: utcTimestamp,
      };
      recipes[recipeIndex] = recipe;
    } else {
      recipe = {
        recipeId,
        label,
        category,
        dominantCentroidDist: currentDist,
        dinoPrototypeVector: options?.dinoEmbedding,
        source,
        confidence,
        version: 1,
        observationCount: 1,
        createdAt: utcTimestamp,
        updatedAt: utcTimestamp,
      };
      recipes.push(recipe);
    }

    try {
      await Bun.write(this.objectRegistryFilePath, JSON.stringify(recipes, null, 2));
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not write object recipes:", err);
    }

    // 3. Append to Evidence Ledger (learning_evidence.json)
    const evidenceId = `evi_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
    const evidence: LearningEvidenceRecord = {
      evidenceId,
      recipeId,
      label,
      category,
      observedTokenId: tokenId,
      sourceArchiveHash: options?.sourceArchiveHash,
      boundingBox,
      observedCentroids: taughtCentroids,
      latitude,
      longitude,
      utcTimestamp,
      observedAt,
      source,
      confidence,
    };

    try {
      let existingEvidence: LearningEvidenceRecord[] = [];
      let eviFile = Bun.file(this.learningEvidenceFilePath);
      if (!(await eviFile.exists()) && this.isDefaultPaths && this.learningEvidenceFilePath !== "./data/learning_evidence.json") {
        const fallback = Bun.file("./data/learning_evidence.json");
        if (await fallback.exists()) {
          eviFile = fallback;
        }
      }
      if (await eviFile.exists()) {
        try {
          existingEvidence = (await eviFile.json()) as LearningEvidenceRecord[];
        } catch {}
      }
      existingEvidence.push(evidence);
      await Bun.write(this.learningEvidenceFilePath, JSON.stringify(existingEvidence, null, 2));
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not append learning evidence:", err);
    }

    // 4. Return backward-compatible ObjectAnnotation
    const annotation: ObjectAnnotation = {
      id: evidenceId,
      tokenId,
      label,
      category,
      boundingBox,
      centroidIds: taughtCentroids,
      createdAt: observedAt,
    };

    return { taughtCentroids, annotation, evidence, recipe };
  }

  /**
   * Retrieves all abstract, token-independent object recipes.
   */
  public async getRecipes(): Promise<AbstractObjectRecipe[]> {
    try {
      let file = Bun.file(this.objectRegistryFilePath);
      if (!(await file.exists()) && this.isDefaultPaths && this.objectRegistryFilePath !== "./data/object_registry.json") {
        const fallback = Bun.file("./data/object_registry.json");
        if (await fallback.exists()) {
          file = fallback;
        }
      }
      if (await file.exists()) {
        const raw = (await file.json()) as any[];
        return raw.filter((r) => r && typeof r.recipeId === "string");
      }
    } catch {}
    return [];
  }

  /**
   * Retrieves all observation evidence records.
   */
  public async getEvidence(): Promise<LearningEvidenceRecord[]> {
    try {
      let file = Bun.file(this.learningEvidenceFilePath);
      if (!(await file.exists()) && this.isDefaultPaths && this.learningEvidenceFilePath !== "./data/learning_evidence.json") {
        const fallback = Bun.file("./data/learning_evidence.json");
        if (await fallback.exists()) {
          file = fallback;
        }
      }
      if (await file.exists()) {
        return (await file.json()) as LearningEvidenceRecord[];
      }
    } catch {}
    return [];
  }

  /**
   * Retrieves object annotations (backward-compatible view from evidence ledger).
   */
  public async getObjectAnnotations(): Promise<ObjectAnnotation[]> {
    try {
      const evidence = await this.getEvidence();
      if (evidence.length > 0) {
        return evidence.map((e) => ({
          id: e.evidenceId,
          tokenId: e.observedTokenId,
          label: e.label,
          category: e.category,
          boundingBox: e.boundingBox,
          centroidIds: e.observedCentroids,
          createdAt: e.observedAt,
        }));
      }

      // Fallback if legacy object_registry.json hasn't been migrated yet
      let objFile = Bun.file(this.objectRegistryFilePath);
      if (!(await objFile.exists()) && this.isDefaultPaths && this.objectRegistryFilePath !== "./data/object_registry.json") {
        const fallback = Bun.file("./data/object_registry.json");
        if (await fallback.exists()) {
          objFile = fallback;
        }
      }
      if (await objFile.exists()) {
        const raw = (await objFile.json()) as any[];
        return raw.filter((a) => a && typeof a.tokenId === "string");
      }
    } catch {}
    return [];
  }

  /**
   * Retrieves all Level 1 dictionary mappings.
   */
  public getAll(): Record<number, TokenTraitMetadata> {
    return Object.fromEntries(this.registry);
  }

  /**
   * Evaluates a 256-cell VQ grid in ~0.02 ms and returns dominant semantic visual traits.
   */
  public evaluateGrid(vqIndices: Uint8Array): EvaluatedTrait[] {
    if (!vqIndices || vqIndices.length === 0) return [];

    const totalCells = vqIndices.length;
    const freq = new Map<number, number>();

    for (let i = 0; i < totalCells; i++) {
      const id = vqIndices[i]!;
      freq.set(id, (freq.get(id) ?? 0) + 1);
    }

    const traits: EvaluatedTrait[] = [];

    for (const [id, count] of freq.entries()) {
      const pct = Math.round((count / totalCells) * 1000) / 10;
      const rgb = VQ_CODEBOOK[id] ?? { r: 128, g: 128, b: 128 };

      if (this.registry.has(id)) {
        const meta = this.registry.get(id)!;
        traits.push({
          tokenId: id,
          name: meta.name,
          category: meta.category,
          isManuallyVerified: meta.isManuallyVerified ?? false,
          count,
          percentage: pct,
          rgb,
        });
      } else {
        let category: VisualTraitCategory = "StructuralGrid";
        let name = `Structural Feature Component #${id}`;

        if (pct >= 28.0) {
          category = "ColorSpace";
          name = `Dominant Field Matrix Envelope #${id}`;
        } else if (pct >= 8.0 && pct < 28.0) {
          category = "Texture";
          name = `Surface Texture Signature #${id}`;
        } else if (pct >= 2.0 && pct < 8.0) {
          category = "Silhouette";
          name = `Focal Entity Signature #${id}`;
        }

        traits.push({
          tokenId: id,
          name,
          category,
          isManuallyVerified: false,
          count,
          percentage: pct,
          rgb,
        });
      }
    }

    traits.sort((a, b) => b.count - a.count);
    return traits;
  }

  private generateRecipeId(label: string): string {
    const slug = label
      .toLowerCase()
      .trim()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 32);
    return `rcp_${slug || "unnamed"}`;
  }

  private computeCentroidDistribution(centroidIds: number[]): Record<number, number> {
    const counts: Record<number, number> = {};
    for (const cid of centroidIds) {
      if (cid >= 0 && cid <= 255) {
        counts[cid] = (counts[cid] ?? 0) + 1;
      }
    }
    const total = Object.values(counts).reduce((a, b) => a + b, 0) || 1;
    const dist: Record<number, number> = {};
    for (const [k, v] of Object.entries(counts)) {
      dist[Number(k)] = Math.round((v / total) * 1000) / 1000;
    }
    return dist;
  }

  private blendDistributions(
    oldDist: Record<number, number>,
    newDist: Record<number, number>,
    weightNew = 0.5,
  ): Record<number, number> {
    const blended: Record<number, number> = {};
    const allKeys = new Set([...Object.keys(oldDist).map(Number), ...Object.keys(newDist).map(Number)]);
    let sum = 0;

    for (const k of allKeys) {
      const vOld = oldDist[k] ?? 0;
      const vNew = newDist[k] ?? 0;
      const val = vOld * (1 - weightNew) + vNew * weightNew;
      blended[k] = val;
      sum += val;
    }

    if (sum > 0) {
      for (const k of Object.keys(blended).map(Number)) {
        blended[k] = Math.round((blended[k]! / sum) * 1000) / 1000;
      }
    }
    return blended;
  }

  /**
   * Computes the Total Variation Distance (drift) between two normalized centroid distributions (0.0 to 1.0).
   */
  public computeDistributionDrift(
    distA: Record<number, number>,
    distB: Record<number, number>,
  ): number {
    const keys = new Set([...Object.keys(distA).map(Number), ...Object.keys(distB).map(Number)]);
    let sumAbsDiff = 0;
    for (const k of keys) {
      const a = distA[k] ?? 0;
      const b = distB[k] ?? 0;
      sumAbsDiff += Math.abs(a - b);
    }
    return sumAbsDiff / 2;
  }

  /**
   * Matches a token's VQ indices against known abstract object recipes.
   * Returns the best matching recipe if similarity exceeds minSimilarity (default 0.20).
   */
  public matchObjectRecipe(
    vqIndices: Uint8Array,
    recipes: AbstractObjectRecipe[],
    minSimilarity = 0.20,
  ): { recipe: AbstractObjectRecipe; similarity: number } | null {
    if (!vqIndices || vqIndices.length === 0 || recipes.length === 0) return null;
    const tokenDist = this.computeCentroidDistribution(Array.from(vqIndices));

    let bestRecipe: AbstractObjectRecipe | null = null;
    let highestScore = 0;

    for (const recipe of recipes) {
      if (!recipe.dominantCentroidDist) continue;
      let overlap = 0;
      for (const [cidStr, weight] of Object.entries(recipe.dominantCentroidDist)) {
        const cid = Number(cidStr);
        const tokenWeight = tokenDist[cid] ?? 0;
        overlap += Math.min(weight, tokenWeight);
      }

      if (overlap > highestScore) {
        highestScore = overlap;
        bestRecipe = recipe;
      }
    }

    if (bestRecipe && highestScore >= minSimilarity) {
      return { recipe: bestRecipe, similarity: Math.round(highestScore * 100) / 100 };
    }
    return null;
  }
}

// Global singleton instance
export const visualTraitRegistry = new VisualTraitRegistry();
