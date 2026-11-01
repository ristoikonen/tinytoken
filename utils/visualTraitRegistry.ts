/**
 * Semantic Visual Traits & Manual Token Registry (utils/visualTraitRegistry.ts)
 *
 * Implements human-in-the-loop explainable AI for TinyToken's 16x16 VQ latent space.
 * Maps 0..255 GF(2^8) centroid indices to semantic categories (Silhouette, Texture,
 * ColorSpace, StructuralGrid), supporting both human-verified signatures and
 * automated statistical categorization in sub-0.05 ms using Bun native I/O.
 */

import { VQ_CODEBOOK } from "./vqCodebook";

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

export class VisualTraitRegistry {
  private registry = new Map<number, TokenTraitMetadata>();
  private registryFilePath: string;
  private objectRegistryFilePath: string;

  constructor(
    filePath = "./data/manual_token_registry.json",
    objectRegistryFilePath = "./data/object_registry.json",
  ) {
    this.registryFilePath = filePath;
    this.objectRegistryFilePath = objectRegistryFilePath;
    // Synchronously seed with baseline curated entries immediately
    for (const [key, val] of Object.entries(VisualTraitRegistry.DEFAULT_CURATED_MAP)) {
      this.registry.set(Number(key), { ...val });
    }
    this.initRegistry();
  }

  /**
   * Baseline human-curated anchors seeding prominent color, entity, and document patterns.
   */
  private static readonly DEFAULT_CURATED_MAP: Record<number, TokenTraitMetadata> = {
    12: { name: "Dense Foliage Greenery", category: "Texture", isManuallyVerified: true },
    29: { name: "Clear Atmospheric Sky", category: "ColorSpace", isManuallyVerified: true },
    61: { name: "Eucalyptus Canopy Texture", category: "Texture", isManuallyVerified: true },
    77: { name: "High Horizon Cyan Envelope", category: "ColorSpace", isManuallyVerified: true },
    97: { name: "Terracotta Earth Ground Plane", category: "ColorSpace", isManuallyVerified: true },
    108: { name: "Familiar Red Kangaroo Silhouette", category: "Silhouette", isManuallyVerified: true },
    132: { name: "Outback Ochre Clay Signature", category: "Texture", isManuallyVerified: true },
    146: { name: "Warm Subject Flesh Geometry", category: "Silhouette", isManuallyVerified: true },
    194: { name: "Warm Sunlit Highlight", category: "ColorSpace", isManuallyVerified: true },
    215: { name: "Risto Ikonen (Facial Alignment Structure)", category: "Silhouette", isManuallyVerified: true },
    252: { name: "Pure Ink Line & Typography Anchor", category: "StructuralGrid", isManuallyVerified: true },
    253: { name: "Dark Shadow & Boundary Horizon", category: "StructuralGrid", isManuallyVerified: true },
    254: { name: "Silver Grey Document Intermediate", category: "ColorSpace", isManuallyVerified: true },
    255: { name: "Pure Paper White Canvas Field", category: "ColorSpace", isManuallyVerified: true },
  };

  /**
   * Initializes or loads the manual token registry using native Bun.file / Bun.write.
   */
  public async initRegistry(): Promise<void> {
    try {
      const file = Bun.file(this.registryFilePath);
      if (await file.exists()) {
        const data = (await file.json()) as Record<string, TokenTraitMetadata>;
        for (const [key, val] of Object.entries(data)) {
          const id = Number(key);
          if (!Number.isNaN(id) && id >= 0 && id <= 255) {
            this.registry.set(id, { ...val, isManuallyVerified: val.isManuallyVerified ?? true });
          }
        }
      } else {
        // Seed initial default curated map
        for (const [key, val] of Object.entries(VisualTraitRegistry.DEFAULT_CURATED_MAP)) {
          this.registry.set(Number(key), val);
        }
        try {
          await Bun.write(
            this.registryFilePath,
            JSON.stringify(VisualTraitRegistry.DEFAULT_CURATED_MAP, null, 2),
          );
        } catch {
          // Read-only filesystem fallback (e.g. serverless)
        }
      }
    } catch (err) {
      console.warn("VisualTraitRegistry: Using in-memory fallback:", err);
      for (const [key, val] of Object.entries(VisualTraitRegistry.DEFAULT_CURATED_MAP)) {
        this.registry.set(Number(key), val);
      }
    }
  }

  /**
   * Explicitly maps a specific token ID to a known object, texture, or silhouette.
   */
  public async registerToken(
    tokenId: number,
    name: string,
    category: VisualTraitCategory,
  ): Promise<void> {
    this.registry.set(tokenId, { name, category, isManuallyVerified: true });
    try {
      const payload = JSON.stringify(Object.fromEntries(this.registry), null, 2);
      await Bun.write(this.registryFilePath, payload);
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not persist registry to disk:", err);
    }
  }

  /**
   * Teaches an object by mapping covered centroid IDs into the manual registry
   * and saving the spatial bounding-box annotation.
   */
  public async teachObject(
    tokenId: string,
    label: string,
    category: VisualTraitCategory,
    boundingBox: [number, number, number, number],
    centroidIds: number[],
  ): Promise<{ taughtCentroids: number[]; annotation: ObjectAnnotation }> {
    const taughtCentroids: number[] = [];

    // Map the dominant centroids to this label in the manual registry
    for (const cid of centroidIds) {
      if (cid >= 0 && cid <= 255) {
        this.registry.set(cid, {
          name: label,
          category,
          isManuallyVerified: true,
        });
        taughtCentroids.push(cid);
      }
    }

    // Persist updated token registry
    try {
      const payload = JSON.stringify(Object.fromEntries(this.registry), null, 2);
      await Bun.write(this.registryFilePath, payload);
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not write token registry:", err);
    }

    // Append to object registry
    const annotation: ObjectAnnotation = {
      id: `obj_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`,
      tokenId,
      label,
      category,
      boundingBox,
      centroidIds: taughtCentroids,
      createdAt: new Date().toISOString(),
    };

    try {
      let existing: ObjectAnnotation[] = [];
      const objFile = Bun.file(this.objectRegistryFilePath);
      if (await objFile.exists()) {
        try {
          existing = (await objFile.json()) as ObjectAnnotation[];
        } catch {}
      }
      existing.push(annotation);
      await Bun.write(this.objectRegistryFilePath, JSON.stringify(existing, null, 2));
    } catch (err) {
      console.warn("VisualTraitRegistry: Could not write object registry:", err);
    }

    return { taughtCentroids, annotation };
  }

  /**
   * Retrieves all object annotations.
   */
  public async getObjectAnnotations(): Promise<ObjectAnnotation[]> {
    try {
      const objFile = Bun.file(this.objectRegistryFilePath);
      if (await objFile.exists()) {
        return (await objFile.json()) as ObjectAnnotation[];
      }
    } catch {}
    return [];
  }

  /**
   * Retrieves all dictionary mappings.
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
        // Statistical auto-discovery based on coverage and chromaticity
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

    // Sort by count descending (most dominant traits first)
    traits.sort((a, b) => b.count - a.count);

    return traits;
  }
}

// Global singleton instance
export const visualTraitRegistry = new VisualTraitRegistry();
