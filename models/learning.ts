/**
 * Knowledge Layer & Archive Generation Models (models/learning.ts)
 *
 * Implements the 3-tier persistent intelligence model sitting above TinyToken's binary representation:
 * 1. CentroidTraitDefinition (Level 1 Centroid Intelligence): Human-confirmed RGB centroid traits
 * 2. AbstractObjectRecipe (Level 2 & 3 Object Intelligence): Versioned, token-ID agnostic object recipes
 * 3. LearningEvidenceRecord (Observation Ledger): Append-oriented groundings with geo & timestamp
 */

import type { VisualTraitCategory } from "../utils/visualTraitRegistry";

export type LearningSource =
  | "human_verified"
  | "human_suggested"
  | "synthesized_recipe"
  | "model_cluster"
  | "auto_color_namer";

/**
 * Level 1: Centroid Intelligence (manual_token_registry.json)
 * Deterministic mapping from centroid ID (0..255) to human-confirmed semantic traits.
 */
export interface CentroidTraitDefinition {
  centroidId: number;
  name: string;
  category: VisualTraitCategory;
  isManuallyVerified: boolean;
  rgb?: [number, number, number];
  source: LearningSource;
  confidence: number; // 0.0 - 1.0 (default 1.0 for human_verified)
  updatedAt: string;
}

/**
 * Level 2 & 3: Token-Agnostic Object Recipes (object_registry.json)
 * INVARIANT: Never depends on a specific token ID!
 * Feature representations can evolve independently of individual image tokens.
 */
export interface AbstractObjectRecipe {
  recipeId: string; // e.g. "rcp_charred_wood"
  label: string; // e.g. "Charred Tree Trunk"
  category: VisualTraitCategory;
  dominantCentroidDist: Record<number, number>; // Normalized frequency distribution (0..255 -> 0..1)
  spatialAdjacencyPriors?: [number, number][]; // Co-occurring centroid pairs within spatial neighborhood
  dinoPrototypeVector?: number[]; // Normalized 384-dim prototype vector
  boundingBoxAspectRatio?: number; // Normalized width / height ratio
  source: LearningSource;
  confidence: number; // 1.0 for human_verified, 0.3 for suggestion
  version: number; // Starts at 1, increments when recipe is updated/consolidated
  createdAt: string;
  updatedAt: string;
}

/**
 * Observation Ledger: Grounded Observation Record (learning_evidence.json)
 * Append-oriented log of specific visual sightings referencing concrete tokens.
 */
export interface LearningEvidenceRecord {
  evidenceId: string; // Unique observation ID e.g. "evi_1793500200_abc12"
  recipeId: string; // Foreign key to AbstractObjectRecipe
  label: string;
  category: VisualTraitCategory;
  observedTokenId: string; // Grounded token ID instance
  sourceArchiveHash?: string; // CRC-32 or hash of active tokens.zip when observed
  boundingBox: [number, number, number, number]; // [normX, normY, normW, normH]
  observedCentroids: number[]; // Centroids found within the bounding box
  latitude: number; // Default: -33.8688
  longitude: number; // Default: 151.2093
  utcTimestamp: string; // ISO 8601 UTC timestamp
  observedAt: string; // Preserved observation timestamp
  source: LearningSource; // "human_verified" | "human_suggested" | "model_cluster"
  confidence: number; // 1.0 (verified), 0.3 (suggestion)
}

/**
 * Archive Generation State
 * Tracks the transactional generation lifecycle of the active tokens.zip.
 */
export interface ArchiveGenerationState {
  generation: number;
  activeZipPath: string;
  totalTokens: number;
  archiveSizeBytes: number;
  archiveCrc32Hex: string;
  swappedAt: string;
}

export interface ArchiveValidationResult {
  valid: boolean;
  tokenCount: number;
  byteSize: number;
  crc32Hex: string;
  error?: string;
  sampleTokens?: string[];
}

