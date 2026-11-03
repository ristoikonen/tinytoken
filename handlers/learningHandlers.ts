/**
 * HTTP API Handlers for Learning & Transactional Archive Operations (handlers/learningHandlers.ts)
 *
 * Implements:
 * - Second archive-ingestion path accepting pre-built ZIP archives
 * - Generation-based transactional archive hot-swap with strict validation and rollback
 * - Zero-downtime archive swapping while keeping persistent knowledge layer intact
 * - Query endpoints for versioned object recipes and append-only evidence ledger
 */

import type { LearningSource } from "../models/learning";
import type { ArchiveManager } from "../utils/archiveManager";
import { errorResponse, jsonResponse, verifySameOriginMutation } from "../utils/response";
import type { VisualTraitCategory, VisualTraitRegistry } from "../utils/visualTraitRegistry";

export class LearningHandlers {
  private swapLockInProgress = false;

  constructor(
    private archiveManager: ArchiveManager,
    private visualRegistry: VisualTraitRegistry,
  ) {}

  /**
   * POST /api/archive/swap
   * Second archive-ingestion path: Accepts a pre-built tokens.zip archive directly,
   * validates it strictly, and transactionally swaps it into active generation.
   */
  public async handleArchiveSwap(req: Request): Promise<Response> {
    const forbidden = verifySameOriginMutation(req, [
      "application/zip",
      "application/octet-stream",
      "multipart/form-data",
    ]);
    if (forbidden) return forbidden;

    if (this.swapLockInProgress) {
      return errorResponse("Another archive swap is currently in progress", 409);
    }

    this.swapLockInProgress = true;
    try {
      const contentType = req.headers.get("content-type") || "";
      let archiveBytes: Uint8Array | null = null;

      if (contentType.includes("multipart/form-data")) {
        try {
          const formData = await req.formData();
          const file = formData.get("archive") || formData.get("file");
          if (file && typeof (file as any).arrayBuffer === "function") {
            archiveBytes = new Uint8Array(await (file as Blob).arrayBuffer());
          }
        } catch (err: any) {
          return errorResponse(`Could not parse multipart form: ${err?.message || String(err)}`, 400);
        }
      } else {
        try {
          archiveBytes = new Uint8Array(await req.arrayBuffer());
        } catch (err: any) {
          return errorResponse(`Could not read request body: ${err?.message || String(err)}`, 400);
        }
      }

      if (!archiveBytes || archiveBytes.length < 22) {
        return errorResponse("Empty or invalid archive payload. Provide a valid ZIP container.", 400);
      }

      const swapResult = await this.archiveManager.swapArchive(archiveBytes);
      if (!swapResult.success) {
        return jsonResponse(
          {
            success: false,
            error: swapResult.error || "Archive validation failed. Active archive remains untouched.",
            activeGeneration: swapResult.generation,
            activeTokens: swapResult.totalTokens,
            archiveSizeBytes: swapResult.archiveSizeBytes,
          },
          400,
        );
      }

      const recipes = await this.visualRegistry.getRecipes();
      const evidence = await this.visualRegistry.getEvidence();

      return jsonResponse({
        success: true,
        message: `Swapped to Archive Generation ${swapResult.generation} (${swapResult.totalTokens} tokens). Learnings preserved.`,
        generation: swapResult.generation,
        totalTokens: swapResult.totalTokens,
        archiveSizeBytes: swapResult.archiveSizeBytes,
        archiveCrc32Hex: swapResult.crc32Hex,
        learningsPreserved: true,
        activeRecipesCount: recipes.length,
        activeEvidenceCount: evidence.length,
      });
    } catch (err: any) {
      return errorResponse(`Archive swap failed: ${err?.message || String(err)}`, 500);
    } finally {
      this.swapLockInProgress = false;
    }
  }

  /**
   * GET /api/archive/status
   * Reports active archive generation and knowledge layer statistics.
   */
  public async handleArchiveStatus(): Promise<Response> {
    const gen = this.archiveManager.getGenerationState();
    const recipes = await this.visualRegistry.getRecipes();
    const evidence = await this.visualRegistry.getEvidence();
    const traits = this.visualRegistry.getAll();

    return jsonResponse({
      success: true,
      archive: gen,
      knowledge: {
        totalCentroidTraits: Object.keys(traits).length,
        totalObjectRecipes: recipes.length,
        totalEvidenceObservations: evidence.length,
      },
    });
  }

  /**
   * GET /api/learning/recipes
   * Returns all versioned, token-independent object recipes.
   */
  public async handleGetRecipes(): Promise<Response> {
    const recipes = await this.visualRegistry.getRecipes();
    return jsonResponse({
      success: true,
      count: recipes.length,
      recipes,
    });
  }

  /**
   * GET /api/learning/evidence
   * Returns observation records from the append-oriented evidence ledger.
   */
  public async handleGetEvidence(): Promise<Response> {
    const evidence = await this.visualRegistry.getEvidence();
    return jsonResponse({
      success: true,
      count: evidence.length,
      evidence,
    });
  }

  /**
   * POST /api/learning/evidence
   * Appends an observation record directly to the evidence ledger.
   */
  public async handleRecordEvidence(req: Request): Promise<Response> {
    const forbidden = verifySameOriginMutation(req);
    if (forbidden) return forbidden;

    try {
      const body = (await req.json()) as {
        tokenId?: string;
        label?: string;
        category?: VisualTraitCategory;
        boundingBox?: [number, number, number, number];
        centroidIds?: number[];
        latitude?: number;
        longitude?: number;
        source?: LearningSource;
        confidence?: number;
      };

      if (!body.tokenId || !body.label) {
        return errorResponse("Both 'tokenId' and 'label' are required", 400);
      }

      const res = await this.visualRegistry.teachObject(
        body.tokenId,
        body.label,
        body.category || "Silhouette",
        body.boundingBox || [0, 0, 1, 1],
        body.centroidIds || [],
        {
          latitude: body.latitude,
          longitude: body.longitude,
          source: body.source,
          confidence: body.confidence,
        },
      );

      return jsonResponse({
        success: true,
        evidence: res.evidence,
        recipe: res.recipe,
      });
    } catch (err: any) {
      return errorResponse(`Failed to record evidence: ${err?.message || String(err)}`, 500);
    }
  }
}
