import { serve, type BunRequest } from "bun";
import { LearningHandlers } from "./handlers/learningHandlers";
import { TokenApiHandlers } from "./handlers/tokenHandlers";
import { ArchiveManager } from "./utils/archiveManager";
import type { Win11TokenPipeline } from "./utils/indexer";
import { errorResponse, jsonResponse, verifySameOriginMutation, corsResponse } from "./utils/response";
import {
  fetchAndIndexGooglePhotos,
  getGooglePhotosAuthUrl,
  GOOGLE_PHOTOS_READ_SCOPES,
} from "./utils/googlePhotos";
import { visualTraitRegistry } from "./utils/visualTraitRegistry";
import { VqInvertedIndexEngine } from "./utils/vqInvertedIndex";
import { CompressedTokenSearchEngine } from "./utils/zipEngine";
import helpPage from "./pages/help.html" with { type: "text" };
import helpCardsPage from "./pages/help-cards.html" with { type: "text" };

// Configuration
const isVercel = Bun.env.APP_VERCEL;
const DATA_DIR = Bun.env.DATA_DIR ?? "/tmp/data";
const ZIP_ARCHIVE_PATH = Bun.env.ZIP_ARCHIVE_PATH ?? `${DATA_DIR}/tokens.zip`;
const IMAGE_DIR =
  Bun.env.IMAGE_DIR ??
  Bun.argv[2] ??
  "C:/Users/risto/OneDrive/Pictures/Nikon/toke";
const DOCUMENT_DIR = "./documents";
const INDEXED_DIR_STAMP = `${DATA_DIR}/.indexed_dir`;
const useZip = Bun.env.APP_ZIP_USE !== "false";
const isTest = Bun.env.NODE_ENV === "test";
const rawPort = Bun.env.PORT ?? Bun.env.APP_PORT ?? process.env.PORT;
const port = isTest
  ? 0
  : isVercel
    ? (rawPort ? Number(rawPort) : undefined)
    : Number(rawPort ?? 3000);
const host = isVercel ? (Bun.env.APP_HOST ?? "0.0.0.0") : (Bun.env.APP_HOST ?? "localhost");

// Cached HTML template
let cachedHtml: string | null = null;
async function getIndexHtml(): Promise<string> {
  if (!cachedHtml) {
    const f1 = Bun.file("./public/index.html");
    if (await f1.exists()) {
      cachedHtml = await f1.text();
    } else {
      const f2 = Bun.file("./index.html");
      cachedHtml = (await f2.exists()) ? await f2.text() : "<!DOCTYPE html><html><body>TinyToken</body></html>";
    }
  }
  return cachedHtml;
}

// Lazy indexer pipeline loader
let pipelineInstance: Win11TokenPipeline | null = null;
async function getPipeline(): Promise<Win11TokenPipeline> {
  if (!pipelineInstance) {
    const { Win11TokenPipeline } = await import("./utils/indexer");
    pipelineInstance = new Win11TokenPipeline({
      sourceDirectory: IMAGE_DIR,
      documentDirectory: DOCUMENT_DIR,
      outputZipPath: ZIP_ARCHIVE_PATH,
      dataDirectory: DATA_DIR,
      useZip,
      allowedExtensions: [".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".svg", ".pdf", ".docx"],
      maxTokenSizeKb: 25,
    });
  }
  return pipelineInstance;
}

const pipeline: Win11TokenPipeline = {
  buildTokenArchive: async (dir?: string) => {
    const p = await getPipeline();
    return p.buildTokenArchive(dir);
  },
} as unknown as Win11TokenPipeline;

// Core Engines
const searchEngine = new CompressedTokenSearchEngine(ZIP_ARCHIVE_PATH, {
  useZip,
  dataDirPath: DATA_DIR,
});
const invertedIndex = new VqInvertedIndexEngine(":memory:");
const archiveManager = new ArchiveManager(searchEngine, ZIP_ARCHIVE_PATH, invertedIndex);
const learningHandlers = new LearningHandlers(archiveManager, visualTraitRegistry);
const apiHandlers = new TokenApiHandlers(
  searchEngine,
  pipeline,
  IMAGE_DIR,
  invertedIndex,
  DOCUMENT_DIR,
);

// Initialization lifecycle
let initPromise: Promise<void> | null = null;
export function ensureInitialized(): Promise<void> {
  if (!initPromise) {
    initPromise = initIndex();
  }
  return initPromise;
}

async function directoryExists(dir: string): Promise<boolean> {
  try {
    const glob = new Bun.Glob("*");
    for await (const _ of glob.scan({ cwd: dir })) break;
    return true;
  } catch {
    return false;
  }
}

async function ensureDataDirectory(targetDir: string, zipPath: string): Promise<void> {
  if (targetDir === "./data") return;
  const filesToSeed = [
    { src: "./data/tokens.zip", dst: zipPath },
    { src: "./data/manual_token_registry.json", dst: `${targetDir}/manual_token_registry.json` },
    { src: "./data/object_registry.json", dst: `${targetDir}/object_registry.json` },
    { src: "./data/learning_evidence.json", dst: `${targetDir}/learning_evidence.json` },
    { src: "./data/attestation_ed25519.json", dst: `${targetDir}/attestation_ed25519.json` },
  ];

  for (const { src, dst } of filesToSeed) {
    try {
      const dstFile = Bun.file(dst);
      if (!(await dstFile.exists()) || dstFile.size === 0) {
        const srcFile = Bun.file(src);
        if (await srcFile.exists()) {
          const content = await srcFile.arrayBuffer();
          await Bun.write(dst, content);
        }
      }
    } catch (err) {
      console.warn(`Could not seed ${dst} from ${src}:`, err);
    }
  }
}

async function initIndex(): Promise<void> {
  await ensureDataDirectory(DATA_DIR, ZIP_ARCHIVE_PATH);
  const t0 = performance.now();
  const isZip = searchEngine.isZipMode();
  const zipFile = Bun.file(ZIP_ARCHIVE_PATH);
  const stampFile = Bun.file(INDEXED_DIR_STAMP);
  const lastIndexedDir = (await stampFile.exists()) ? (await stampFile.text()).trim() : null;
  const dirChanged = lastIndexedDir !== null && lastIndexedDir !== IMAGE_DIR;
  const imageDirValid = await directoryExists(IMAGE_DIR);

  let tokensExist = isZip ? await zipFile.exists() : false;
  if (!isZip) {
    try {
      const glob = new Bun.Glob("*.token");
      for await (const _ of glob.scan({ cwd: DATA_DIR })) {
        tokensExist = true;
        break;
      }
    } catch {
      tokensExist = false;
    }
  }

  const shouldRebuild = (!tokensExist && imageDirValid) || (dirChanged && imageDirValid);

  if (shouldRebuild) {
    console.log(`Building tokens from ${IMAGE_DIR} and ${DOCUMENT_DIR} (ZIP=${isZip})...`);
    await (await getPipeline()).buildTokenArchive(IMAGE_DIR);
    try {
      await Bun.write(INDEXED_DIR_STAMP, IMAGE_DIR);
    } catch { }
  } else if (imageDirValid && !(await stampFile.exists())) {
    try {
      await Bun.write(INDEXED_DIR_STAMP, IMAGE_DIR);
    } catch { }
  }

  try {
    await searchEngine.buildIndex();
    await invertedIndex.syncTokens(searchEngine.getAllEntries());
    const totalMs = (performance.now() - t0).toFixed(2);
    const storageLabel = isZip ? "archive" : `*.token files in ${DATA_DIR}`;
    console.log(
      `Indexed ${searchEngine.getCount()} tokens into RAM in ${totalMs} ms (${(searchEngine.archiveSizeBytes / 1024).toFixed(1)} KB, ${storageLabel})`,
    );
  } catch (err) {
    console.error("Warning: Could not build index/sync tokens:", err);
  }
}

// Unified API Route Dispatcher
async function handleApiRoute(path: string, req: BunRequest, url: URL): Promise<Response> {
  if (path === "/api/stats") return apiHandlers.handleStats();
  if (path === "/api/codebook") return apiHandlers.handleCodebook();
  if (path === "/api/registry") return apiHandlers.handleGetRegistry();
  if (path === "/api/objects") return apiHandlers.handleGetObjectAnnotations();
  if (path === "/api/search") return apiHandlers.handleSearch(url);
  if (path === "/api/search/pattern") return apiHandlers.handlePatternSearch(url);
  if (path === "/api/index/rebuild" && req.method === "POST") return apiHandlers.handleRebuildIndex(req);
  if (path === "/api/archive/swap" && req.method === "POST") return learningHandlers.handleArchiveSwap(req);
  if (path === "/api/archive/status") return learningHandlers.handleArchiveStatus();
  if (path === "/api/learning/recipes") return learningHandlers.handleGetRecipes();
  if (path === "/api/learning/evidence") {
    if (req.method === "POST") return learningHandlers.handleRecordEvidence(req);
    return learningHandlers.handleGetEvidence();
  }
  if (path === "/api/photos/auth-url" && req.method === "GET") {
    return jsonResponse({
      success: true,
      authUrl: getGooglePhotosAuthUrl(),
      scope: GOOGLE_PHOTOS_READ_SCOPES[0],
    });
  }
  if (path === "/api/photos/index-batch" && req.method === "POST") {
    const forbidden = verifySameOriginMutation(req);
    if (forbidden) return forbidden;
    let token: string | undefined;
    let code: string | undefined;
    try {
      const body = (await req.json()) as { token?: string; code?: string };
      token = body?.token;
      code = body?.code;
    } catch { }
    const res = await fetchAndIndexGooglePhotos(token, code, {
      pipeline,
      searchEngine,
      invertedIndex,
      archiveManager,
      zipArchivePath: ZIP_ARCHIVE_PATH,
    });
    return jsonResponse(res);
  }

  // Dynamic parameterized routes
  if (path.startsWith("/api/token/")) {
    const rest = path.slice("/api/token/".length);
    if (rest.endsWith("/teach") && req.method === "POST") {
      return apiHandlers.handleTeachObject(decodeURIComponent(rest.slice(0, -"/teach".length)), req);
    }
    if (rest.endsWith("/preview") && req.method === "GET") {
      return apiHandlers.handleGetTokenPreview(decodeURIComponent(rest.slice(0, -"/preview".length)));
    }
    if (rest.endsWith("/mutate") && req.method === "POST") {
      return apiHandlers.handleHomomorphicMutate(decodeURIComponent(rest.slice(0, -"/mutate".length)), req);
    }
    if (req.method === "GET" && rest.length > 0) {
      return apiHandlers.handleGetToken(decodeURIComponent(rest));
    }
  }

  if (path.startsWith("/api/image/") && req.method === "GET") {
    return apiHandlers.handleGetSourceImage(decodeURIComponent(path.slice("/api/image/".length)));
  }

  if (path.startsWith("/api/document/") && req.method === "GET") {
    return apiHandlers.handleGetSourceDocument(decodeURIComponent(path.slice("/api/document/".length)));
  }

  return errorResponse("Not Found", 404, false);
}

// Unified Request Handler
export async function fetchHandler(req: Request): Promise<Response> {
  if (req.method === "OPTIONS") return corsResponse(req);
  const url = new URL(req.url);
  try {
    await ensureInitialized();
    if (url.pathname === "/" || url.pathname === "/index.html") {
      return new Response(await getIndexHtml(), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      });
    }
    return await handleApiRoute(url.pathname, req as BunRequest, url);
  } catch (err) {
    if (err instanceof URIError) {
      return errorResponse("Invalid URI encoding", 400);
    }
    throw err;
  }
}

// HTTP Server
const server = serve({
  hostname: host,
  ...(port !== undefined ? { port } : {}),
  routes: {
    "/": async () =>
      new Response(await getIndexHtml(), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      }),
    "/index.html": async () =>
      new Response(await getIndexHtml(), {
        headers: { "Content-Type": "text/html; charset=utf-8" },
      }),

    "/help": {
      GET: () => new Response(String(helpPage), { headers: { "Content-Type": "text/html" } })
    },
    "/help-cards": {
      GET: () => new Response(String(helpCardsPage), { headers: { "Content-Type": "text/html" } })
    },
    "/guide": {
      GET: () => new Response(String(helpCardsPage), { headers: { "Content-Type": "text/html" } })
    },


  },
  fetch(req: BunRequest) {
    return fetchHandler(req);
  },
});

console.log(`TinyToken Server running at http://${server.hostname}:${server.port}`);

// Trigger initial index build immediately
ensureInitialized().catch((err) => {
  console.error("Failed to initialize token index:", err);
});

export default server;
