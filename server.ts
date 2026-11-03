import { serve, type BunRequest } from "bun";
import { LearningHandlers } from "./handlers/learningHandlers";
import { TokenApiHandlers } from "./handlers/tokenHandlers";
import { ArchiveManager } from "./utils/archiveManager";
import { Win11TokenPipeline } from "./utils/indexer";
import { errorResponse, jsonResponse, verifySameOriginMutation } from "./utils/response";
import {
  fetchAndIndexGooglePhotos,
  getGooglePhotosAuthUrl,
  GOOGLE_PHOTOS_READ_SCOPES,
} from "./utils/googlePhotos";
import { visualTraitRegistry } from "./utils/visualTraitRegistry";
import { VqInvertedIndexEngine } from "./utils/vqInvertedIndex";
import { CompressedTokenSearchEngine } from "./utils/zipEngine";
import { corsResponse } from "./utils/response";

const isVercel = Boolean(Bun.env.VERCEL || Bun.env.NOW_REGION);
const IMAGE_DIR =
  Bun.env.IMAGE_DIR ??
  Bun.argv[2] ??
  (isVercel ? "./images" : "C:/Users/risto/OneDrive/Pictures/Nikon/toke");
function getZipArchivePath(): string {
  const candidates = [
    `${import.meta.dir}/data/tokens.zip`,
    `${import.meta.dir}/../data/tokens.zip`,
    `${process.cwd()}/data/tokens.zip`,
    `./data/tokens.zip`,
  ];
  for (const c of candidates) {
    if (Bun.file(c).size > 0) return c;
  }
  return "./data/tokens.zip";
}

const DOCUMENT_DIR = "./documents";
const DATA_DIR = "./data";
const ZIP_ARCHIVE_PATH = getZipArchivePath();
const INDEXED_DIR_STAMP = "./data/.indexed_dir";
const useZip = Bun.env.APP_ZIP_USE !== "false";

const pipeline = new Win11TokenPipeline({
  sourceDirectory: IMAGE_DIR,
  documentDirectory: DOCUMENT_DIR,
  outputZipPath: ZIP_ARCHIVE_PATH,
  dataDirectory: DATA_DIR,
  useZip,
  allowedExtensions: [".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".svg", ".pdf", ".docx"],
  maxTokenSizeKb: 25,
});

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

const port = Number(Bun.env.PORT ?? Bun.env.APP_PORT ?? 3000);
const host = Bun.env.APP_HOST ?? (isVercel ? "0.0.0.0" : "localhost");

// Track index initialization so serverless cold starts never encounter uninitialized state
let initPromise: Promise<void> | null = null;
export function ensureInitialized(): Promise<void> {
  if (!initPromise) {
    initPromise = initIndex();
  }
  return initPromise;
}

// Start HTTP server immediately so `bun --hot server.ts` never hangs or misses port binding
const server = serve({
  hostname: host,
  port,
  routes: {
    "/": () => {
      const file = Bun.file("./public/index.html");
      return file.exists().then((exists) =>
        exists
          ? new Response(file, { headers: { "Content-Type": "text/html; charset=utf-8" } })
          : errorResponse("Index.html not found", 404),
      );
    },
    "/index.html": () => {
      const file = Bun.file("./public/index.html");
      return file.exists().then((exists) =>
        exists
          ? new Response(file, { headers: { "Content-Type": "text/html; charset=utf-8" } })
          : errorResponse("Index.html not found", 404),
      );
    },
    "/api/stats": async () => {
      await ensureInitialized();
      return apiHandlers.handleStats();
    },
    "/api/codebook": async () => {
      await ensureInitialized();
      return apiHandlers.handleCodebook();
    },
    "/api/registry": async () => {
      await ensureInitialized();
      return apiHandlers.handleGetRegistry();
    },
    "/api/objects": async () => {
      await ensureInitialized();
      return apiHandlers.handleGetObjectAnnotations();
    },
    "/api/search": async (req: BunRequest) => {
      await ensureInitialized();
      return apiHandlers.handleSearch(new URL(req.url));
    },
    "/api/search/pattern": async (req: BunRequest) => {
      await ensureInitialized();
      return apiHandlers.handlePatternSearch(new URL(req.url));
    },
    "/api/index/rebuild": {
      POST: async (req: BunRequest) => {
        await ensureInitialized();
        return apiHandlers.handleRebuildIndex(req);
      },
      OPTIONS: (req: BunRequest) => corsResponse(req),
    },
    "/api/archive/swap": {
      POST: async (req: BunRequest) => {
        await ensureInitialized();
        return learningHandlers.handleArchiveSwap(req);
      },
      OPTIONS: (req: BunRequest) => corsResponse(req),
    },
    "/api/archive/status": async () => {
      await ensureInitialized();
      return learningHandlers.handleArchiveStatus();
    },
    "/api/learning/recipes": async () => {
      await ensureInitialized();
      return learningHandlers.handleGetRecipes();
    },
    "/api/learning/evidence": {
      GET: async () => {
        await ensureInitialized();
        return learningHandlers.handleGetEvidence();
      },
      POST: async (req: BunRequest) => {
        await ensureInitialized();
        return learningHandlers.handleRecordEvidence(req);
      },
      OPTIONS: (req: BunRequest) => corsResponse(req),
    },
    "/api/photos/auth-url": {
      GET: () => {
        return jsonResponse({
          success: true,
          authUrl: getGooglePhotosAuthUrl(),
          scope: GOOGLE_PHOTOS_READ_SCOPES[0],
        });
      },
    },
    "/api/photos/index-batch": {
      POST: async (req: BunRequest) => {
        console.log('ix batch');
        await ensureInitialized();
        const forbidden = verifySameOriginMutation(req);
        if (forbidden) return forbidden;
        let token: string | undefined;
        let code: string | undefined;
        try {
          const body = (await req.json()) as { token?: string; code?: string };
          token = body?.token;
          code = body?.code;
        } catch {
          // empty body ok
        }
        console.log('fetchAndIndexGooglePhotos');
        const res = await fetchAndIndexGooglePhotos(token, code, {
          pipeline,
          searchEngine,
          invertedIndex,
          archiveManager,
          zipArchivePath: ZIP_ARCHIVE_PATH,
        });
        console.log('done fetchAndIndexGooglePhotos');
        return jsonResponse(res);
      },
      OPTIONS: (req: BunRequest) => corsResponse(req),
    },
  },

  async fetch(req: BunRequest) {
    const url = new URL(req.url);
    let path = url.pathname;

    // Handle Vercel serverless rewrite parameters & forwarding headers
    const vercelPath = url.searchParams.get("__path");
    if (vercelPath) {
      path = vercelPath.startsWith("/") ? vercelPath : `/${vercelPath}`;
      url.searchParams.delete("__path");
    } else if (
      path === "/api/index" ||
      path === "/api/index.ts" ||
      path === "/api" ||
      path === "/api/"
    ) {
      const forwarded =
        req.headers.get("x-forwarded-uri") ||
        req.headers.get("x-matched-path") ||
        req.headers.get("x-invoke-path");
      if (forwarded && forwarded.startsWith("/api/")) {
        path = forwarded.split("?")[0]!;
      }
    }

    if (req.method === "OPTIONS") {
      return corsResponse(req);
    }

    // Route dispatch (handles both direct fetch invocation and fallback)
    try {
      await ensureInitialized();

      if (path === "/" || path === "/index.html") {
        const file = Bun.file("./public/index.html");
        return (await file.exists())
          ? new Response(file, { headers: { "Content-Type": "text/html; charset=utf-8" } })
          : errorResponse("Index.html not found", 404);
      }
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
        } catch {
          // empty body ok
        }
        const res = await fetchAndIndexGooglePhotos(token, code, {
          pipeline,
          searchEngine,
          invertedIndex,
          archiveManager,
          zipArchivePath: ZIP_ARCHIVE_PATH,
        });
        console.log(res);
        return jsonResponse(res);
      }

      if (path.startsWith("/api/token/")) {
        const rest = path.slice("/api/token/".length);
        if (rest.endsWith("/teach") && req.method === "POST") {
          const id = decodeURIComponent(rest.slice(0, -"/teach".length));
          return apiHandlers.handleTeachObject(id, req);
        }
        if (rest.endsWith("/preview") && req.method === "GET") {
          const id = decodeURIComponent(rest.slice(0, -"/preview".length));
          return apiHandlers.handleGetTokenPreview(id);
        }
        if (rest.endsWith("/mutate") && req.method === "POST") {
          const id = decodeURIComponent(rest.slice(0, -"/mutate".length));
          return apiHandlers.handleHomomorphicMutate(id, req);
        }
        if (req.method === "GET" && rest.length > 0) {
          const id = decodeURIComponent(rest);
          return apiHandlers.handleGetToken(id);
        }
      }

      if (path.startsWith("/api/image/") && req.method === "GET") {
        const name = decodeURIComponent(path.slice("/api/image/".length));
        return apiHandlers.handleGetSourceImage(name);
      }

      if (path.startsWith("/api/document/") && req.method === "GET") {
        const name = decodeURIComponent(path.slice("/api/document/".length));
        return apiHandlers.handleGetSourceDocument(name);
      }
    } catch (err) {
      if (err instanceof URIError) {
        return errorResponse("Invalid URI encoding", 400);
      }
      throw err;
    }

    return errorResponse("Not Found", 404, false);
  },
});

console.log(`TinyToken Server running at http://${server.hostname}:${server.port}`);

async function directoryExists(dir: string): Promise<boolean> {
  try {
    const glob = new Bun.Glob("*");
    for await (const _ of glob.scan({ cwd: dir })) break;
    return true;
  } catch {
    return false;
  }
}

// Initialize token index and native bun:sqlite VQ Inverted Index asynchronously
async function initIndex() {
  const t0 = performance.now();
  const isZip = searchEngine.isZipMode();
  const zipFile = Bun.file(ZIP_ARCHIVE_PATH);
  const stampFile = Bun.file(INDEXED_DIR_STAMP);
  const lastIndexedDir = (await stampFile.exists()) ? (await stampFile.text()).trim() : null;
  const dirChanged = lastIndexedDir !== null && lastIndexedDir !== IMAGE_DIR;
  const imageDirValid = await directoryExists(IMAGE_DIR);

  let tokensExist = false;
  if (isZip) {
    tokensExist = await zipFile.exists();
  } else {
    // Check if any *.token files exist in data dir
    const glob = new Bun.Glob("*.token");
    for await (const _ of glob.scan({ cwd: DATA_DIR })) {
      tokensExist = true;
      break;
    }
    // If running with APP_ZIP_USE=false and no *.token files in data dir yet,
    // unpack existing tokens from tokens.zip so they are immediately available
    if (!tokensExist && (await zipFile.exists())) {
      console.log(`[APP_ZIP_USE=false] Extracting existing tokens from ${ZIP_ARCHIVE_PATH} to ${DATA_DIR}...`);
      const tempEngine = new CompressedTokenSearchEngine(ZIP_ARCHIVE_PATH, { useZip: true });
      await tempEngine.buildIndex();
      for (const entry of tempEngine.getAllEntries()) {
        const tokenBytes = await tempEngine.extractToken(entry.id);
        if (tokenBytes) {
          await Bun.write(`${DATA_DIR}/${entry.entryName}`, tokenBytes);
        }
      }
      tokensExist = tempEngine.getCount() > 0;
    }
  }

  // Rebuild only if running locally with a valid source directory
  const shouldRebuild = !isVercel && (
    (!tokensExist && imageDirValid) ||
    (dirChanged && imageDirValid)
  );

  if (shouldRebuild) {
    if (dirChanged) {
      console.log(
        `Image directory changed (${lastIndexedDir} -> ${IMAGE_DIR}), rebuilding tokens...`,
      );
    } else {
      console.log(
        `Building tokens from ${IMAGE_DIR} and ${DOCUMENT_DIR} (ZIP=${isZip})...`,
      );
    }
    await pipeline.buildTokenArchive(IMAGE_DIR);
    try {
      await Bun.write(INDEXED_DIR_STAMP, IMAGE_DIR);
    } catch { }
  } else if (!isVercel && imageDirValid && !(await stampFile.exists())) {
    try {
      await Bun.write(INDEXED_DIR_STAMP, IMAGE_DIR);
    } catch { }
  }

  await searchEngine.buildIndex();
  await invertedIndex.syncTokens(searchEngine.getAllEntries());
  const totalMs = (performance.now() - t0).toFixed(2);
  const storageLabel = isZip ? "archive" : "*.token files in ./data";
  console.log(
    `Indexed ${searchEngine.getCount()} tokens into RAM + bun:sqlite Inverted Index in ${totalMs} ms (${(searchEngine.archiveSizeBytes / 1024).toFixed(1)} KB, ${storageLabel})`,
  );
}

// Trigger initial index build immediately
ensureInitialized().catch((err) => {
  console.error("Failed to initialize token index:", err);
});

export default server;
