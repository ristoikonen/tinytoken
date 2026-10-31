import { serve, type BunRequest } from "bun";
import { TokenApiHandlers } from "./handlers/tokenHandlers";
import { Win11TokenPipeline } from "./utils/indexer";
import { errorResponse } from "./utils/response";
import { VqInvertedIndexEngine } from "./utils/vqInvertedIndex";
import { CompressedTokenSearchEngine } from "./utils/zipEngine";
import { corsResponse } from "./utils/response";

const IMAGE_DIR = "./images";
const DOCUMENT_DIR = "./documents";
const ZIP_ARCHIVE_PATH = "./data/tokens.zip";

const pipeline = new Win11TokenPipeline({
  sourceDirectory: IMAGE_DIR,
  documentDirectory: DOCUMENT_DIR,
  outputZipPath: ZIP_ARCHIVE_PATH,
  allowedExtensions: [".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".svg", ".pdf", ".docx"],
  maxTokenSizeKb: 25,
});

const searchEngine = new CompressedTokenSearchEngine(ZIP_ARCHIVE_PATH);
const invertedIndex = new VqInvertedIndexEngine(":memory:");
const apiHandlers = new TokenApiHandlers(
  searchEngine,
  pipeline,
  IMAGE_DIR,
  invertedIndex,
  DOCUMENT_DIR,
);

const port = Number(Bun.env.APP_PORT ?? 3000);
const host = Bun.env.APP_HOST ?? "localhost";

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
    "/api/stats": () => apiHandlers.handleStats(),
    "/api/codebook": () => apiHandlers.handleCodebook(),
    "/api/search": (req: BunRequest) => apiHandlers.handleSearch(new URL(req.url)),
    "/api/search/pattern": (req: BunRequest) => apiHandlers.handlePatternSearch(new URL(req.url)),
    "/api/index/rebuild": {
      POST: (req: BunRequest) => apiHandlers.handleRebuildIndex(req),
      OPTIONS: (req: BunRequest) => corsResponse(req),
    },
  },

  async fetch(req: BunRequest) {
    const url = new URL(req.url);
    const path = url.pathname;

    if (req.method === "OPTIONS") {
      return corsResponse(req);
    }

    // Dynamic token & image routes
    try {
      if (path.startsWith("/api/token/")) {
        const rest = path.slice("/api/token/".length);
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

console.log(`🚀 TinyToken Server running at http://${server.hostname}:${server.port}`);

// Initialize ZIP index and native bun:sqlite VQ Inverted Index asynchronously
(async function initIndex() {
  const t0 = performance.now();
  const zipFile = Bun.file(ZIP_ARCHIVE_PATH);
  if (!(await zipFile.exists())) {
    console.log(
      `📦 Building TinyToken 25 KB GF(2^8) + VQ + SHARK archive from ${IMAGE_DIR} and ${DOCUMENT_DIR}...`,
    );
    await pipeline.buildTokenArchive();
  }
  await searchEngine.buildIndex();
  await invertedIndex.syncTokens(searchEngine.getAllEntries());
  const totalMs = (performance.now() - t0).toFixed(2);
  console.log(
    `⚡ Indexed ${searchEngine.getCount()} tokens into RAM + bun:sqlite Inverted Index in ${totalMs} ms (${(searchEngine.archiveSizeBytes / 1024).toFixed(1)} KB archive)`,
  );
})().catch((err) => {
  console.error("Failed to initialize token index:", err);
});

export default server;
