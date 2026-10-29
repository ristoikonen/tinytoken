// @ts-ignore Bun provides this module at runtime; its types may not be installed in the editor.
import { serve, type BunRequest } from "bun";
import { TokenApiHandlers } from "./handlers/tokenHandlers";
import { Win11TokenPipeline } from "./utils/indexer";
import { errorResponse } from "./utils/response";
import { CompressedTokenSearchEngine } from "./utils/zipEngine";

const IMAGE_DIR = "./images";
const ZIP_ARCHIVE_PATH = "./data/tokens.zip";

const pipeline = new Win11TokenPipeline({
  sourceDirectory: IMAGE_DIR,
  outputZipPath: ZIP_ARCHIVE_PATH,
  allowedExtensions: [".jpg", ".jpeg", ".png", ".webp", ".avif", ".bmp", ".svg", ".pdf", ".docx"],
  maxTokenSizeKb: 25,
});

const searchEngine = new CompressedTokenSearchEngine(ZIP_ARCHIVE_PATH);
const apiHandlers = new TokenApiHandlers(searchEngine, pipeline, IMAGE_DIR);

(async function main() {
  // Ensure initial token archive exists and index its Central Directory into RAM on boot
  const zipFile = Bun.file(ZIP_ARCHIVE_PATH);
  if (!(await zipFile.exists())) {
    console.log("📦 Building initial TinyToken 25 KB GF(2^8) + VQ + SHARK archive...");
    await pipeline.buildTokenArchive(24);
  }
  await searchEngine.buildIndex();
  console.log(
    `⚡ Indexed ${searchEngine.getCount()} tokens into RAM in ${searchEngine.lastBootIndexMs} ms (${(searchEngine.archiveSizeBytes / 1024).toFixed(1)} KB archive)`,
  );

  const port = Number(Bun.env.APP_PORT ?? 3000);
  const host = Bun.env.APP_HOST ?? "localhost";

  serve({
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
      "/api/index/rebuild": {
        POST: (req: BunRequest) => apiHandlers.handleRebuildIndex(req),
      },
    },

    async fetch(req: BunRequest) {
      const url = new URL(req.url);
      const path = url.pathname;

      if (req.method === "OPTIONS") {
        return new Response(null, {
          status: 204,
          headers: {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Session-ID, X-Correlation-ID",
          },
        });
      }

      // Dynamic token & image routes
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

      return errorResponse("Not Found", 404, false);
    },
  });

  console.log(`🚀 TinyToken Server running at http://${host}:${port}`);
})().catch((err) => {
  console.error(err);
  process.exitCode = 1;
});
