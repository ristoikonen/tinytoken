import server, { ensureInitialized } from "../server";

export default async function handler(req: any, res?: any): Promise<Response | void> {
  try {
    await ensureInitialized();

    // 1. Web Standard Request (Bun runtime / Vercel Edge / Bun.serve)
    if (
      req instanceof Request ||
      (req && typeof req.text === "function" && typeof req.arrayBuffer === "function")
    ) {
      return await server.fetch(req);
    }

    // 2. Node.js runtime (req: IncomingMessage, res: ServerResponse)
    if (res && typeof res.setHeader === "function" && typeof res.end === "function") {
      const protocol = req.headers?.["x-forwarded-proto"] || "https";
      const host = req.headers?.["x-forwarded-host"] || req.headers?.["host"] || "localhost";
      const url = `${protocol}://${host}${req.url}`;

      const headers = new Headers();
      if (req.headers) {
        for (const [k, v] of Object.entries(req.headers)) {
          if (v !== undefined) {
            headers.set(k, Array.isArray(v) ? v.join(", ") : String(v));
          }
        }
      }

      let body: any = null;
      if (req.method !== "GET" && req.method !== "HEAD") {
        const chunks: Buffer[] = [];
        for await (const chunk of req) chunks.push(chunk);
        if (chunks.length > 0) {
          body = Buffer.concat(chunks);
        }
      }

      const webReq = new Request(url, {
        method: req.method || "GET",
        headers,
        body,
      });

      const webRes = await server.fetch(webReq);

      res.statusCode = webRes.status;
      webRes.headers.forEach((val, key) => {
        res.setHeader(key, val);
      });
      const data = await webRes.arrayBuffer();
      res.end(Buffer.from(data));
      return;
    }

    // 3. Fallback: Request-like object with a url string
    if (req && req.url) {
      return await server.fetch(new Request(req.url, req));
    }
  } catch (err: any) {
    console.error("Vercel Serverless Function Error:", err);
    if (res && typeof res.status === "function") {
      res.status(500).json({
        error: "Internal Serverless Error",
        message: err?.message || String(err),
      });
      return;
    }
    return new Response(
      JSON.stringify(
        {
          error: "Internal Serverless Error",
          message: err?.message || String(err),
          stack: err?.stack,
        },
        null,
        2,
      ),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}

export const fetch = (req: Request) => handler(req);
export const GET = handler;
export const POST = handler;
export const OPTIONS = handler;
