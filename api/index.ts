import server, { ensureInitialized } from "../server";

async function handleRequest(req: Request): Promise<Response> {
  try {
    await ensureInitialized();
    return await server.fetch(req);
  } catch (err: any) {
    console.error("Vercel Bun Function Error:", err);
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

export default {
  fetch(req: Request): Promise<Response> {
    return handleRequest(req);
  },
};

export const GET = (req: Request) => handleRequest(req);
export const POST = (req: Request) => handleRequest(req);
export const OPTIONS = (req: Request) => handleRequest(req);
