import { fetchHandler, ensureInitialized } from "../server";

async function handleRequest(req: Request): Promise<Response> {
  try {
    await ensureInitialized();
    return await fetchHandler(req);
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

const handler = (req: Request) => handleRequest(req);
handler.fetch = (req: Request) => handleRequest(req);

export default handler;
export const GET = (req: Request) => handleRequest(req);
export const POST = (req: Request) => handleRequest(req);
export const PUT = (req: Request) => handleRequest(req);
export const DELETE = (req: Request) => handleRequest(req);
export const OPTIONS = (req: Request) => handleRequest(req);
