import server, { ensureInitialized } from "../server";

export default async function handler(req: Request): Promise<Response> {
  try {
    await ensureInitialized();
    return await server.fetch(req);
  } catch (err: any) {
    console.error("Vercel Serverless Function Error:", err);
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

export const GET = handler;
export const POST = handler;
export const OPTIONS = handler;
