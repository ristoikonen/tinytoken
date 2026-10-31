import server, { ensureInitialized } from "../server";

export default async function handler(req: Request): Promise<Response> {
  await ensureInitialized();
  return server.fetch(req);
}

export const GET = handler;
export const POST = handler;
export const OPTIONS = handler;

