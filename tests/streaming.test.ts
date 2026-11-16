import { describe, expect, test } from "bun:test";
import { StreamingArchiveHandlers } from "../handlers/streamingArchiveHandlers";
import { TOKEN_MAGIC } from "../models/token";
import { consumeTinyTokenStream, createFramedToken } from "../utils/streamConsumer";
import { inspectBinaryTokenHeader } from "../utils/tokenCodec";
import { CompressedTokenSearchEngine } from "../utils/zipEngine";

describe("application/vnd.tinytoken.stream Binary Streaming Pipeline", () => {
  const searchEngine = new CompressedTokenSearchEngine("./data/tokens.zip", {
    useZip: true,
    dataDirPath: "./data",
  });

  const handlers = new StreamingArchiveHandlers(searchEngine);

  test("consumeTinyTokenStream correctly frames 4-byte Uint32LE prefixed tokens", async () => {
    // Generate dummy token buffers
    const tokenA = new Uint8Array(100).fill(0xaa);
    const tokenB = new Uint8Array(250).fill(0xbb);
    const tokenC = new Uint8Array(50).fill(0xcc);

    const frameA = createFramedToken(tokenA);
    const frameB = createFramedToken(tokenB);
    const frameC = createFramedToken(tokenC);

    expect(frameA.byteLength).toBe(104);
    expect(frameB.byteLength).toBe(254);
    expect(frameC.byteLength).toBe(54);

    // Stream frames as a single ReadableStream
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(frameA);
        controller.enqueue(frameB);
        controller.enqueue(frameC);
        controller.close();
      },
    });

    const received: Uint8Array[] = [];
    for await (const tok of consumeTinyTokenStream(stream)) {
      received.push(tok);
    }

    expect(received.length).toBe(3);
    expect(received[0]).toEqual(tokenA);
    expect(received[1]).toEqual(tokenB);
    expect(received[2]).toEqual(tokenC);
  });

  test("consumeTinyTokenStream survives arbitrary TCP chunk fragmentation (e.g. 17-byte chunks)", async () => {
    const tokenA = new Uint8Array(80).fill(0x11);
    const tokenB = new Uint8Array(120).fill(0x22);

    const allBytes = new Uint8Array(
      createFramedToken(tokenA).length + createFramedToken(tokenB).length,
    );
    allBytes.set(createFramedToken(tokenA), 0);
    allBytes.set(createFramedToken(tokenB), createFramedToken(tokenA).length);

    // Fragment into small 17-byte TCP packets
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        const chunkSize = 17;
        for (let i = 0; i < allBytes.length; i += chunkSize) {
          controller.enqueue(allBytes.subarray(i, Math.min(allBytes.length, i + chunkSize)));
        }
        controller.close();
      },
    });

    const tokens: Uint8Array[] = [];
    for await (const tok of consumeTinyTokenStream(stream)) {
      tokens.push(tok);
    }

    expect(tokens.length).toBe(2);
    expect(tokens[0]).toEqual(tokenA);
    expect(tokens[1]).toEqual(tokenB);
  });

  test("End-to-end HTTP binary streaming: GET /api/v1/archives/active/tokens?format=full", async () => {
    await searchEngine.buildIndex();
    const count = searchEngine.getCount();
    expect(count).toBeGreaterThan(0);

    const url = new URL("http://localhost:3000/api/v1/archives/active/tokens?format=full&limit=3");
    const req = new Request(url.toString(), {
      headers: {
        Accept: "application/vnd.tinytoken.stream",
      },
    });

    const res = await handlers.handleStreamArchiveTokens("active", req, url);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/vnd.tinytoken.stream");
    expect(res.headers.get("x-total-tokens")).toBe(String(Math.min(3, count)));

    const stream = res.body;
    expect(stream).not.toBeNull();

    const streamedTokens: Uint8Array[] = [];
    for await (const tokenBytes of consumeTinyTokenStream(stream!)) {
      streamedTokens.push(tokenBytes);
      const insp = inspectBinaryTokenHeader(tokenBytes);
      expect(insp.magic).toBe(TOKEN_MAGIC);
      expect(insp.totalTokenBytes).toBe(tokenBytes.byteLength);
      expect(insp.totalTokenBytes).toBeLessThanOrEqual(25600);
    }

    expect(streamedTokens.length).toBe(Math.min(3, count));
  });

  test("format=data omits preview thumbnail section and reduces payload size by ~80%", async () => {
    await searchEngine.buildIndex();
    const url = new URL("http://localhost:3000/api/v1/archives/active/tokens?format=data&limit=1");
    const req = new Request(url.toString(), {
      headers: {
        Accept: "application/vnd.tinytoken.stream",
      },
    });

    const res = await handlers.handleStreamArchiveTokens("active", req, url);
    expect(res.status).toBe(200);

    const tokens: Uint8Array[] = [];
    for await (const tokenBytes of consumeTinyTokenStream(res.body!)) {
      tokens.push(tokenBytes);
      const insp = inspectBinaryTokenHeader(tokenBytes);
      expect(insp.magic).toBe(TOKEN_MAGIC);
      // format=data size should strictly equal 704 + metaLength (no preview section)
      expect(tokenBytes.byteLength).toBe(704 + insp.metaLength);
      expect(tokenBytes.byteLength).toBeLessThan(5000); // substantially smaller than 25 KB
    }

    expect(tokens.length).toBe(1);
  });

  test("Resumable cursor (?after=N) skips previous tokens and resumes stream seamlessly", async () => {
    await searchEngine.buildIndex();
    if (searchEngine.getCount() < 2) return;

    // 1. Fetch first 2 tokens
    const url1 = new URL("http://localhost:3000/api/v1/archives/active/tokens?format=data&limit=2");
    const req1 = new Request(url1.toString(), { headers: { Accept: "application/vnd.tinytoken.stream" } });
    const res1 = await handlers.handleStreamArchiveTokens("active", req1, url1);

    const batch1: Uint8Array[] = [];
    for await (const tok of consumeTinyTokenStream(res1.body!)) {
      batch1.push(tok);
    }
    expect(batch1.length).toBe(2);

    const token0Header = inspectBinaryTokenHeader(batch1[0]!);
    const token1Header = inspectBinaryTokenHeader(batch1[1]!);

    // 2. Resume with ?after=1 (skip index 0)
    const url2 = new URL(
      "http://localhost:3000/api/v1/archives/active/tokens?format=data&after=1&limit=1",
    );
    const req2 = new Request(url2.toString(), { headers: { Accept: "application/vnd.tinytoken.stream" } });
    const res2 = await handlers.handleStreamArchiveTokens("active", req2, url2);

    const batch2: Uint8Array[] = [];
    for await (const tok of consumeTinyTokenStream(res2.body!)) {
      batch2.push(tok);
    }
    expect(batch2.length).toBe(1);

    const resumedHeader = inspectBinaryTokenHeader(batch2[0]!);
    expect(resumedHeader.uuidHex).toBe(token1Header.uuidHex);
    expect(resumedHeader.uuidHex).not.toBe(token0Header.uuidHex);
  });

  test("Single token streaming: GET /api/v1/archives/active/tokens/{tokenId}", async () => {
    await searchEngine.buildIndex();
    const firstEntry = searchEngine.getAllEntries()[0];
    if (!firstEntry) return;

    const url = new URL(
      `http://localhost:3000/api/v1/archives/active/tokens/${firstEntry.id}?format=data`,
    );
    const req = new Request(url.toString(), { headers: { Accept: "application/vnd.tinytoken.stream" } });

    const res = await handlers.handleGetArchiveToken("active", firstEntry.id, req, url);
    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/vnd.tinytoken.stream");

    // Sliced single framed token
    const stream = new ReadableStream<Uint8Array>({
      async start(c) {
        c.enqueue(new Uint8Array(await res.arrayBuffer()));
        c.close();
      },
    });

    const tokens: Uint8Array[] = [];
    for await (const tok of consumeTinyTokenStream(stream)) {
      tokens.push(tok);
    }

    expect(tokens.length).toBe(1);
    const tok0 = tokens[0]!;
    const insp = inspectBinaryTokenHeader(tok0);
    expect(insp.magic).toBe(TOKEN_MAGIC);
    expect(tok0.byteLength).toBe(704 + insp.metaLength);
  });
});
