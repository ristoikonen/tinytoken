import { S3Client } from "bun";

/**
 * 
bun run S3Runner.ts

[R2] Target: bun-app/tokens/doc-001.tiny (https://6eae64fa83e289be218119704100c3b1.r2.cloudflarestorage.com)
[R2] Write OK: 25600B in 890.67ms
[R2] Read OK: 25600B in 240.58ms

bun run S3Runner.ts /list
[R2] Listed in 528.98ms: [ "tokens/doc-001.tiny (25600B)" ]

 * 
**/


export function getR2Client(): { s3: S3Client; bucket: string; endpoint: string } {
    const endpoint = Bun.env.S3_ENDPOINT || "https://6eae64fa83e289be218119704100c3b1.r2.cloudflarestorage.com";
    const accessKeyId = Bun.env.S3_ACCESS_KEY || "";
    const secretAccessKey = Bun.env.S3_SECRET || "";
    const bucket =
        Bun.env.S3_BUCKET ||
        (Bun.env.S3_API ? Bun.env.S3_API.replace(/\/+$/, "").split("/").pop()! : "bun-app");

    const s3 = new S3Client({
        accessKeyId,
        secretAccessKey,
        bucket,
        endpoint,
        region: "auto",
    });

    return { s3, bucket, endpoint };
}

export async function runS3Runner(args: string[] = process.argv.slice(2)): Promise<void> {
    const { s3, bucket, endpoint } = getR2Client();

    let mode: "roundtrip" | "write" | "read" | "list" | "delete" = "roundtrip";
    let key = "tokens/doc-001.tiny";
    let sizeKb = 25;

    for (let i = 0; i < args.length; i++) {
        const arg = args[i]!.toLowerCase();
        if (arg === "/write" || arg === "--write" || arg === "/put") mode = "write";
        else if (arg === "/read" || arg === "--read" || arg === "/get") mode = "read";
        else if (arg === "/list" || arg === "--list" || arg === "/ls") mode = "list";
        else if (arg === "/delete" || arg === "--delete" || arg === "/del") mode = "delete";
        else if ((arg === "/key" || arg === "--key") && args[i + 1]) key = args[++i]!;
        else if ((arg === "/size" || arg === "--size") && args[i + 1]) sizeKb = parseInt(args[++i]!, 10);
        else if (arg === "/?" || arg === "/help" || arg === "--help") {
            console.log(`Cloudflare R2 S3 Runner\nUsage: bun run S3Runner.ts [/write] [/read] [/list] [/delete] [/key <path>] [/size <kb>]\nBucket: ${bucket} | Endpoint: ${endpoint}`);
            return;
        }
    }

    console.log(`[R2] Target: ${bucket}/${key} (${endpoint})`);

    try {
        if (mode === "list") {
            const t0 = performance.now();
            const listing = await s3.list({ prefix: key.startsWith("tokens/") ? "tokens" : "" });
            const elapsed = (performance.now() - t0).toFixed(2);
            console.log(`[R2] Listed in ${elapsed}ms:`, listing.contents?.map((c) => `${c.key} (${c.size}B)`) ?? []);
            return;
        }

        if (mode === "delete") {
            const t0 = performance.now();
            await s3.delete(key);
            console.log(`[R2] Deleted ${key} in ${(performance.now() - t0).toFixed(2)}ms`);
            return;
        }

        // Prepare payload (default: 25 KB binary token buffer)
        const payload = new Uint8Array(sizeKb * 1024);
        payload.fill(0x5a);

        if (mode === "write" || mode === "roundtrip") {
            const t0 = performance.now();
            await s3.write(key, payload, { type: "application/octet-stream" });
            const ms = (performance.now() - t0).toFixed(2);
            console.log(`[R2] Write OK: ${payload.byteLength}B in ${ms}ms`);
        }

        if (mode === "read" || mode === "roundtrip") {
            const t0 = performance.now();
            const fileRef = s3.file(key);
            if (!(await fileRef.exists())) {
                console.log(`[R2] Key not found: ${key}`);
                return;
            }
            const data = await fileRef.arrayBuffer();
            const ms = (performance.now() - t0).toFixed(2);
            console.log(`[R2] Read OK: ${data.byteLength}B in ${ms}ms`);
        }
    } catch (err: any) {
        console.error(`[R2 ERR] ${err.name || "Error"}: ${err.message || err}`);
        if (String(err).includes("AccessDenied") || String(err).includes("Access Denied")) {
            console.error(`[HINT] Check Cloudflare R2 token IP restrictions (Client IP Filtering) or bucket permissions.`);
        }
    }
}

if (import.meta.main) {
    await runS3Runner();
}
