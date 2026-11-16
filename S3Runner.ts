#!/usr/bin/env bun
/**
 * Root CLI entrypoint for Cloudflare R2 / S3 Runner
 * Delegates to ./utils/S3Runner.ts
 *
 * Usage:
 *   bun run S3Runner.ts
 *   bun run S3Runner.ts /write /key tokens/test.tiny
 *   bun run S3Runner.ts /read /key tokens/test.tiny
 *   bun run S3Runner.ts /list
 */

import { runS3Runner } from "./utils/S3Runner";

await runS3Runner(process.argv.slice(2));
