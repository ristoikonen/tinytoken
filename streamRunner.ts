#!/usr/bin/env bun
/**
 * Root CLI entrypoint for streamRunner
 * Delegates to ./utils/streamRunner.ts
 *
 * Usage:
 *   bun run streamRunner.ts /after 10 /format data
 *   bun run streamRunner.ts /format full /limit 5
 */

import { parseCommandLineArgs, runStreamClient } from "./utils/streamRunner";

const options = parseCommandLineArgs();
await runStreamClient(options);
