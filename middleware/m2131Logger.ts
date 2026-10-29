import type { BunRequest } from "bun";
import { createClient } from "@libsql/client";
import { z } from "zod";
import { encryptIP, generateIPHash } from "../utils/hash";
import { errorResponse } from "../utils/response";

const ipAddressValidator = z.union([z.ipv4(), z.ipv6()]);

const db = createClient({
  url: Bun.env.TURSO_DATABASE_URL || "file:local.db",
  authToken: Bun.env.TURSO_AUTH_TOKEN,
});

export interface M2131LogRecord {
  timestamp: string; // AS ISO 8601 UTC millisecond precision: YYYY-MM-DDThh:mm:ss.mmmZ
  correlationId: string;
  method: string;
  path: string;
  clientIpHash: string;
  userAgent: string;
  statusCode?: number;
  durationMs: number;
  clientIpNonce: string | null;
  clientIpCiphertext: string | null;
  encryptionKeyVersion: string | null;
}

export interface M2131TelemetryOptions {
  getClientIp: (req: BunRequest) => string | null;
  storeRecoverableIp?: boolean;
}

/**
 * Saves a log record to the Turso SQLite m2131_logs table.
 */
async function saveLogToDb(
  record: M2131LogRecord,
  requireEncryptedStorage: boolean,
): Promise<void> {
  try {
    if (requireEncryptedStorage) {
      await db.execute({
        sql: `
          INSERT INTO m2131_logs (
            timestamp, correlation_id, method, path, client_ip_hash, user_agent,
            status_code, duration_ms, client_ip_nonce, client_ip_ciphertext,
            encryption_key_version
          ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        `,
        args: [
          record.timestamp,
          record.correlationId,
          record.method,
          record.path,
          record.clientIpHash,
          record.userAgent,
          record.statusCode ?? null,
          record.durationMs,
          record.clientIpNonce,
          record.clientIpCiphertext,
          record.encryptionKeyVersion,
        ],
      });
      return;
    }

    await db.execute({
      sql: `
        INSERT INTO m2131_logs (
          timestamp, correlation_id, method, path, client_ip_hash, user_agent, status_code, duration_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `,
      args: [
        record.timestamp,
        record.correlationId,
        record.method,
        record.path,
        record.clientIpHash,
        record.userAgent,
        record.statusCode ?? null,
        record.durationMs,
      ],
    });
  } catch (dbError) {
    console.error(
      "Telemetry write to Turso failed; continuing request without persisted telemetry.",
      dbError,
    );
    if (requireEncryptedStorage) {
      throw new Error("Failed to persist telemetry with encrypted IP.", { cause: dbError });
    }
  }
}

/**
 * Formats a log record according to M-21-31 telemetry and ACSC standards.
 */
export function formatM2131Log(record: M2131LogRecord): string {
  return JSON.stringify({
    schemaVersion: "M-21-31-ACSC-1.0",
    timestamp: record.timestamp,
    correlationId: record.correlationId,
    http: {
      method: record.method,
      path: record.path,
      statusCode: record.statusCode || 200,
      durationMs: record.durationMs,
    },
    security: {
      clientIpHash: record.clientIpHash,
      userAgent: record.userAgent,
    },
  });
}

/**
 * Middleware wrapper using BunRequest to capture request telemetry and save to Turso.
 */
export async function handleM2131Telemetry(
  req: BunRequest,
  next: (req: BunRequest) => Promise<Response>,
  options: M2131TelemetryOptions,
): Promise<Response> {
  const storeRecoverableIp = options.storeRecoverableIp ?? false;
  const startNs = Bun.nanoseconds();
  const correlationId = req.headers.get("x-correlation-id") || crypto.randomUUID();

  const rawIp = options.getClientIp(req);
  if (!rawIp || !ipAddressValidator.safeParse(rawIp).success) {
    console.error("Unable to determine a valid client IP from the connection.");
    return errorResponse("Client IP is unavailable.", 503);
  }

  const clientIpHash = await generateIPHash(rawIp);
  let clientIpNonce: string | null = null;
  let clientIpCiphertext: string | null = null;
  let encryptionKeyVersion: string | null = null;

  if (storeRecoverableIp) {
    encryptionKeyVersion = Bun.env.IP_ENCRYPTION_KEY_VERSION?.trim() || null;
    if (!encryptionKeyVersion) {
      console.error("Recoverable IP logging is enabled but IP_ENCRYPTION_KEY_VERSION is missing.");
      return errorResponse("Encrypted IP logging is unavailable.", 503);
    }

    try {
      const encryptedIp = await encryptIP(rawIp);
      encryptionKeyVersion = encryptedIp.keyVersion;
      clientIpNonce = encryptedIp.nonce;
      clientIpCiphertext = encryptedIp.ciphertext;
    } catch (error) {
      console.error("Recoverable IP logging encryption failed.", error);
      return errorResponse("Encrypted IP logging is unavailable.", 503);
    }
  }

  const userAgent = req.headers.get("user-agent") || "unknown";
  const url = new URL(req.url);

  let response: Response;
  try {
    response = await next(req);
  } catch (error) {
    const durationMs = Math.round((Bun.nanoseconds() - startNs) / 1_000) / 1_000;
    const errorRecord: M2131LogRecord = {
      timestamp: new Date().toISOString(),
      correlationId,
      method: req.method,
      path: url.pathname,
      clientIpHash,
      userAgent,
      statusCode: 500,
      durationMs,
      clientIpNonce,
      clientIpCiphertext,
      encryptionKeyVersion,
    };

    console.error(formatM2131Log(errorRecord));
    await saveLogToDb(errorRecord, storeRecoverableIp);

    throw error;
  }

  const durationMs = Math.round((Bun.nanoseconds() - startNs) / 1_000) / 1_000;
  const logRecord: M2131LogRecord = {
    timestamp: new Date().toISOString(),
    correlationId,
    method: req.method,
    path: url.pathname,
    clientIpHash,
    userAgent,
    statusCode: response.status,
    durationMs,
    clientIpNonce,
    clientIpCiphertext,
    encryptionKeyVersion,
  };

  console.log("LOG:" + formatM2131Log(logRecord));
  await saveLogToDb(logRecord, storeRecoverableIp);

  const newHeaders = new Headers(response.headers);
  newHeaders.set("X-Correlation-ID", correlationId);

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers: newHeaders,
  });
}
