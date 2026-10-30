/**
 * Standardized HTTP Response & Same-Origin / CORS Security Helpers (utils/response.ts)
 *
 * - Disables wildcard (`*`) CORS globally across JSON and preflight responses.
 * - Restricts preflight (`OPTIONS`) and CORS headers to same-origin or explicit
 *   `CORS_ALLOWED_ORIGINS` allowlist entries.
 * - Enforces Fetch Metadata (`Sec-Fetch-Site`), `Origin`, and `Content-Type: application/json`
 *   verification on state-mutating `POST` routes to block cross-origin CSRF / `no-cors` attacks.
 */

function getConfiguredAllowedOrigins(): Set<string> {
  const raw = Bun.env.CORS_ALLOWED_ORIGINS ?? "";
  const allowed = new Set<string>();
  for (const part of raw.split(",")) {
    const trimmed = part.trim();
    if (trimmed && trimmed !== "*") {
      allowed.add(trimmed);
    }
  }
  return allowed;
}

/**
 * Resolves a validated, non-wildcard origin if `req`'s `Origin` matches the server's own origin
 * or an explicit entry in `CORS_ALLOWED_ORIGINS`. Returns `null` for disallowed or missing origins.
 */
export function resolveAllowedOrigin(req?: Request): string | null {
  if (!req) return null;
  const origin = req.headers.get("origin")?.trim();
  if (!origin || origin === "null" || origin === "*") {
    return null;
  }

  try {
    const serverOrigin = new URL(req.url).origin;
    if (origin === serverOrigin) {
      return origin;
    }
  } catch {
    return null;
  }

  const configured = getConfiguredAllowedOrigins();
  return configured.has(origin) ? origin : null;
}

function constantTimeEquals(a: string, b: string): boolean {
  const enc = new TextEncoder();
  const bufA = enc.encode(a);
  const bufB = enc.encode(b);
  if (bufA.length !== bufB.length) return false;
  let diff = 0;
  for (let i = 0; i < bufA.length; i++) {
    diff |= (bufA[i] ?? 0) ^ (bufB[i] ?? 0);
  }
  return diff === 0;
}

/**
 * Guards destructive or CPU-intensive `POST` endpoints (`/api/index/rebuild`, `/api/token/:id/mutate`)
 * against cross-origin browser requests (`fetch`, `XMLHttpRequest`, `<form>`, and `mode: "no-cors"`).
 * Returns a `403`/`401`/`415` error `Response` when blocked, or `null` when safe to proceed.
 */
export function verifySameOriginMutation(req?: Request): Response | null {
  if (!req) {
    return errorResponse("Missing request context for mutating operation", 403);
  }

  // 1. Optional Admin Bearer / Header token enforcement when TINYTOKEN_ADMIN_TOKEN is configured
  const requiredToken = Bun.env.TINYTOKEN_ADMIN_TOKEN?.trim();
  if (requiredToken) {
    const authHeader = req.headers.get("authorization") ?? "";
    const bearer = authHeader.startsWith("Bearer ") ? authHeader.slice(7).trim() : "";
    const headerToken = req.headers.get("x-admin-token")?.trim() ?? "";
    const candidate = bearer || headerToken;
    if (!candidate || !constantTimeEquals(candidate, requiredToken)) {
      return errorResponse("Unauthorized mutation request", 401);
    }
  }

  // 2. W3C Fetch Metadata check (browser-enforced, unspoofable from web JS)
  const secFetchSite = req.headers.get("sec-fetch-site")?.trim().toLowerCase();
  if (secFetchSite && secFetchSite !== "same-origin" && secFetchSite !== "none") {
    return errorResponse("Cross-origin POST forbidden (Sec-Fetch-Site)", 403);
  }

  // 3. Strict Origin header verification (sent by browsers on all CORS and same/cross-origin POSTs)
  let serverOrigin: string;
  try {
    serverOrigin = new URL(req.url).origin;
  } catch {
    return errorResponse("Invalid request URL", 400);
  }

  const origin = req.headers.get("origin")?.trim();
  if (origin && origin !== serverOrigin) {
    return errorResponse("Cross-origin POST forbidden (Origin mismatch)", 403);
  }

  // Fallback Referer check if browser omitted Origin/Sec-Fetch-Site
  if (!origin && !secFetchSite) {
    const referer = req.headers.get("referer")?.trim();
    if (referer) {
      try {
        if (new URL(referer).origin !== serverOrigin) {
          return errorResponse("Cross-origin POST forbidden (Referer mismatch)", 403);
        }
      } catch {
        return errorResponse("Malformed Referer header", 403);
      }
    }
  }

  // 4. Require `Content-Type: application/json` so HTML `<form>` and `fetch(..., { mode: "no-cors" })`
  //    simple cross-origin requests cannot bypass preflight checks.
  const contentType = (req.headers.get("content-type") ?? "").toLowerCase();
  if (!contentType.includes("application/json")) {
    return errorResponse("Mutating endpoints require Content-Type: application/json", 415);
  }

  return null;
}

/**
 * Creates a standardized JSON response.
 * CORS is disabled by default (`enableCors = false`) and wildcard origins (`*`) are never emitted.
 */
export function jsonResponse<T>(
  data: T,
  status = 200,
  enableCors = false,
  allowOrigin = "",
): Response {
  const headers: Record<string, string> = {
    "Content-Type": "application/json; charset=utf-8",
    "X-Content-Type-Options": "nosniff",
  };

  if (enableCors && allowOrigin && allowOrigin !== "*") {
    headers["Access-Control-Allow-Origin"] = allowOrigin;
    headers["Access-Control-Allow-Methods"] = "GET, OPTIONS";
    headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-ID, X-Correlation-ID";
    headers["Vary"] = "Origin";
  }

  return new Response(JSON.stringify(data), {
    status,
    headers,
  });
}

/**
 * Handles `OPTIONS` preflight requests strictly for same-origin or explicitly allowlisted origins.
 * Cross-origin preflights from untrusted websites receive `403 Forbidden` with zero CORS headers.
 */
export function corsResponse(
  req?: Request,
  status = 204,
): Response {
  const allowedOrigin = resolveAllowedOrigin(req);
  if (!allowedOrigin) {
    return errorResponse("Cross-origin preflight forbidden", 403);
  }

  const isSameOrigin = (() => {
    try {
      return req ? allowedOrigin === new URL(req.url).origin : false;
    } catch {
      return false;
    }
  })();

  const headers: Record<string, string> = {
    "Access-Control-Allow-Origin": allowedOrigin,
    "Access-Control-Allow-Methods": isSameOrigin ? "GET, POST, OPTIONS" : "GET, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Session-ID, X-Correlation-ID",
    "Vary": "Origin",
    "X-Content-Type-Options": "nosniff",
  };

  return new Response(null, {
    status,
    headers,
  });
}

export function fileResponse(
  status = 204,
  fileType = "",
): Response {
  const headers: Record<string, string> = {
    "Content-Type": fileType,
    "Cache-Control": "public, max-age=3600",
  };

  return new Response(null, {
    status,
    headers,
  });
}

/**
 * Creates a standardized error JSON Response (CORS disabled by default, wildcard never emitted).
 */
export function errorResponse(
  message: string,
  status = 400,
  enableCors = false,
  allowOrigin = "",
): Response {
  return jsonResponse(
    {
      success: false,
      error: message,
      timestamp: new Date().toISOString(), // AS ISO 8601 UTC
    },
    status,
    enableCors,
    allowOrigin,
  );
}
