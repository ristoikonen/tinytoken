
/**
 * Creates a standardized JSON response. CORS is enabled by default for any origin;
 * pass a specific origin to restrict access, or set enableCors to false to disable it.
 */
export function jsonResponse<T>(
  data: T,
  status = 200,
  enableCors = true,
  allowOrigin = "*",
): Response {
  const headers: Record<string, string> = {
    "Content-Type": "application/json; charset=utf-8",
  };

  if (enableCors) {
    headers["Access-Control-Allow-Origin"] = allowOrigin;
    headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS";
    headers["Access-Control-Allow-Headers"] = "Content-Type, X-Session-ID, X-Correlation-ID";
  }

  return new Response(JSON.stringify(data), {
    status,
    headers,
  });
}

export function corsResponse(
  status = 204,
  allowOrigin = "*",
): Response {
  const headers: Record<string, string> = {
    "Access-Control-Allow-Origin": allowOrigin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, X-Session-ID, X-Correlation-ID",
  }

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

  }

  return new Response(null, {
    status,
    headers,
  });
}

/**
 * Creates a standardized error JSON Response.
 */
export function errorResponse(
  message: string,
  status = 400,
  enableCors = true,
  allowOrigin = "*",
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