// Thin fetch wrapper. Non-2xx responses become ApiError with the server's "detail" message.

export class ApiError extends Error {
  constructor(status, detail) {
    super(detail);
    this.status = status;
    this.detail = detail;
  }
}

async function handle(response) {
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = typeof body?.detail === "string" ? body.detail
      : Array.isArray(body?.detail) ? body.detail.map((d) => d.msg).join("; ")
      : `The server answered ${response.status}.`;
    throw new ApiError(response.status, detail);
  }
  return body;
}

export async function getJSON(path, params = {}) {
  const query = new URLSearchParams(
    Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== ""),
  ).toString();
  let response;
  try {
    response = await fetch(query ? `${path}?${query}` : path, { headers: { Accept: "application/json" } });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check that it is running, then reload.");
  }
  return handle(response);
}

export async function postJSON(path, body) {
  let response;
  try {
    response = await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify(body),
    });
  } catch {
    throw new ApiError(0, "Can't reach the server. Check that it is running, then reload.");
  }
  return handle(response);
}
