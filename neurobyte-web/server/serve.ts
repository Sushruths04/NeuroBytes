/**
 * Production server for the Docker image: serves the built app (dist/) and the same
 * read-only data API as the dev server (server/dataPlugin.ts), with no Vite at runtime.
 *
 *   node server/serve.ts            # Node >= 22.18 runs the .ts directly (type stripping)
 *
 *   HOST (default 127.0.0.1), PORT (default 8080), plus the NEUROBYTES_* paths of server/paths.ts.
 *
 * The data API has no authentication. Inside Docker HOST is 0.0.0.0, so publish the
 * port on localhost only: docker run -p 127.0.0.1:8080:8080 ...
 */
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { createHandler } from "./dataPlugin.ts";
import { resolvePaths } from "./paths.ts";

const APP_DIR = path.resolve(import.meta.dirname, "..");
const DIST = path.join(APP_DIR, "dist");
const HOST = process.env.HOST ?? "127.0.0.1";
const PORT = Number(process.env.PORT ?? 8080);

const TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

const paths = resolvePaths(APP_DIR);
const api = createHandler(paths);

/** Static files from dist/; anything else gets index.html (the app routes by #hash). */
function serveStatic(req: http.IncomingMessage, res: http.ServerResponse) {
  if (req.method !== "GET" && req.method !== "HEAD") {
    res.writeHead(405).end("method not allowed");
    return;
  }
  let rel: string;
  try {
    rel = decodeURIComponent((req.url ?? "/").split("?")[0]);
  } catch {
    res.writeHead(400).end("bad request");
    return;
  }
  // Resolve inside dist/ only: no path traversal.
  let file = path.resolve(DIST, "." + path.posix.normalize("/" + rel));
  if (!file.startsWith(DIST + path.sep) && file !== DIST) {
    res.writeHead(403).end("forbidden");
    return;
  }
  if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) file = path.join(DIST, "index.html");
  const hashed = file.includes(`${path.sep}assets${path.sep}`); // Vite fingerprints these
  res.writeHead(200, {
    "Content-Type": TYPES[path.extname(file)] ?? "application/octet-stream",
    "Content-Length": fs.statSync(file).size,
    "Cache-Control": hashed ? "public, max-age=31536000, immutable" : "no-cache",
  });
  if (req.method === "HEAD") return void res.end();
  fs.createReadStream(file).pipe(res);
}

const server = http.createServer((req, res) => {
  res.setHeader("X-Content-Type-Options", "nosniff");
  res.setHeader("Referrer-Policy", "no-referrer");
  res.setHeader("X-Frame-Options", "DENY");
  api(req, res, () => serveStatic(req, res));
});

server.listen(PORT, HOST, () => {
  console.log(`NeuroBytes on http://${HOST}:${PORT}`);
  console.log(`  data: ${paths.dataRoot}\n  submission: ${paths.submissionDir}\n  models: ${paths.modelsDir}\n  label reads logged to ${paths.labelLog}`);
});

for (const sig of ["SIGINT", "SIGTERM"] as const) process.on(sig, () => server.close(() => process.exit(0)));
