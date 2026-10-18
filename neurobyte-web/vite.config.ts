import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { dataPlugin } from "./server/dataPlugin.ts";
import { resolvePaths } from "./server/paths.ts";

export default defineConfig({
  plugins: [react(), tailwindcss(), dataPlugin(resolvePaths(import.meta.dirname))],
  // Local only: the data API has no authentication.
  server: { host: "127.0.0.1", port: 5173 },
  preview: { host: "127.0.0.1", port: 4173 },
});
