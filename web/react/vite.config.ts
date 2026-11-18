import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Served by the same nginx gateway as the plain page, under /react/.
// `npm run dev` proxies /api to the running gateway, so the dev server
// talks to the real APIs without CORS.
export default defineConfig({
  base: "/react/",
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8080" },
    // The stylesheet is shared with the plain page in ../site.
    fs: { allow: [".."] },
  },
});
