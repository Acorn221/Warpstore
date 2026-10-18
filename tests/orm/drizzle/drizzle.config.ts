import { defineConfig } from "drizzle-kit";
export default defineConfig({ dialect: "mysql", schema: "./schema.ts", out: "./migrations",
  dbCredentials: { host: "127.0.0.1", port: Number(process.env.WARPSTORE_PORT || 3307), user: "root", database: "app" } });
