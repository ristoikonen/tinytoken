/**
 * Archive Registry Models & Configuration (models/archive.ts)
 *
 * Defines the persisted and in-memory `TokenArchive` representation mapping
 * stable archive IDs to approved `.zip` containers inside `APP_ARCHIVE_DIR`.
 */

export const DEFAULT_ARCHIVE_DIR = "./data/archives";

export function getAppArchiveDir(): string {
  return (Bun.env.APP_ARCHIVE_DIR ?? DEFAULT_ARCHIVE_DIR)
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/, "");
}

export function getArchiveRegistryPath(): string {
  const dataDir = (Bun.env.APP_DATA_DIR ?? Bun.env.DATA_DIR ?? "./data")
    .trim()
    .replace(/\\/g, "/")
    .replace(/\/+$/, "");
  return `${dataDir}/archive_registry.json`;
}

export interface TokenArchive {
  id: string;
  filename: string;
  path: string;
  size: number;
  modified: Date;
  label?: string;
  sha256?: string;
}

export interface SerializedTokenArchive {
  id: string;
  filename: string;
  path: string;
  size: number;
  modified: string;
  label?: string;
  sha256?: string;
}
