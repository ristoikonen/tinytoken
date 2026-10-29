import { z } from "zod";

const ipSchema = z.string().min(1, "IP cannot be empty");


export async function generateIPHash(ip: string): Promise<string> {
  const keyHex = Bun.env.IP_HMAC_KEY_HEX;

  if (!keyHex || !/^[0-9a-fA-F]{64}$/.test(keyHex)) {
    throw new Error("IP_HMAC_KEY_HEX must be a 64-character hex-encoded HMAC key.");
  }

  const keyBytes = new Uint8Array(32);
  for (let i = 0; i < keyBytes.length; i++) {
    keyBytes[i] = Number.parseInt(keyHex.slice(i * 2, i * 2 + 2), 16);
  }

  const key = await crypto.subtle.importKey(
    "raw",
    keyBytes,
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const mac = await crypto.subtle.sign(
    "HMAC",
    key,
    new TextEncoder().encode(ipSchema.parse(ip)),
  );
  const hashHex = Array.from(new Uint8Array(mac))
    .map((byte) => byte.toString(16).padStart(2, "0"))
    .join("");

  return `anon_${hashHex.substring(0, 24)}`;
}

export interface EncryptedIp {
  algorithm: "AES-256-GCM";
  keyVersion: string;
  nonce: string;
  ciphertext: string;
}

function normalizeEncryptionKeyVersion(keyVersion: string): string {
  const normalized = keyVersion.trim().toLowerCase();
  if (!/^v[1-9]\d*$/.test(normalized)) {
    throw new Error(`Unsupported IP encryption key version: ${keyVersion}`);
  }
  return normalized;
}

export function getEncryptionKeyBytes(keyVersion: string): Uint8Array<ArrayBuffer> {
  if (!/^v[1-9]\d*$/.test(keyVersion)) {
    throw new Error(`Unsupported IP encryption key version: ${keyVersion}`);
  }

  const versionedKeyName = `IP_ENCRYPTION_KEY_${keyVersion.toUpperCase()}_HEX`;
  const keyHex = Bun.env[versionedKeyName];

  if (!keyHex || !/^[0-9a-fA-F]{64}$/.test(keyHex)) {
    throw new Error(`${versionedKeyName} must be a 64-character hex-encoded AES-256 key.`);
  }

  const keyBytes = new Uint8Array(new ArrayBuffer(32));
  const keyHexPairs = keyHex.match(/.{2}/g)!;
  for (const [i, pair] of keyHexPairs.entries()) {
    keyBytes[i] = Number.parseInt(pair, 16);
  }
  return keyBytes;
}

function toBase64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) {
    binary += String.fromCharCode(byte);
  }
  return btoa(binary);
}

/** Encrypts an IP address; retain nonce and ciphertext together to decrypt later. */
export async function encryptIP(ip: string): Promise<EncryptedIp> {
  const plaintext = ipSchema.parse(ip);
  const configuredKeyVersion = Bun.env.IP_ENCRYPTION_KEY_VERSION;
  if (!configuredKeyVersion?.trim()) {
    throw new Error("IP_ENCRYPTION_KEY_VERSION is missing.");
  }
  const keyVersion = normalizeEncryptionKeyVersion(configuredKeyVersion);
  const keyBytes = getEncryptionKeyBytes(keyVersion);

  const key = await crypto.subtle.importKey(
    "raw",
    keyBytes,
    { name: "AES-GCM" },
    false,
    ["encrypt"],
  );
  const nonce = crypto.getRandomValues(new Uint8Array(12));
  const ciphertext = await crypto.subtle.encrypt(
    { name: "AES-GCM", iv: nonce },
    key,
    new TextEncoder().encode(plaintext),
  );

  return {
    algorithm: "AES-256-GCM",
    keyVersion,
    nonce: toBase64(nonce),
    ciphertext: toBase64(new Uint8Array(ciphertext)),
  };
}

export async function decryptIP(
  nonceBase64: string,
  ciphertextBase64: string,
  keyVersion: string,
): Promise<string> {
  const normalizedKeyVersion = normalizeEncryptionKeyVersion(keyVersion);
  const keyBytes = getEncryptionKeyBytes(normalizedKeyVersion);
  const key = await crypto.subtle.importKey(
    "raw",
    keyBytes,
    { name: "AES-GCM" },
    false,
    ["decrypt"],
  );
  const nonce = Uint8Array.from(atob(nonceBase64), (char) => char.charCodeAt(0));
  if (nonce.length !== 12) {
    throw new Error("AES-GCM nonce must be 12 bytes.");
  }
  const ciphertext = Uint8Array.from(
    atob(ciphertextBase64),
    (char) => char.charCodeAt(0),
  );
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: nonce },
    key,
    ciphertext,
  );

  return new TextDecoder().decode(plaintext);
}
