function validateIp(ip: string): string {
  if (typeof ip !== "string" || ip.trim().length === 0) {
    throw new Error("IP cannot be empty");
  }
  return ip.trim();
}

/**
 * Generates an anonymized HMAC-SHA256 IP hash using Bun's native CryptoHasher & Uint8Array.fromHex.
 */
export async function generateIPHash(ip: string): Promise<string> {
  const keyHex = Bun.env.IP_HMAC_KEY_HEX;

  if (!keyHex || !/^[0-9a-fA-F]{64}$/.test(keyHex)) {
    throw new Error("IP_HMAC_KEY_HEX must be a 64-character hex-encoded HMAC key.");
  }

  const keyBytes = Uint8Array.fromHex(keyHex);
  const hmac = new Bun.CryptoHasher("sha256", keyBytes);
  hmac.update(validateIp(ip));
  const hashHex = hmac.digest("hex");

  return `anon_${hashHex.slice(0, 24)}`;
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

  return Uint8Array.fromHex(keyHex);
}

/** Encrypts an IP address using WebCrypto AES-GCM and Bun native Uint8Array.prototype.toBase64(). */
export async function encryptIP(ip: string): Promise<EncryptedIp> {
  const plaintext = validateIp(ip);
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
    nonce: nonce.toBase64(),
    ciphertext: new Uint8Array(ciphertext).toBase64(),
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
  const nonce = Uint8Array.fromBase64(nonceBase64);
  if (nonce.length !== 12) {
    throw new Error("AES-GCM nonce must be 12 bytes.");
  }
  const ciphertext = Uint8Array.fromBase64(ciphertextBase64);
  const plaintext = await crypto.subtle.decrypt(
    { name: "AES-GCM", iv: nonce },
    key,
    ciphertext,
  );

  return new TextDecoder().decode(plaintext);
}
