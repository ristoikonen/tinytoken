/**
 * Ed25519 Asymmetric Cryptographic Signature & 4-Leaf SHA-256 Merkle Provenance Engine
 * (utils/snarkAttest.ts — 256-Byte Section I Attestation Block, bytes 128..383)
 *
 * Replaces the previous keyless hash/GF(2^8) stub with a genuine asymmetric Ed25519 signature
 * (via Bun's native BoringSSL WebCrypto `crypto.subtle`) over a domain-separated SHA-256 Merkle tree
 * covering ALL sections of the 25 KB token:
 *   - Leaf 0: Origin Header Identity (`uuid`, `originalFileSize`, `width`, `height`, `page`, `previewFormat`, `embeddingModelId`)
 *   - Leaf 1: Section III Preview Payload (`previewBytes`)
 *   - Leaf 2: Section II Metadata & OCR Payload (`metaBytes`)
 *   - Leaf 3: Section I 16x16 VQ Codebook Grid (`vqIndices`)
 *
 * 256-Byte Attestation Block Layout (Offsets relative to `SNARK_STUB_OFFSET = 128`):
 * - 0..3     (4B)  : Magic `"ED25"` (`0x45, 0x44, 0x32, 0x35`)
 * - 4..35    (32B) : Leaf 0 — Origin Header SHA-256 Commitment
 * - 36..67   (32B) : Leaf 1 — Section III Preview SHA-256 Commitment
 * - 68..99   (32B) : Leaf 2 — Section II Metadata & OCR SHA-256 Commitment
 * - 100..131 (32B) : Leaf 3 — Section I VQ 16x16 Grid SHA-256 Commitment
 * - 132..163 (32B) : Signer Ed25519 Raw Public Key (`32 bytes`)
 * - 164..227 (64B) : Ed25519 Detached Signature (`64 bytes`) over canonical Merkle transcript
 * - 228..235 (8B)  : Signer Key ID (`SHA-256("TT-KEYID-v3" || pubKey)[0..7]`)
 * - 236..255 (20B) : GF(2^8) Polynomial Evaluation Audit Tags over `signature[0..63]`
 */

import type { EmbeddingModelId, PreviewFormat, SnarkAttestationInfo } from "../models/token";
import { gfPolyEval } from "./galois";

const ED25519_ATTEST_MAGIC = [0x45, 0x44, 0x32, 0x35] as const; // "ED25"
const DEFAULT_KEYPAIR_PATH = "./data/attestation_ed25519.json";
const TRANSCRIPT_DOMAIN = new TextEncoder().encode("TinyToken-Ed25519-Attest-v3\0"); // 28 bytes

export interface AuthorityAttestationKeys {
  privateKey: CryptoKey | null;
  publicKey: CryptoKey;
  publicKeyRaw: Uint8Array<ArrayBuffer>;
  keyIdBytes: Uint8Array<ArrayBuffer>;
}

type BunJwk = Exclude<Awaited<ReturnType<typeof crypto.subtle.exportKey>>, ArrayBuffer>;

interface PersistedKeyFile {
  protocol: "ED25519-MERKLE-GF28-v3";
  keyIdHex: string;
  publicKeyHex: string;
  privateKeyJwk: BunJwk;
  createdAt: string;
}

export interface OriginCommitmentInputs {
  uuidBytes: Uint8Array; // 16B
  originalFileSize: number;
  width: number;
  height: number;
  page: number;
  previewFormat: PreviewFormat;
  embeddingModelId: EmbeddingModelId;
}

export interface AttestationInputs extends OriginCommitmentInputs {
  vqIndices: Uint8Array; // 256B
  metaBytes: Uint8Array; // Section II raw/compressed bytes
  previewBytes: Uint8Array; // Section III preview bytes
  signingKeys?: AuthorityAttestationKeys;
}

export interface VerifyAttestationInputs extends OriginCommitmentInputs {
  snarkStub: Uint8Array; // 256B
  merkleRoot: Uint8Array; // 32B
  vqIndices: Uint8Array; // 256B
  metaBytes: Uint8Array; // Section II slice from raw token
  previewBytes: Uint8Array; // Section III slice from raw token
  envelopeBoundsValid?: boolean;
  trustedPublicKeyRaw?: Uint8Array;
}

export function bytesToHex(bytes: Uint8Array): string {
  return bytes.toHex();
}

function constantTimeBytesEqual(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) {
    diff |= (a[i] ?? 0) ^ (b[i] ?? 0);
  }
  return diff === 0;
}

function sha256Domain(domainAscii: string, payload: Uint8Array): Uint8Array<ArrayBuffer> {
  const hasher = new Bun.CryptoHasher("sha256");
  hasher.update(domainAscii);
  hasher.update(payload);
  const digest = hasher.digest();
  const out = new Uint8Array(new ArrayBuffer(32));
  out.set(new Uint8Array(digest.buffer, digest.byteOffset, 32));
  return out;
}

function computeKeyId(publicKeyRaw: Uint8Array): Uint8Array<ArrayBuffer> {
  const full = sha256Domain("TT-KEYID-v3:", publicKeyRaw);
  const kid = new Uint8Array(new ArrayBuffer(8));
  kid.set(full.subarray(0, 8));
  return kid;
}

function computeOriginCommitment(inputs: OriginCommitmentInputs): Uint8Array<ArrayBuffer> {
  const desc = new Uint8Array(30);
  desc.set(inputs.uuidBytes.subarray(0, 16), 0);
  const view = new DataView(desc.buffer);
  view.setUint32(16, inputs.originalFileSize >>> 0, true);
  view.setUint16(20, inputs.width & 0xffff, true);
  view.setUint16(22, inputs.height & 0xffff, true);
  view.setUint16(24, inputs.page & 0xffff, true);
  view.setUint16(26, inputs.previewFormat & 0xffff, true);
  desc[28] = inputs.embeddingModelId & 0xff;
  desc[29] = 0x03; // Attestation layout v3
  return sha256Domain("TT-ORIGIN-v3:", desc);
}

function computeMerkleRoot4(
  originHash: Uint8Array,
  previewHash: Uint8Array,
  metaHash: Uint8Array,
  vqHash: Uint8Array,
): Uint8Array<ArrayBuffer> {
  const leftBuf = new Uint8Array(65);
  leftBuf[0] = 0x01;
  leftBuf.set(originHash, 1);
  leftBuf.set(previewHash, 33);
  const leftBranch = sha256Domain("TT-NODE:", leftBuf);

  const rightBuf = new Uint8Array(65);
  rightBuf[0] = 0x01;
  rightBuf.set(metaHash, 1);
  rightBuf.set(vqHash, 33);
  const rightBranch = sha256Domain("TT-NODE:", rightBuf);

  const rootBuf = new Uint8Array(65);
  rootBuf[0] = 0x02;
  rootBuf.set(leftBranch, 1);
  rootBuf.set(rightBranch, 33);
  return sha256Domain("TT-ROOT:", rootBuf);
}

function buildSigningTranscript(
  merkleRoot: Uint8Array,
  originHash: Uint8Array,
  previewHash: Uint8Array,
  metaHash: Uint8Array,
  vqHash: Uint8Array,
): Uint8Array<ArrayBuffer> {
  const transcript = new Uint8Array(new ArrayBuffer(28 + 32 * 5));
  transcript.set(TRANSCRIPT_DOMAIN, 0);
  transcript.set(merkleRoot, 28);
  transcript.set(originHash, 60);
  transcript.set(previewHash, 92);
  transcript.set(metaHash, 124);
  transcript.set(vqHash, 156);
  return transcript;
}

/**
 * Generates a fresh Ed25519 keypair in memory using Bun's native WebCrypto implementation.
 */
export async function generateAttestationKeyPair(): Promise<AuthorityAttestationKeys> {
  const kp = (await crypto.subtle.generateKey({ name: "Ed25519" }, true, [
    "sign",
    "verify",
  ])) as unknown as CryptoKeyPair;
  const rawPub = new Uint8Array(await crypto.subtle.exportKey("raw", kp.publicKey));
  const publicKeyRaw = new Uint8Array(new ArrayBuffer(32));
  publicKeyRaw.set(rawPub.subarray(0, 32));

  return {
    privateKey: kp.privateKey,
    publicKey: kp.publicKey,
    publicKeyRaw,
    keyIdBytes: computeKeyId(publicKeyRaw),
  };
}

let cachedAuthorityKeysPromise: Promise<AuthorityAttestationKeys> | null = null;
const importedPublicKeyCache = new Map<string, CryptoKey>();

async function importRawEd25519PublicKey(rawBytes: Uint8Array): Promise<CryptoKey> {
  const hex = rawBytes.toHex();
  const cached = importedPublicKeyCache.get(hex);
  if (cached) return cached;

  const clean = new Uint8Array(new ArrayBuffer(32));
  clean.set(rawBytes.subarray(0, 32));
  const key = await crypto.subtle.importKey("raw", clean, { name: "Ed25519" }, true, ["verify"]);
  importedPublicKeyCache.set(hex, key);
  return key;
}

async function loadOrInitAuthorityKeys(): Promise<AuthorityAttestationKeys> {
  // 1. Environment variable private JWK (`TOKEN_ATTESTATION_PRIVATE_JWK`)
  const envPrivJwk = Bun.env.TOKEN_ATTESTATION_PRIVATE_JWK?.trim();
  if (envPrivJwk) {
    const jwk = JSON.parse(envPrivJwk) as BunJwk;
    const privateKey = await crypto.subtle.importKey("jwk", jwk, { name: "Ed25519" }, true, [
      "sign",
    ]);
    const pubJwk: BunJwk = { kty: jwk.kty, crv: jwk.crv, x: jwk.x, ext: true };
    const publicKey = await crypto.subtle.importKey("jwk", pubJwk, { name: "Ed25519" }, true, [
      "verify",
    ]);
    const rawPub = new Uint8Array(await crypto.subtle.exportKey("raw", publicKey));
    const publicKeyRaw = new Uint8Array(new ArrayBuffer(32));
    publicKeyRaw.set(rawPub.subarray(0, 32));
    importedPublicKeyCache.set(publicKeyRaw.toHex(), publicKey);
    return {
      privateKey,
      publicKey,
      publicKeyRaw,
      keyIdBytes: computeKeyId(publicKeyRaw),
    };
  }

  // 2. Verify-only public key (`TOKEN_ATTESTATION_PUBLIC_HEX`)
  const envPubHex = Bun.env.TOKEN_ATTESTATION_PUBLIC_HEX?.trim();
  if (envPubHex && /^[0-9a-fA-F]{64}$/.test(envPubHex)) {
    const parsed = Uint8Array.fromHex(envPubHex);
    const publicKeyRaw = new Uint8Array(new ArrayBuffer(32));
    publicKeyRaw.set(parsed);
    const publicKey = await importRawEd25519PublicKey(publicKeyRaw);
    return {
      privateKey: null,
      publicKey,
      publicKeyRaw,
      keyIdBytes: computeKeyId(publicKeyRaw),
    };
  }

  // 3. Persisted local authority keypair in gitignored `./data/attestation_ed25519.json`
  const keyFile = Bun.file(DEFAULT_KEYPAIR_PATH);
  if (await keyFile.exists()) {
    try {
      const saved = (await keyFile.json()) as PersistedKeyFile;
      if (saved.privateKeyJwk && /^[0-9a-fA-F]{64}$/.test(saved.publicKeyHex)) {
        const privateKey = await crypto.subtle.importKey(
          "jwk",
          saved.privateKeyJwk,
          { name: "Ed25519" },
          true,
          ["sign"],
        );
        const parsedPub = Uint8Array.fromHex(saved.publicKeyHex);
        const publicKeyRaw = new Uint8Array(new ArrayBuffer(32));
        publicKeyRaw.set(parsedPub);
        const publicKey = await importRawEd25519PublicKey(publicKeyRaw);
        return {
          privateKey,
          publicKey,
          publicKeyRaw,
          keyIdBytes: computeKeyId(publicKeyRaw),
        };
      }
    } catch {
      // Regenerate below if corrupted
    }
  }

  // 4. First-run bootstrap: generate Ed25519 authority keypair and persist to `./data/`
  const generated = await generateAttestationKeyPair();
  const privateKeyJwk = await crypto.subtle.exportKey("jwk", generated.privateKey!);
  const record: PersistedKeyFile = {
    protocol: "ED25519-MERKLE-GF28-v3",
    keyIdHex: generated.keyIdBytes.toHex(),
    publicKeyHex: generated.publicKeyRaw.toHex(),
    privateKeyJwk,
    createdAt: new Date().toISOString(),
  };
  await Bun.write(DEFAULT_KEYPAIR_PATH, JSON.stringify(record, null, 2));
  importedPublicKeyCache.set(generated.publicKeyRaw.toHex(), generated.publicKey);
  return generated;
}

export function getAuthorityAttestationKeys(): Promise<AuthorityAttestationKeys> {
  if (!cachedAuthorityKeysPromise) {
    cachedAuthorityKeysPromise = loadOrInitAuthorityKeys();
  }
  return cachedAuthorityKeysPromise;
}

/**
 * Signs all 4 token sections (Origin Header + VQ Grid + Section II Metadata + Section III Preview)
 * with Ed25519 and packs the 256-byte Section I attestation block + 32-byte Merkle root.
 */
export async function generateSnarkAttestationStub(inputs: AttestationInputs): Promise<{
  snarkStub: Uint8Array<ArrayBuffer>;
  merkleRoot: Uint8Array<ArrayBuffer>;
}> {
  const keys = inputs.signingKeys ?? (await getAuthorityAttestationKeys());
  if (!keys.privateKey) {
    throw new Error(
      "Cannot sign token: authority Ed25519 private key is not available (verify-only mode).",
    );
  }

  const stub = new Uint8Array(new ArrayBuffer(256));
  stub[0] = ED25519_ATTEST_MAGIC[0];
  stub[1] = ED25519_ATTEST_MAGIC[1];
  stub[2] = ED25519_ATTEST_MAGIC[2];
  stub[3] = ED25519_ATTEST_MAGIC[3];

  const originHash = computeOriginCommitment(inputs);
  const previewHash = sha256Domain("TT-PREVIEW-v3:", inputs.previewBytes);
  const metaHash = sha256Domain("TT-META-v3:", inputs.metaBytes);
  const vqHash = sha256Domain("TT-VQ-v3:", inputs.vqIndices);

  stub.set(originHash, 4); // 4..35
  stub.set(previewHash, 36); // 36..67
  stub.set(metaHash, 68); // 68..99
  stub.set(vqHash, 100); // 100..131

  const merkleRoot = computeMerkleRoot4(originHash, previewHash, metaHash, vqHash);

  // 132..163: 32-byte raw Ed25519 signer public key
  stub.set(keys.publicKeyRaw, 132);

  // 164..227: 64-byte Ed25519 signature over canonical 188-byte Merkle transcript
  const transcript = buildSigningTranscript(merkleRoot, originHash, previewHash, metaHash, vqHash);
  const sigBuf = await crypto.subtle.sign({ name: "Ed25519" }, keys.privateKey, transcript);
  const signature64 = new Uint8Array(sigBuf);
  stub.set(signature64.subarray(0, 64), 164);

  // 228..235: 8-byte Key ID
  stub.set(keys.keyIdBytes, 228);

  // 236..255: 20-byte GF(2^8) polynomial audit tags over signature[0..63]
  for (let k = 0; k < 20; k++) {
    const alpha = (merkleRoot[k] ?? 1) || 1;
    stub[236 + k] = gfPolyEval(signature64, alpha);
  }

  return { snarkStub: stub, merkleRoot };
}

/**
 * Cryptographically verifies:
 * 1. Exact SHA-256 leaf commitments recomputed from the live Origin Header, VQ Grid,
 *    Section II `metaBytes`, and Section III `previewBytes` (detecting any tampering or bit-rot).
 * 2. Domain-separated 4-leaf SHA-256 Merkle root equality.
 * 3. Signer Ed25519 public key match against the pinned authority public key (rejecting rogue keys).
 * 4. Ed25519 detached signature validity via BoringSSL WebCrypto (`crypto.subtle.verify`).
 */
export async function verifySnarkAttestationStub(
  inputs: VerifyAttestationInputs,
): Promise<SnarkAttestationInfo> {
  const t0Ns = Bun.nanoseconds();
  const { snarkStub, merkleRoot } = inputs;

  if (
    snarkStub.length !== 256 ||
    snarkStub[0] !== ED25519_ATTEST_MAGIC[0] ||
    snarkStub[1] !== ED25519_ATTEST_MAGIC[1] ||
    snarkStub[2] !== ED25519_ATTEST_MAGIC[2] ||
    snarkStub[3] !== ED25519_ATTEST_MAGIC[3]
  ) {
    return {
      protocol: "ED25519-MERKLE-GF28-v3",
      valid: false,
      signatureVerified: false,
      trustedKeyMatch: false,
      sectionsVerified: {
        originHeader: false,
        vqGrid: false,
        section2Metadata: false,
        section3Preview: false,
        merkleRoot: false,
      },
      signerKeyIdHex: "",
      signerPublicKeyHex: "",
      circuitCommitmentHex: "",
      assetOriginHashHex: "",
      previewHashHex: "",
      ocrWitnessHashHex: "",
      vqCommitmentHex: "",
      gfPolynomialEval: 0,
      verifiedInUs: Math.max(1, Math.round((Bun.nanoseconds() - t0Ns) / 1000)),
      failureReason: "invalid_or_legacy_unsigned_attestation_magic",
    };
  }

  const stubOriginHash = snarkStub.subarray(4, 36);
  const stubPreviewHash = snarkStub.subarray(36, 68);
  const stubMetaHash = snarkStub.subarray(68, 100);
  const stubVqHash = snarkStub.subarray(100, 132);
  const stubPublicKeyRaw = snarkStub.subarray(132, 164);
  const stubSignature = snarkStub.subarray(164, 228);
  const stubKeyId = snarkStub.subarray(228, 236);
  const stubPolyTags = snarkStub.subarray(236, 256);

  // 1. Re-hash every actual section of the decoded token buffer
  const actualOriginHash = computeOriginCommitment(inputs);
  const actualPreviewHash = sha256Domain("TT-PREVIEW-v3:", inputs.previewBytes);
  const actualMetaHash = sha256Domain("TT-META-v3:", inputs.metaBytes);
  const actualVqHash = sha256Domain("TT-VQ-v3:", inputs.vqIndices);

  const originHeaderOk = constantTimeBytesEqual(actualOriginHash, stubOriginHash);
  const section3PreviewOk =
    inputs.envelopeBoundsValid !== false &&
    constantTimeBytesEqual(actualPreviewHash, stubPreviewHash);
  const section2MetadataOk =
    inputs.envelopeBoundsValid !== false &&
    constantTimeBytesEqual(actualMetaHash, stubMetaHash);
  const vqGridOk = constantTimeBytesEqual(actualVqHash, stubVqHash);

  // 2. Recompute 4-leaf Merkle root from the ACTUAL recomputed section hashes
  const actualMerkleRoot = computeMerkleRoot4(
    actualOriginHash,
    actualPreviewHash,
    actualMetaHash,
    actualVqHash,
  );
  const merkleRootOk = constantTimeBytesEqual(actualMerkleRoot, merkleRoot);

  // 3. Verify signer public key matches pinned authority public key (prevents rogue key forgery)
  const expectedPubRaw =
    inputs.trustedPublicKeyRaw ?? (await getAuthorityAttestationKeys()).publicKeyRaw;
  const trustedKeyMatch = constantTimeBytesEqual(stubPublicKeyRaw, expectedPubRaw);
  const expectedKeyId = computeKeyId(stubPublicKeyRaw);
  const keyIdOk = constantTimeBytesEqual(stubKeyId, expectedKeyId);

  // 4. Verify GF(2^8) polynomial audit tags over signature
  const alpha0 = (merkleRoot[0] ?? 1) || 1;
  const eval0 = gfPolyEval(stubSignature, alpha0);
  let polyOk = true;
  for (let k = 0; k < 20; k++) {
    const alpha = (merkleRoot[k] ?? 1) || 1;
    if (gfPolyEval(stubSignature, alpha) !== (stubPolyTags[k] ?? 0)) {
      polyOk = false;
      break;
    }
  }

  // 5. Verify Ed25519 cryptographic signature against the pinned authority public key
  let signatureVerified = false;
  if (trustedKeyMatch) {
    try {
      const verifyKey = await importRawEd25519PublicKey(expectedPubRaw);
      const sigCopy = new Uint8Array(new ArrayBuffer(64));
      sigCopy.set(stubSignature);
      const transcript = buildSigningTranscript(
        merkleRoot,
        stubOriginHash,
        stubPreviewHash,
        stubMetaHash,
        stubVqHash,
      );
      signatureVerified = await crypto.subtle.verify(
        { name: "Ed25519" },
        verifyKey,
        sigCopy,
        transcript,
      );
    } catch {
      signatureVerified = false;
    }
  }

  const allValid =
    originHeaderOk &&
    section3PreviewOk &&
    section2MetadataOk &&
    vqGridOk &&
    merkleRootOk &&
    trustedKeyMatch &&
    keyIdOk &&
    polyOk &&
    signatureVerified;

  let failureReason: string | undefined;
  if (!allValid) {
    if (!trustedKeyMatch) failureReason = "untrusted_signer_public_key";
    else if (!signatureVerified) failureReason = "ed25519_signature_invalid";
    else if (inputs.envelopeBoundsValid === false) failureReason = "token_envelope_length_mismatch";
    else if (!originHeaderOk) failureReason = "origin_header_commitment_mismatch";
    else if (!vqGridOk) failureReason = "vq_grid_commitment_mismatch";
    else if (!section2MetadataOk) failureReason = "section2_metadata_hash_mismatch";
    else if (!section3PreviewOk) failureReason = "section3_preview_hash_mismatch";
    else if (!merkleRootOk) failureReason = "merkle_root_mismatch";
    else if (!keyIdOk) failureReason = "signer_key_id_mismatch";
    else failureReason = "gf28_audit_tag_mismatch";
  }

  const verifiedInUs = Math.max(1, Math.round((Bun.nanoseconds() - t0Ns) / 1000));

  return {
    protocol: "ED25519-MERKLE-GF28-v3",
    valid: allValid,
    signatureVerified,
    trustedKeyMatch,
    sectionsVerified: {
      originHeader: originHeaderOk,
      vqGrid: vqGridOk,
      section2Metadata: section2MetadataOk,
      section3Preview: section3PreviewOk,
      merkleRoot: merkleRootOk,
    },
    signerKeyIdHex: bytesToHex(stubKeyId),
    signerPublicKeyHex: bytesToHex(stubPublicKeyRaw),
    circuitCommitmentHex: bytesToHex(stubSignature.subarray(0, 16)),
    assetOriginHashHex: bytesToHex(stubOriginHash.subarray(0, 16)),
    previewHashHex: bytesToHex(stubPreviewHash.subarray(0, 16)),
    ocrWitnessHashHex: bytesToHex(stubMetaHash.subarray(0, 16)),
    vqCommitmentHex: bytesToHex(stubVqHash.subarray(0, 16)),
    gfPolynomialEval: eval0,
    verifiedInUs,
    ...(failureReason ? { failureReason } : {}),
  };
}
