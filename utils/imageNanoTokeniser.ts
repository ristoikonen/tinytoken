/**
 * =================================================================================================
 * NANOTOKEN SPECIFICATION V1: 512-BYTE SECTOR-ALIGNED SELF-HEALING SEMANTIC VECTOR CONTAINER
 * Module: utils/imageNanoTokeniser.ts
 * =================================================================================================
 *
 * ARCHITECTURAL OVERVIEW:
 * NanoToken is an ultra-dense, 512-byte (exactly 1 physical disk sector / 2^9 hardware aligned)
 * binary container representing high-resolution images (PNG, JPEG, WebP, AVIF, SVG, PDF) in
 * sub-kilobyte memory. It solves four critical architectural challenges:
 *
 *  1. Semantic Vector: 384-dimensional DINOv3 ViT-S/16 Int8 normalized embedding.
 *  2. Payload Self-Healing: Dual-lane GF(2^8) Reed-Solomon parity covering the ENTIRE 448-byte
 *     data payload (Header Metadata + Target Pointer + DINO Embedding). Note: The trailing 32-byte
 *     authentication tag sits outside RS coverage; corruption of the tag causes safe rejection.
 *  3. Cryptographic Tamper Defense: 32-byte HMAC-SHA256 authenticator (Candidate 1) or
 *     64-byte domain-separated Ed25519 signature (Alternative).
 *  4. Canonical Cryptographic Pointer: Fixed 32-byte ContentId (SHA-256 Merkle Root) pointing
 *     unambiguously to the parent TinyToken (or null-terminated short filename).
 *
 * -------------------------------------------------------------------------------------------------
 * CRITICAL ARCHITECTURAL CORRECTIONS APPLIED:
 * -------------------------------------------------------------------------------------------------
 *  [FIX 1: Disjoint Checksum Boundaries & Acyclic Generation Dependency]
 *    Checksum fields NEVER cover their own offsets or downstream parity bytes.
 *    - headerCrc32 covers strictly 0x000..0x017 (Metadata only).
 *    - payloadCrc32 covers strictly 0x020..0x1BF (Pointer + Embedding only, EXCLUDING RS parity).
 *    Crucially, because payloadCrc32 is stored in the header (which lies inside RS Lane 0 data),
 *    excluding RS parity from payloadCrc32 breaks the circular deadlock:
 *    (RS Parity -> payloadCrc32 -> RS Parity) and guarantees a strictly linear generation order.
 *
 *  [FIX 2: Full-Payload Reed-Solomon Protection (Header + Pointer + Embedding)]
 *    RS parity protects the ENTIRE 448-byte payload (Header + Pointer + Embedding).
 *    Recovery scope is formally PAYLOAD self-healing (0x000..0x1BF), as the trailing 32-byte
 *    authentication tag sits outside RS coverage to avoid circular signature dependencies.
 *    Split into two 224-byte lanes:
 *      - Lane 0 (224 B): Header Metadata (32B) + Pointer (32B) + Embedding Half 0 (160B)
 *      - Lane 1 (224 B): Embedding Half 1 (224B)
 *    If wire noise corrupts bytes in the header, pointer, or embedding, RS parity HEALS THEM.
 *
 *  [FIX 3: Accurate Reed-Solomon Error Correction Capability]
 *    With nsym = 16 parity bytes per lane in GF(2^8), minimum code distance is d = 17.
 *    Theoretical correction capability is t = floor(nsym / 2) = UP TO 8 BYTE ERRORS PER LANE
 *    (up to 16 byte errors total across both lanes) using Berlekamp-Massey / Forney decoding,
 *    or up to 16 erasures per lane when error locations are known.
 *
 *  [FIX 4: Correct Verification Sequence (Heal Before Authenticate)]
 *    Authentication (HMAC or Ed25519) commits to the PRISTINE transmitter bytes.
 *    Evaluating HMAC on corrupted wire bytes will always fail.
 *    Execution order MUST be:
 *      1. Receive wire buffer (512 B).
 *      2. Compute Reed-Solomon syndromes across Lane 0 and Lane 1.
 *      3. If syndromes != 0, execute RS error locator & evaluator to HEAL the bytes in RAM.
 *      4. Verify HMAC-SHA256 (or Ed25519) on the HEALED pristine bytes.
 *      5. If signature passes, the token is 100% authentic and verified.
 *
 *  [FIX 5: Immutable On-Wire Flags (Zero State Mutation)]
 *    Authenticated bytes must remain strictly immutable. The "RS Healed" status is NEVER written
 *    into the on-wire header flags (which would invalidate the issuer's HMAC).
 *    "healed: boolean" and "correctedBytes: number" are strictly ephemeral runtime return values.
 *
 *  [FIX 6: 32-Byte ContentId Pointer (Eliminates Filename Truncation)]
 *    Filenames like "tok_00042_satellite_canberra_fire_sector_b_20261117.jpg.token" exceed 32 bytes.
 *    NanoToken standardizes on the parent TinyToken's 32-byte SHA-256 Merkle Root (ContentId / CID)
 *    as the canonical pointer. It is always exactly 32 bytes, cryptographically unforgeable,
 *    and resolves in O(1) via zipEngine.getByContentId() or CAS/S3.
 *
 *  [FIX 7: Empirical Benchmarking Policy]
 *    Latency and throughput figures are to be measured empirically via Bun.nanoseconds()
 *    benchmarks rather than stated as static hardcoded assumptions.
 *
 *  [FIX 8: Ed25519 Domain Prefix Length (Exact 18 Bytes)]
 *    The domain prefix string is strictly "TT-NANO-ED25519-V1" (18 ASCII characters/bytes, zero
 *    null terminator). The signed input buffer is: 18B prefix + 448B payload = 466 Bytes Total.
 *    The domain prefix is ephemeral to the signer/verifier transcript and is NOT stored on the wire.
 *
 *  [FIX 9: Complete Parent Independence & Two-Phase CID-128 Resolution]
 *    NanoToken does NOT depend on the parent token for verification, RS healing, or DINO search.
 *    The pointer is an opaque routing handle. When resolving CID-128, resolvers must guard against
 *    16-byte prefix ambiguity and verify the parent's full 32-byte Merkle root upon hydration.
 *
 * -------------------------------------------------------------------------------------------------
 * SELECTED PRIMARY ARCHITECTURE: 512-BYTE HMAC-SHA256 CONTAINER (CANDIDATE 1)
 * -------------------------------------------------------------------------------------------------
 *
 * Container Size: Exactly 512 Bytes (0x000 .. 0x1FF)
 * Magic Signature: "NANO" (0x4E 0x41 0x4E 0x4F)
 *
 * 0                   32                  64                                448                  480                  512 Bytes
 * ┌───────────────────┬───────────────────┬──────────────────────────────────┬───────────────────┬────────────────────┐
 * │ Fixed Header Meta │ Target Pointer    │  DINOv3 Semantic Embedding       │  RS Parity Bytes  │  Cryptographic     │
 * │    (32 Bytes)     │ (Parent Merkle /  │      (384 Bytes Int8)            │  (32 Bytes GF28)  │  HMAC-SHA256 Tag   │
 * │                   │  Filename, 32 B)  │                                  │                   │  (32 Bytes)        │
 * └───────────────────┴───────────────────┴──────────────────────────────────┴───────────────────┴────────────────────┘
 * │◄────────────────────────────── PROTECTED BY RS PARITY (448 B) ─────────────────────────────►│
 * │◄────────────────────────────── SIGNED BY HMAC-SHA256 (480 B) ──────────────────────────────►│
 *
 * EXACT BYTE OFFSET MAP (512 BYTES TOTAL):
 * -------------------------------------------------------------------------------------------------
 * Offset (Hex)     Size     Field Name          Type        Description / Purpose
 * -------------------------------------------------------------------------------------------------
 * 0x000 .. 0x003   4 B      magic               ASCII[4]    "NANO" (0x4E414E4F)
 * 0x004 .. 0x005   2 B      version             Uint16LE    0x0001 (NanoToken Specification V1.0)
 * 0x006 .. 0x007   2 B      modelId             Uint16LE    0x0002 = DINOv3 ViT-S/16 Q8; 0x0001 = Fallback
 * 0x008 .. 0x009   2 B      origWidth           Uint16LE    Original image width in pixels
 * 0x00A .. 0x00B   2 B      origHeight          Uint16LE    Original image height in pixels
 * 0x00C .. 0x00D   2 B      embeddingBytes      Uint16LE    0x0180 (384 dimensions)
 * 0x00E .. 0x00F   2 B      parityBytes         Uint16LE    0x0020 (32 bytes Reed-Solomon parity)
 * 0x010            1 B      sourceFormat        Uint8       1=PNG, 2=JPEG, 3=WebP, 4=AVIF, 5=SVG, 6=PDF
 * 0x011            1 B      pointerType         Uint8       0=Canonical 32B Merkle CID, 1=Short ASCII Name
 * 0x012 .. 0x013   2 B      flags               Uint16LE    Bit 0: Attested; Bits 1-15: Reserved (Immutable)
 * 0x014 .. 0x017   4 B      utcTimestamp        Uint32LE    Unix epoch creation seconds
 * 0x018 .. 0x01B   4 B      headerCrc32         Uint32LE    CRC-32 over bytes 0x000..0x017 (EXCLUDES self and payload)
 * 0x01C .. 0x01F   4 B      payloadCrc32        Uint32LE    CRC-32 over bytes 0x020..0x1BF (EXCLUDES self and RS parity)
 *
 * 0x020 .. 0x03F   32 B     targetPointer       Raw[32]     CANONICAL TARGET POINTER:
 *                                                           - Primary (pointerType=0): 32-byte SHA-256 Merkle Root
 *                                                             (ContentId) of parent TinyToken. Immune to filename size.
 *                                                           - Fallback (pointerType=1): Null-terminated ASCII filename
 *                                                             (valid only if <= 31 chars + null terminator).
 *
 * 0x040 .. 0x1BF   384 B    dinoEmbedding       Int8[384]   384-dim normalized DINOv3 ViT-S/16 feature vector
 *
 * -------------------------------------------------------------------------------------------------
 * REED-SOLOMON DUAL-LANE ASSIGNMENT (448 DATA BYTES -> 2 x 224 B):
 * -------------------------------------------------------------------------------------------------
 * Lane 0 Data:    Offsets 0x000 .. 0x0DF (224 Bytes):
 *                 Contains Header (32B) + Pointer (32B) + DINO Dimensions 0..159 (160B).
 * Lane 0 Parity:  Offsets 0x1C0 .. 0x1CF (16 Bytes, nsym=16 in GF(2^8)).
 *                 -> Protects and heals Header, Pointer, and First Half of Embedding!
 *
 * Lane 1 Data:    Offsets 0x0E0 .. 0x1BF (224 Bytes):
 *                 Contains DINO Dimensions 160..383 (224B).
 * Lane 1 Parity:  Offsets 0x1D0 .. 0x1DF (16 Bytes, nsym=16 in GF(2^8)).
 *                 -> Protects and heals Second Half of Embedding!
 *
 * -------------------------------------------------------------------------------------------------
 * CRYPTOGRAPHIC AUTHENTICATION TAG:
 * -------------------------------------------------------------------------------------------------
 * 0x1E0 .. 0x1FF   32 B     hmacSha256          Uint8[32]   HMAC-SHA256(AuthoritySecret, Bytes 0x000 .. 0x1DF).
 *                                                           Covers the entire 480-byte healed container.
 *                                                           Evaluated after RS healing restores pristine bytes.
 *
 * -------------------------------------------------------------------------------------------------
 * CANONICAL GENERATION SEQUENCE (STRICT FORWARD DEPENDENCY ORDER):
 * -------------------------------------------------------------------------------------------------
 *  Step 1: Populate Header Metadata (0x000..0x017), Target Pointer (0x020..0x03F), and DINO (0x040..0x1BF).
 *  Step 2: Calculate headerCrc32 = CRC32(Bytes 0x000..0x017); write to 0x018..0x01B.
 *  Step 3: Calculate payloadCrc32 = CRC32(Bytes 0x020..0x1BF); write to 0x01C..0x01F.
 *          (Both CRCs are now frozen in Lane 0 data before RS parity generation).
 *  Step 4: Calculate Reed-Solomon parity for both lanes:
 *          - Lane 0 Parity = RSEncode(Lane 0 Data: 0x000..0x0DF, nsym=16); write to 0x1C0..0x1CF.
 *          - Lane 1 Parity = RSEncode(Lane 1 Data: 0x0E0..0x1BF, nsym=16); write to 0x1D0..0x1DF.
 *  Step 5: Calculate HMAC-SHA256(AuthoritySecret, Bytes 0x000..0x1DF); write to 0x1E0..0x1FF.
 *          (Every stage depends only on previously frozen bytes; zero circular dependencies).
 *
 * -------------------------------------------------------------------------------------------------
 * INGESTION & HEALING VERIFICATION PIPELINE (POST-HEAL VALIDATION):
 * -------------------------------------------------------------------------------------------------
 *  CRITICAL INVARIANT (ISSUE A):
 *  Because Reed-Solomon Lane 0 protects the entire 0x000..0x0DF block—including the magic bytes
 *  at 0x000..0x003—early rejection on unhealed magic would defeat recovery of single-byte wire
 *  flips (e.g. "MANO" instead of "NANO"). Therefore, RS extraction and syndrome correction MUST
 *  precede all magic, version, header, and signature validations.
 *
 *  Step 1: Validate exact buffer length === 512 bytes (reject under/oversized records immediately).
 *  Step 2: Extract RS lanes using fixed offsets:
 *          - Lane 0 Data: 0x000..0x0DF (224B) with Lane 0 Parity: 0x1C0..0x1CF (16B).
 *          - Lane 1 Data: 0x0E0..0x1BF (224B) with Lane 1 Parity: 0x1D0..0x1DF (16B).
 *  Step 3: Check RS syndromes across both lanes:
 *          - If syndromes == 0: Record is pristine on the wire.
 *          - If syndromes != 0: Solve error locator/evaluator polynomials. Repair corrupted bytes
 *            (up to 8 per lane) directly in RAM. Record healed = true, errorsCorrected = count.
 *  Step 4: Validate healed magic ("NANO") and version (0x0001). Reject if invalid after healing.
 *  Step 5: Verify cryptographic authenticator on healed bytes:
 *          - Primary: HMAC-SHA256(Secret, Bytes 0x000..0x1DF) === Bytes 0x1E0..0x1FF.
 *          - Alternative: Ed25519Verify(PublicKey, Bytes 0x000..0x1BF, Signature 0x1C0..0x1FF).
 *          Reject if signature fails (tampered or unrecoverable error).
 *  Step 6: Validate CRC fields:
 *          - headerCrc32 over 0x000..0x017 (confirms healed header integrity).
 *          - payloadCrc32 over 0x020..0x1BF (confirms healed payload integrity).
 *  Step 7: Validate modelId, flags, dimensions, and targetPointer syntax/path-traversal safety.
 *  Step 8: Return authenticated record { valid: true, healed, errorsCorrected, embedding, pointer }.
 *
 * =================================================================================================
 * ALTERNATIVE ARCHITECTURE: 512-BYTE ASYMMETRIC ED25519 SIGNATURE CONTAINER
 * =================================================================================================
 *
 * CONTEXT:
 * Required when verification occurs in zero-trust environments (public browsers, third-party nodes,
 * external API clients) where a shared symmetric HMAC secret key cannot be safely distributed.
 *
 * To accommodate 64 bytes of Ed25519 signature within the strict 512-byte sector budget:
 *  - Header Metadata is compacted to 16 bytes.
 *  - Target Pointer is compacted to 16 bytes (UUIDv7 or 128-bit truncated Merkle CID).
 *  - DINO Embedding remains 384 bytes.
 *  - Reed-Solomon parity remains 32 bytes (protecting all 416 bytes of data).
 *  - Ed25519 Signature occupies the final 64 bytes.
 *
 * 0                   16                  32                                416                  448                  512 Bytes
 * ┌───────────────────┬───────────────────┬──────────────────────────────────┬───────────────────┬────────────────────┐
 * │ Compact Header    │ Compact Pointer   │  DINOv3 Semantic Embedding       │  RS Parity Bytes  │  Full Asymmetric   │
 * │    (16 Bytes)     │ (16-Byte UUIDv7 / │      (384 Bytes Int8)            │  (32 Bytes GF28)  │  Ed25519 Signature │
 * │                   │  CID-128, 16 B)   │                                  │                   │  (64 Bytes)        │
 * └───────────────────┴───────────────────┴──────────────────────────────────┴───────────────────┴────────────────────┘
 * │◄────────────────────────────── PROTECTED BY RS PARITY (416 B) ─────────────────────────────►│
 * │◄────────────────────────────── SIGNED BY ED25519 (448 B) ──────────────────────────────────►│
 *
 * EXACT BYTE OFFSET MAP (ED25519 ALTERNATIVE):
 * -------------------------------------------------------------------------------------------------
 * Offset (Hex)     Size     Field Name          Type        Description / Purpose
 * -------------------------------------------------------------------------------------------------
 * 0x000 .. 0x003   4 B      magic               ASCII[4]    "NAED" (NanoToken Asymmetric Ed25519)
 * 0x004 .. 0x005   2 B      version             Uint16LE    0x0001
 * 0x006 .. 0x007   2 B      modelId             Uint16LE    0x0002 (DINOv3 ViT-S/16 Q8)
 * 0x008 .. 0x009   2 B      origWidth           Uint16LE    Original asset width
 * 0x00A .. 0x00B   2 B      origHeight          Uint16LE    Original asset height
 * 0x00C            1 B      sourceFormat        Uint8       1=PNG, 2=JPEG, 3=WebP, 4=AVIF, 5=SVG, 6=PDF
 * 0x00D            1 B      flags               Uint8       BITFIELD FLAGS (Immutable once signed):
 *                                                           - Bit 0 (0x01): FLAG_ATTESTED (1 = Authority signed)
 *                                                           - Bit 1 (0x02): FLAG_POINTER_MODE:
 *                                                               0 = Mode 0: 16-byte UUIDv7 (S3 key)
 *                                                               1 = Mode 1: 16-byte CID-128 (Merkle prefix)
 *                                                           - Bits 2-7: Reserved (Must be 0)
 * 0x00E .. 0x00F   2 B      headerCrc16         Uint16LE    CRC-16 over bytes 0x000..0x00D (EXCLUDES self)
 *
 * 0x010 .. 0x01F   16 B     compactPointer      Raw[16]     COMPACT PARENT TARGET POINTER:
 *                                                           Selected via flags Bit 1 (FLAG_POINTER_MODE):
 *                                                           - Mode 0 (flags & 0x02 === 0): 128-bit RFC 9562 UUIDv7
 *                                                             (48-bit unix ms timestamp + 74 bits entropy). Maps
 *                                                             deterministically to S3: `s3://bucket/{uuidHex}.token`.
 *                                                           - Mode 1 (flags & 0x02 === 2): 128-bit CID-128 (First 16
 *                                                             bytes of parent TinyToken's 32-byte SHA-256 Merkle root).
 *                                                             Enables O(1) prefix lookup via zipEngine.getByPrefix16().
 *
 * 0x020 .. 0x19F   384 B    dinoEmbedding       Int8[384]   384-dim normalized DINOv3 feature vector
 *
 * 0x1A0 .. 0x1BF   32 B     rsParity            Uint8[32]   REED-SOLOMON PARITY (Protects 416 data bytes):
 *                                                           - Lane 0 Data: 0x000..0x0CF (208 B: Header+Ptr+DINO_A)
 *                                                           - Lane 0 Parity: 0x1A0..0x1AF (16 B, nsym=16)
 *                                                           - Lane 1 Data: 0x0D0..0x19F (208 B: DINO_B)
 *                                                           - Lane 1 Parity: 0x1B0..0x1BF (16 B, nsym=16)
 *
 * 0x1C0 .. 0x1FF   64 B     ed25519Signature    Uint8[64]   ASYMMETRIC ED25519 SIGNATURE:
 *                                                           Ed25519Sign(PrivateKey, DomainPrefix || Bytes 0x000..0x1BF).
 *                                                           Publicly verifiable with 32-byte Ed25519 Public Key.
 *
 * -------------------------------------------------------------------------------------------------
 * ED25519 CRYPTOGRAPHIC BINDING & DOMAIN SEPARATION SPECIFICATION:
 * -------------------------------------------------------------------------------------------------
 *  1. Domain Separation Prefix (Exact 18 Bytes):
 *     To prevent signature cross-protocol replay attacks (e.g. replaying a NanoToken signature
 *     against a full TinyToken Section I attestation), the signed message MUST prepend:
 *       DomainPrefix = "TT-NANO-ED25519-V1" (Exact 18 ASCII Bytes, zero null terminator):
 *       Hex: [0x54, 0x54, 0x2D, 0x4E, 0x41, 0x4E, 0x4F, 0x2D, 0x45, 0x44, 0x32, 0x35, 0x35, 0x31, 0x39, 0x2D, 0x56, 0x31]
 *     Signed Message = DomainPrefix (18 B) || Healed Payload (448 B: 0x000..0x1BF) = 466 Bytes Total.
 *     (Note: Stored Ed25519 signature is 64 B at 0x1C0..0x1FF; DomainPrefix is NOT stored on wire).
 *
 *  2. Strict Model Identity Binding:
 *     modelId (0x006..0x007, e.g. 0x0002 for DINOv3 ViT-S/16 Q8) is explicitly covered under
 *     the Ed25519 signature. The verifier MUST confirm that modelId matches the local vector
 *     search index configuration before performing cosine similarity. This prevents cross-model
 *     vector substitution attacks (e.g. signing a text BGE vector and masquerading as visual DINO).
 *
 *  3. Pointer Mode Selection & Two-Phase CID-128 Resolution:
 *     The parser inspects flags (0x00D) bit 1 (FLAG_POINTER_CID128) to determine semantics:
 *     - If (flags & 0x02) === 0 (Mode 0: UUIDv7):
 *       Resolver converts bytes 0x010..0x01F to canonical hex UUID: `s3://bucket/${uuidHex}.token`
 *       or invokes HTTP endpoint `GET /archives/tokens/${uuidHex}`.
 *     - If (flags & 0x02) === 2 (Mode 1: CID-128, Default):
 *       Two-Phase Resolution Protocol:
 *         Phase 1 (Routing Hint): Find entries in active archive index matching prefix16.
 *           -> Ambiguity Guard: If > 1 entry matches prefix16, ABORT resolution with
 *              AMBIGUOUS_PREFIX error (prevents accidental or crafted prefix collision).
 *         Phase 2 (Full Merkle Attestation): Once candidate 25 KB parent token is loaded,
 *           calculate its full 32-byte Merkle root (ContentId) and verify that
 *           parent.merkleRoot.subarray(0, 16) strictly matches the NanoToken pointer AND
 *           matches the authoritative archive catalog. A 16-byte prefix alone is NEVER
 *           treated as a complete identity check.
 *
 * =================================================================================================
 * LOW-LEVEL FFI, ZIG PARSING & MEMORY ALIGNMENT DIRECTIVES
 * =================================================================================================
 *
 *  1. Explicit Little-Endian Views (Zig / Native Parsers):
 *     NanoToken strictly encodes multi-byte integers as Little-Endian (LE).
 *     Parsers in Zig or C must NEVER cast unaligned byte pointers directly to struct pointers
 *     (e.g. `*const u16` or `*const u32`), as this causes undefined behavior or hardware SIGBUS
 *     exceptions on strict-alignment architectures (ARM64, Apple Silicon).
 *     All multi-byte reads MUST use explicit endian-safe accessors:
 *       - Version (0x004..0x005):     std.mem.readInt(u16, slice[0x004..0x006], .little);
 *       - ModelID (0x006..0x007):     std.mem.readInt(u16, slice[0x006..0x008], .little);
 *       - Header CRC (0x018..0x01B):  std.mem.readInt(u32, slice[0x018..0x01C], .little);
 *       - Payload CRC (0x01C..0x01F): std.mem.readInt(u32, slice[0x01C..0x020], .little);
 *     In TypeScript / Bun, use DataView: `view.getUint16(offset, true)` / `view.getUint32(offset, true)`.
 *
 *  2. Bun FFI & Bun.mmap() Platform Alignment:
 *     - Sector-Aligned Strides: Because NanoToken is exactly 512 bytes (2^9), sequential tokens in a
 *       stream or mmap'd archive file (`tokens.nano`) land at offsets: N * 512 bytes.
 *     - Cache-Line & SIMD Alignment:
 *       Every 512-byte boundary is naturally divisible by 64, 32, 16, 8, 4, 2, and 1.
 *       - In HMAC Layout: DINO embedding begins at offset 0x040 (64 decimal), which is 64-byte aligned!
 *       - In Ed25519 Layout: DINO embedding begins at offset 0x020 (32 decimal), which is 32-byte aligned!
 *       This enables direct, zero-copy AVX-512 (64-byte) and ARM NEON (16/32-byte) vector dot-products
 *       directly over memory-mapped pages without scalar alignment penalties.
 *     - TypedArray Subarray Directives:
 *       When slicing typed arrays via `new Int8Array(mmapBuffer, tokenOffset + DINO_OFFSET, 384)`,
 *       the caller must ensure that `tokenOffset % 64 === 0` so the backing ArrayBuffer pointer satisfies
 *       SIMD hardware alignment contracts.
 *
 * =================================================================================================
 * PARENT ASSET INDEPENDENCE & TWO-TIER OPERATIONAL LIFECYCLE
 * =================================================================================================
 *
 *  1. Zero Dependency on Parent for Core Functions:
 *     NanoToken is 100% self-contained. The parent TinyToken (12-25 KB) DOES NOT need to be
 *     online, present on disk, or reachable on the network for:
 *       - Reed-Solomon payload self-healing (runs over NanoToken's own 32B parity).
 *       - Cryptographic verification (HMAC or Ed25519 verifies NanoToken's own bytes).
 *       - Vector similarity search (DINO 384-dim Int8 cosine distance runs directly in RAM).
 *       - Unsupervised clustering, K-Means, and nearest-neighbor classification.
 *     The pointer (CID-128 or UUIDv7) is strictly an opaque routing handle (like an ISBN on a
 *     library card). The library card can be authenticated and searched even if the book is absent.
 *
 *  2. Graceful Degradation when Parent Asset is Missing:
 *     When a search query yields a top NanoToken hit, asset hydration is optional:
 *       - If Parent is Present: UI displays full 320x320 WebP thumbnail and OCR text.
 *       - If Parent is Absent (Cold storage, detached drive, edge drone): Resolver returns
 *         parentStatus = "unavailable" | "cold_storage". Search rankings, match scores (e.g. 0.94),
 *         original dimensions, format, and timestamp remain 100% valid and operational.
 *
 *  3. Operational Use-Cases Where Parent Intentionally Does Not Exist:
 *     - Tactical Edge Drones / Satellites: Drone takes 10 MB images, generates 512B NanoTokens,
 *       and streams ONLY NanoTokens over satellite to ground control. Raw 10 MB images stay on drone.
 *     - Privacy-Preserving Cloud Search: Sensitive medical/surveillance imagery is indexed on-prem.
 *       Only 512B NanoTokens are uploaded to cloud search clusters. The cloud searches and ranks
 *       embeddings without ever having access to the underlying visual pixels.
 * =================================================================================================
 */
