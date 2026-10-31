/**
 * QuickLZ Level 1 Fast Compression / Decompression for Section II Metadata & OCR Text
 * Pure TypeScript / Bun implementation with collision-free tag bits and safe bounds.
 */

export class QuickLZ {
  public static readonly DEFAULT_HEADER_LEN = 9;
  public static readonly HASH_VALUES = 4096;
  public static readonly DEFAULT_MAX_DECOMPRESSED_SIZE = 64 * 1024; // 64 KB safety limit

  public static compress(source: Uint8Array): Uint8Array<ArrayBuffer> {
    const srcLen = source.length;
    if (srcLen === 0) return new Uint8Array(new ArrayBuffer(0));

    const destination = new Uint8Array(srcLen + 400);
    const hash = new Int32Array(this.HASH_VALUES).fill(-1);

    destination[0] = 3; // Level 1 + Compressed flag
    this.writeUInt32LE(destination, 1, srcLen);

    let controlWordIdx = this.DEFAULT_HEADER_LEN;
    let dstIdx = controlWordIdx + 2;
    let srcIdx = 0;
    let controlWord = 0;
    let controlBitsCount = 0;

    while (srcIdx < srcLen - 4) {
      if (controlBitsCount === 16) {
        this.writeUInt16LE(destination, controlWordIdx, controlWord);
        controlWordIdx = dstIdx;
        dstIdx += 2;
        controlWord = 0;
        controlBitsCount = 0;
      }

      const fetch = this.readUInt32LE(source, srcIdx) & 0xffffff;
      const hashVal = ((fetch >> 12) ^ fetch) & (this.HASH_VALUES - 1);
      const o = hash[hashVal] ?? -1;
      hash[hashVal] = srcIdx;

      if (
        o !== -1 &&
        srcIdx - o < 2048 &&
        srcIdx - o > 1 &&
        source[o] === source[srcIdx] &&
        source[o + 1] === source[srcIdx + 1] &&
        source[o + 2] === source[srcIdx + 2]
      ) {
        let matchLen = 3;
        while (
          srcIdx + matchLen < srcLen &&
          source[o + matchLen] === source[srcIdx + matchLen] &&
          matchLen < 255
        ) {
          matchLen++;
        }

        const offset = srcIdx - o; // 2..2047 (11 bits: high 3 bits in b0, low 8 bits in b1)
        controlWord = (controlWord >> 1) | 0x8000;
        controlBitsCount++;

        if (matchLen < 16) {
          // 2-byte match token: bit 0 = 1, bits 1..3 = offset >> 8, bits 4..7 = matchLen (3..15)
          destination[dstIdx] = (matchLen << 4) | (((offset >> 8) & 0x07) << 1) | 1;
          destination[dstIdx + 1] = offset & 0xff;
          dstIdx += 2;
        } else {
          // 3-byte match token: bit 0 = 0, bits 1..3 = offset >> 8, b1 = offset & 0xff, b2 = matchLen
          destination[dstIdx] = (((offset >> 8) & 0x07) << 1) | 0;
          destination[dstIdx + 1] = offset & 0xff;
          destination[dstIdx + 2] = matchLen;
          dstIdx += 3;
        }
        srcIdx += matchLen;
      } else {
        controlWord >>= 1;
        controlBitsCount++;
        destination[dstIdx++] = source[srcIdx++] ?? 0;
      }
    }

    while (srcIdx < srcLen) {
      if (controlBitsCount === 16) {
        this.writeUInt16LE(destination, controlWordIdx, controlWord);
        controlWordIdx = dstIdx;
        dstIdx += 2;
        controlWord = 0;
        controlBitsCount = 0;
      }
      controlWord >>= 1;
      controlBitsCount++;
      destination[dstIdx++] = source[srcIdx++] ?? 0;
    }

    controlWord >>= 16 - controlBitsCount;
    this.writeUInt16LE(destination, controlWordIdx, controlWord);
    this.writeUInt32LE(destination, 5, dstIdx);

    const out = new Uint8Array(new ArrayBuffer(dstIdx));
    out.set(destination.subarray(0, dstIdx));
    return out;
  }

  public static decompress(
    source: Uint8Array,
    maxDecompressedSize = QuickLZ.DEFAULT_MAX_DECOMPRESSED_SIZE,
  ): Uint8Array<ArrayBuffer> {
    if (source.length === 0) return new Uint8Array(new ArrayBuffer(0));

    if (source.length < this.DEFAULT_HEADER_LEN) {
      throw new Error(
        `Invalid QuickLZ stream: expected at least ${this.DEFAULT_HEADER_LEN} header bytes, got ${source.length}`,
      );
    }

    const sizeDecompressed = this.readUInt32LE(source, 1);
    if (sizeDecompressed === 0) {
      return new Uint8Array(new ArrayBuffer(0));
    }

    if (sizeDecompressed < 0 || sizeDecompressed > maxDecompressedSize) {
      throw new Error(
        `QuickLZ decompressed size (${sizeDecompressed} bytes) exceeds safety limit (${maxDecompressedSize} bytes)`,
      );
    }

    const destination = new Uint8Array(new ArrayBuffer(sizeDecompressed));

    let srcIdx = this.DEFAULT_HEADER_LEN;
    let dstIdx = 0;
    let controlWord = 0;
    let controlBitsCount = 0;

    while (dstIdx < sizeDecompressed) {
      if (controlBitsCount === 0) {
        if (srcIdx + 2 > source.length) {
          throw new Error("Truncated QuickLZ stream: missing control word");
        }
        controlWord = this.readUInt16LE(source, srcIdx);
        srcIdx += 2;
        controlBitsCount = 16;
      }

      if ((controlWord & 1) === 1) {
        let matchLen = 0;
        let offset = 0;
        if (srcIdx >= source.length) {
          throw new Error("Truncated QuickLZ stream: unexpected end of match token");
        }
        const b0 = source[srcIdx] ?? 0;

        if ((b0 & 1) === 1) {
          if (srcIdx + 2 > source.length) {
            throw new Error("Truncated QuickLZ stream: truncated 2-byte match token");
          }
          const b1 = source[srcIdx + 1] ?? 0;
          matchLen = b0 >>> 4;
          offset = (((b0 >>> 1) & 0x07) << 8) | b1;
          srcIdx += 2;
        } else {
          if (srcIdx + 3 > source.length) {
            throw new Error("Truncated QuickLZ stream: truncated 3-byte match token");
          }
          const b1 = source[srcIdx + 1] ?? 0;
          matchLen = source[srcIdx + 2] ?? 0;
          offset = (((b0 >>> 1) & 0x07) << 8) | b1;
          srcIdx += 3;
        }

        if (offset <= 0 || offset > dstIdx) {
          throw new Error(
            `Invalid QuickLZ back-reference offset ${offset} at output position ${dstIdx}`,
          );
        }

        if (dstIdx + matchLen > sizeDecompressed) {
          throw new Error(
            `QuickLZ decompression match overflow: dstIdx=${dstIdx}, matchLen=${matchLen}, max=${sizeDecompressed}`,
          );
        }

        let matchIdx = dstIdx - offset;
        for (let i = 0; i < matchLen; i++) {
          destination[dstIdx++] = destination[matchIdx++] ?? 0;
        }
      } else {
        if (srcIdx >= source.length) {
          throw new Error("Truncated QuickLZ stream: truncated literal byte");
        }
        destination[dstIdx++] = source[srcIdx++] ?? 0;
      }

      controlWord >>= 1;
      controlBitsCount--;
    }

    return destination;
  }

  private static readUInt16LE(buf: Uint8Array, offset: number): number {
    return (buf[offset] ?? 0) | ((buf[offset + 1] ?? 0) << 8);
  }

  private static readUInt32LE(buf: Uint8Array, offset: number): number {
    return (
      ((buf[offset] ?? 0) |
        ((buf[offset + 1] ?? 0) << 8) |
        ((buf[offset + 2] ?? 0) << 16) |
        ((buf[offset + 3] ?? 0) << 24)) >>>
      0
    );
  }

  private static writeUInt16LE(buf: Uint8Array, offset: number, val: number): void {
    buf[offset] = val & 0xff;
    buf[offset + 1] = (val >> 8) & 0xff;
  }

  private static writeUInt32LE(buf: Uint8Array, offset: number, val: number): void {
    buf[offset] = val & 0xff;
    buf[offset + 1] = (val >> 8) & 0xff;
    buf[offset + 2] = (val >> 16) & 0xff;
    buf[offset + 3] = (val >> 24) & 0xff;
  }
}
