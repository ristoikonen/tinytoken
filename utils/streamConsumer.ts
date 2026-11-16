/**
 * TinyToken Binary Stream Consumer (utils/streamConsumer.ts)
 *
 * Implements a zero-copy stateful framer for `application/vnd.tinytoken.stream`.
 * Slices 4-byte Uint32LE length-prefixed TinyToken binary records from any
 * ReadableStream<Uint8Array>, handling arbitrary TCP packet fragmentation and chunk splits.
 */

/**
 * Encapsulates a raw TinyToken byte payload with a 4-byte Uint32LE length prefix.
 */
export function createFramedToken(tokenBytes: Uint8Array): Uint8Array {
  const frame = new Uint8Array(4 + tokenBytes.byteLength);
  const view = new DataView(frame.buffer, frame.byteOffset, frame.byteLength);
  view.setUint32(0, tokenBytes.byteLength, true); // Little-Endian
  frame.set(tokenBytes, 4);
  return frame;
}

/**
 * Consumes a binary stream of 4-byte Uint32LE length-prefixed TinyTokens.
 * Yields each complete token container as a Uint8Array.
 *
 * Resilient to TCP fragmentation (e.g. 1-byte, 17-byte, or multi-packet chunks).
 */
export async function* consumeTinyTokenStream(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<Uint8Array> {
  const reader = stream.getReader();
  let buffer = new Uint8Array(0);

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value || value.byteLength === 0) continue;

      // Accumulate new bytes
      const next = new Uint8Array(buffer.byteLength + value.byteLength);
      next.set(buffer, 0);
      next.set(value, buffer.byteLength);
      buffer = next;

      // Drain all complete frames from the accumulator
      while (buffer.byteLength >= 4) {
        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        const frameLen = view.getUint32(0, true);

        if (buffer.byteLength < 4 + frameLen) {
          // Frame incomplete, await further TCP chunks
          break;
        }

        const tokenBytes = buffer.slice(4, 4 + frameLen);
        buffer = buffer.slice(4 + frameLen);
        yield tokenBytes;
      }
    }
  } finally {
    reader.releaseLock();
  }
}
