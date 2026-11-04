/**
 * Benchmark Instrumenter & Telemetry Reporter for TinyToken Pipeline (utils/benchmark.ts)
 *
 * Instruments each stage independently using Bun high-resolution nanosecond timers (Bun.nanoseconds()):
 * 1. File reading: MB/s
 * 2. Sharp processing: ms/image
 * 3. DINO inference: ms/image
 * 4. VQ generation: ms/image
 * 5. Binary encoding: ms/token
 * 6. Binary decoding: ms/token
 * 7. Signature verification: ms/token
 * 8. Archive writing: MB/s
 *
 * Plus end-to-end latency percentiles (min, avg, p50, p95, p99, max) and automated bottleneck detection.
 * Emits compact [METRIC] lines to the debug console.
 */

export interface LatencyStats {
  minMs: number;
  avgMs: number;
  p50Ms: number;
  p95Ms: number;
  p99Ms: number;
  maxMs: number;
}

export interface PipelineBenchmarkReport {
  fileReadingMbSec: number;
  fileReadingTotalMs: number;
  totalFileBytes: number;
  fileCount: number;

  sharpMsPerImage: number;
  sharpTotalMs: number;
  sharpCount: number;

  dinoMsPerImage: number;
  dinoTotalMs: number;
  dinoCount: number;

  vqMsPerImage: number;
  vqTotalMs: number;
  vqCount: number;

  binaryEncodingMsPerToken: number;
  binaryEncodingTotalMs: number;
  encodeCount: number;

  binaryDecodingMsPerToken: number;
  binaryDecodingTotalMs: number;
  decodeCount: number;

  sigVerifyMsPerToken: number;
  sigVerifyTotalMs: number;
  sigVerifyCount: number;

  archiveWritingMbSec: number;
  archiveWritingTotalMs: number;
  archiveSizeBytes: number;

  latency: LatencyStats;

  summary: {
    totalTokens: number;
    totalPipelineMs: number;
    tokensPerSecond: number;
    primaryBottleneck: string;
    stageBreakdownPct: Record<string, number>;
  };
}

function calculatePercentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0]!;
  const idx = (sorted.length - 1) * p;
  const lower = Math.floor(idx);
  const upper = Math.ceil(idx);
  const weight = idx - lower;
  const val = (sorted[lower]! * (1 - weight)) + (sorted[upper]! * weight);
  return Math.round(val * 100) / 100;
}

export class PipelineBenchmark {
  private fileReadBytes = 0;
  private fileReadNs = 0;
  private fileReadCount = 0;

  private sharpNs = 0;
  private sharpCount = 0;

  private dinoNs = 0;
  private dinoCount = 0;

  private vqNs = 0;
  private vqCount = 0;

  private encodeNs = 0;
  private encodeCount = 0;

  private decodeNs = 0;
  private decodeCount = 0;

  private sigVerifyNs = 0;
  private sigVerifyCount = 0;

  private archiveBytes = 0;
  private archiveWriteNs = 0;

  private tokenLatenciesMs: number[] = [];
  private tStartNs: number;

  constructor() {
    this.tStartNs = Bun.nanoseconds();
  }

  public recordFileRead(bytes: number, durationNs: number): void {
    this.fileReadBytes += bytes;
    this.fileReadNs += Math.max(0, durationNs);
    this.fileReadCount++;
  }

  public recordSharp(durationNs: number, count = 1): void {
    this.sharpNs += Math.max(0, durationNs);
    this.sharpCount += count;
  }

  public recordDino(durationNs: number, count = 1): void {
    this.dinoNs += Math.max(0, durationNs);
    this.dinoCount += count;
  }

  public recordVq(durationNs: number, count = 1): void {
    this.vqNs += Math.max(0, durationNs);
    this.vqCount += count;
  }

  public recordEncode(durationNs: number, count = 1): void {
    this.encodeNs += Math.max(0, durationNs);
    this.encodeCount += count;
  }

  public recordDecode(durationNs: number, count = 1): void {
    this.decodeNs += Math.max(0, durationNs);
    this.decodeCount += count;
  }

  public recordSigVerify(durationNs: number, count = 1): void {
    this.sigVerifyNs += Math.max(0, durationNs);
    this.sigVerifyCount += count;
  }

  public recordArchiveWrite(bytes: number, durationNs: number): void {
    this.archiveBytes += bytes;
    this.archiveWriteNs += Math.max(0, durationNs);
  }

  public recordTokenLatency(durationNs: number): void {
    this.tokenLatenciesMs.push(Math.round((durationNs / 1_000_000) * 100) / 100);
  }

  public getReport(): PipelineBenchmarkReport {
    const totalPipelineNs = Bun.nanoseconds() - this.tStartNs;
    const totalPipelineMs = Math.round((totalPipelineNs / 1_000_000) * 100) / 100;

    const readSec = this.fileReadNs / 1e9;
    const readMb = this.fileReadBytes / (1024 * 1024);
    const fileReadingMbSec = readSec > 0 ? Math.round((readMb / readSec) * 10) / 10 : 0;
    const fileReadingTotalMs = Math.round((this.fileReadNs / 1e6) * 100) / 100;

    const sharpTotalMs = Math.round((this.sharpNs / 1e6) * 100) / 100;
    const sharpMsPerImage =
      this.sharpCount > 0 ? Math.round((sharpTotalMs / this.sharpCount) * 100) / 100 : 0;

    const dinoTotalMs = Math.round((this.dinoNs / 1e6) * 100) / 100;
    const dinoMsPerImage =
      this.dinoCount > 0 ? Math.round((dinoTotalMs / this.dinoCount) * 100) / 100 : 0;

    const vqTotalMs = Math.round((this.vqNs / 1e6) * 100) / 100;
    const vqMsPerImage =
      this.vqCount > 0 ? Math.round((vqTotalMs / this.vqCount) * 1000) / 1000 : 0;

    const binaryEncodingTotalMs = Math.round((this.encodeNs / 1e6) * 100) / 100;
    const binaryEncodingMsPerToken =
      this.encodeCount > 0 ? Math.round((binaryEncodingTotalMs / this.encodeCount) * 100) / 100 : 0;

    const binaryDecodingTotalMs = Math.round((this.decodeNs / 1e6) * 100) / 100;
    const binaryDecodingMsPerToken =
      this.decodeCount > 0 ? Math.round((binaryDecodingTotalMs / this.decodeCount) * 100) / 100 : 0;

    const sigVerifyTotalMs = Math.round((this.sigVerifyNs / 1e6) * 100) / 100;
    const sigVerifyMsPerToken =
      this.sigVerifyCount > 0 ? Math.round((sigVerifyTotalMs / this.sigVerifyCount) * 100) / 100 : 0;

    const writeSec = this.archiveWriteNs / 1e9;
    const writeMb = this.archiveBytes / (1024 * 1024);
    const archiveWritingMbSec = writeSec > 0 ? Math.round((writeMb / writeSec) * 10) / 10 : 0;
    const archiveWritingTotalMs = Math.round((this.archiveWriteNs / 1e6) * 100) / 100;

    // Latency percentiles
    const sorted = [...this.tokenLatenciesMs].sort((a, b) => a - b);
    const count = sorted.length;
    const minMs = count > 0 ? sorted[0]! : 0;
    const maxMs = count > 0 ? sorted[count - 1]! : 0;
    const sumMs = sorted.reduce((acc, v) => acc + v, 0);
    const avgMs = count > 0 ? Math.round((sumMs / count) * 100) / 100 : 0;
    const p50Ms = calculatePercentile(sorted, 0.5);
    const p95Ms = calculatePercentile(sorted, 0.95);
    const p99Ms = calculatePercentile(sorted, 0.99);

    const latency: LatencyStats = {
      minMs,
      avgMs,
      p50Ms,
      p95Ms,
      p99Ms,
      maxMs,
    };

    // Calculate Stage Breakdown & Identify Primary Bottleneck
    const stages: Array<{ key: string; name: string; ms: number }> = [
      { key: "fileRead", name: "File reading", ms: fileReadingTotalMs },
      { key: "sharp", name: "Sharp processing", ms: sharpTotalMs },
      { key: "dino", name: "DINO inference", ms: dinoTotalMs },
      { key: "vq", name: "VQ generation", ms: vqTotalMs },
      { key: "encode", name: "Binary encoding", ms: binaryEncodingTotalMs },
      { key: "decode", name: "Binary decoding", ms: binaryDecodingTotalMs },
      { key: "archiveWrite", name: "Archive writing", ms: archiveWritingTotalMs },
    ];

    const totalStageMs = stages.reduce((acc, s) => acc + s.ms, 0) || 1;
    const stageBreakdownPct: Record<string, number> = {};
    for (const st of stages) {
      stageBreakdownPct[st.key] = Math.round((st.ms / totalStageMs) * 1000) / 10;
    }

    stages.sort((a, b) => b.ms - a.ms);
    const top = stages[0] ?? { name: "None", ms: 0 };
    const topPct = ((top.ms / totalStageMs) * 100).toFixed(1);
    const primaryBottleneck = `${top.name} (${topPct}% of stage time, ${top.ms.toFixed(1)}ms)`;

    const totalTokens = this.encodeCount;
    const pipelineSec = totalPipelineNs / 1e9;
    const tokensPerSecond =
      pipelineSec > 0 ? Math.round((totalTokens / pipelineSec) * 10) / 10 : 0;

    return {
      fileReadingMbSec,
      fileReadingTotalMs,
      totalFileBytes: this.fileReadBytes,
      fileCount: this.fileReadCount,

      sharpMsPerImage,
      sharpTotalMs,
      sharpCount: this.sharpCount,

      dinoMsPerImage,
      dinoTotalMs,
      dinoCount: this.dinoCount,

      vqMsPerImage,
      vqTotalMs,
      vqCount: this.vqCount,

      binaryEncodingMsPerToken,
      binaryEncodingTotalMs,
      encodeCount: this.encodeCount,

      binaryDecodingMsPerToken,
      binaryDecodingTotalMs,
      decodeCount: this.decodeCount,

      sigVerifyMsPerToken,
      sigVerifyTotalMs,
      sigVerifyCount: this.sigVerifyCount,

      archiveWritingMbSec,
      archiveWritingTotalMs,
      archiveSizeBytes: this.archiveBytes,

      latency,

      summary: {
        totalTokens,
        totalPipelineMs,
        tokensPerSecond,
        primaryBottleneck,
        stageBreakdownPct,
      },
    };
  }

  public formatMetricLines(): string[] {
    const r = this.getReport();
    const readMb = (r.totalFileBytes / (1024 * 1024)).toFixed(1);
    const archKb = (r.archiveSizeBytes / 1024).toFixed(1);

    const line1 = `[METRIC] Read: ${r.fileReadingMbSec} MB/s (${r.fileReadingTotalMs.toFixed(1)}ms, ${readMb}MB) | Sharp: ${r.sharpMsPerImage}ms/img | DINO: ${r.dinoMsPerImage}ms/img | VQ: ${r.vqMsPerImage}ms/img`;
    const line2 = `[METRIC] Enc: ${r.binaryEncodingMsPerToken}ms/tok | Dec: ${r.binaryDecodingMsPerToken}ms/tok | SigVerify: ${r.sigVerifyMsPerToken}ms/tok | ArchiveWrite: ${r.archiveWritingMbSec} MB/s (${r.archiveWritingTotalMs.toFixed(1)}ms, ${archKb}KB)`;
    const line3 = `[METRIC] Latency (e2e): min ${r.latency.minMs}ms | avg ${r.latency.avgMs}ms | p50 ${r.latency.p50Ms}ms | p95 ${r.latency.p95Ms}ms | p99 ${r.latency.p99Ms}ms | max ${r.latency.maxMs}ms`;
    const line4 = `[METRIC] Pipeline: ${r.summary.totalTokens} tokens in ${r.summary.totalPipelineMs.toFixed(1)}ms (${r.summary.tokensPerSecond} tok/s) | Bottleneck: ${r.summary.primaryBottleneck}`;

    return [line1, line2, line3, line4];
  }

  public printMetrics(): void {
    const lines = this.formatMetricLines();
    for (const line of lines) {
      console.log(line);
    }
  }
}

