# TinyToken



## Run it

To run my dev script of tsconfig with hot HTTP server reloading( --hot server.ts):

```bash

bun run dev
```
Not so hot..

```bash

bun run start
```

## Glow2 Design Palette

Cyan ( #38bdf8): Active links, logo accents, primary highlights.
Rose / Coral ( #f43f5e): Post-quantum security badges and variant indicators.
Emerald ( #10b981): Live Bun API status indicators and operational states.
Purple ( #a855f7): Specialized padding and UI markers.
Yellow ( #f78522): Telemetry highlights, payload hashes, and byte map tags.

Extending the design palette with more colors, for testing:
#14213D	DkBlue	Deep Canvas / App Void	Root background, main container surfaces, full-bleed canvas
#124E78	DkBlue2	Elevated Surface / Structural Border	Card backgrounds, sidebar headers, active control pills, panel borders
#333533	DkGray	Neutral Inset / Control Surface	Toolbars, input fields, recessed code blocks, muted button baselines
#CFDBD5	Gray	Secondary Text / Structural Borders	Body copy, timestamps, inactive icons, subtle dividers, hash labels
#E5E5E5	LtGray	Primary Text / High Contrast Readout	Headlines, payload data, active button text, speaker names
#F78522	DkOrange	Telemetry / Byte Mapping / Warn	Telemetry highlights, payload hashes, byte-map badges, packet drops



### Key Considerations for the Separate Process Setup

To make the separate-process model robust in production:

Atomic File Swapping: When the indexing process generates an updated tokens.zip, it should write to a temporary file (e.g. tokens.zip.tmp) and perform an atomic file rename onto tokens.zip. This ensures the web server never attempts to slice from a half-written archive.
Fast Cache Invalidation: If the indexer runs while the server is active, the server needs a lightweight signal (such as watching the file timestamp or a local reload signal) to re-read the trailing _index.json (which takes ~20 ms) and update bun:sqlite.
Decouple In-Place Mutations: Currently, /api/token/:id/mutate writes directly into tokens.zip. If the archive is treated as a compiled, immutable build artifact from an external pipeline, token mutation states should ideally be recorded in an SQLite overlay or separate state ledger rather than mutating the batch ZIP on disk.



This project was created using `bun init` in bun v1.3.14. [Bun](https://bun.com) is a fast all-in-one JavaScript runtime.
