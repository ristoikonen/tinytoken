export interface ArchiveRecord {
    id: string;
    path: string;
    size: number;
    sha256?: string;
}

export const archives = new Map<string, ArchiveRecord>([
    [
        "default",
        {
            id: "default",
            path: "./data/tokens.zip",
            size: 0,
        },
    ],
    [
        "november-2026",
        {
            id: "november-2026",
            path: "./data/tokens-november.zip",
            size: 0,
        },
    ],
]);