import { NextRequest, NextResponse } from "next/server";
import { Client } from "@gradio/client";
import fs from "node:fs/promises";
import path from "node:path";
import { Buffer } from "node:buffer";
import diseaseInfo from "@/data/disease_info.json";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type DiseaseInfo = {
  nama?: string;
  deskripsi?: string;
  pengobatan?: string;
  perawatan?: string;
  pencegahan?: string;
};
type Probabilities = Record<string, number>;

function getEnv(name: string): string | undefined {
  const value = process.env[name];
  return value && value.trim().length > 0 ? value : undefined;
}

function isHfToken(value: string): value is `hf_${string}` {
  return value.startsWith("hf_");
}

function toHfToken(value: string): `hf_${string}` {
  return isHfToken(value) ? value : (`hf_${value}` as `hf_${string}`);
}

async function tryReadJson<T = unknown>(filePath: string): Promise<T | null> {
  try {
    const content = await fs.readFile(filePath, "utf-8");
    return JSON.parse(content) as T;
  } catch {
    return null;
  }
}

async function loadDiseaseInfo(): Promise<Record<string, DiseaseInfo>> {
  const backendPath = path.resolve(process.cwd(), "..", "backend", "artifacts", "disease_info.json");
  const fallback = diseaseInfo as Record<string, DiseaseInfo>;
  const data = (await tryReadJson<Record<string, DiseaseInfo>>(backendPath)) ?? fallback;
  return data;
}

async function loadLabels(): Promise<string[]> {
  const backendPath = path.resolve(process.cwd(), "..", "backend", "artifacts", "class_indices.json");
  const mapping = await tryReadJson<Record<string, string>>(backendPath);
  if (mapping) {
    const entries = Object.entries(mapping).sort((a, b) => Number(a[0]) - Number(b[0]));
    return entries.map(([, name]) => name);
  }
  const info = await loadDiseaseInfo();
  return Object.keys(info).filter((k) => k !== "DEFAULT");
}

// Convert various payload variants to a proper File for Gradio
async function coerceToImageFile(value: unknown): Promise<File | null> {
  try {
    if (!value) return null;
    if (value instanceof File) return value;
    if (value instanceof Blob) {
      const maybeName = (value as unknown as { name?: unknown }).name;
      const name = typeof maybeName === "string" ? maybeName : "upload";
      const type = value.type || "application/octet-stream";
      return new File([value], name, { type });
    }
    if (typeof value === "string") {
      // Support data URL or raw base64
      let mime = "image/jpeg";
      let ab: ArrayBuffer | null = null;
      const dataUrlMatch = /^data:([^;]+);base64,(.*)$/i.exec(value);
      if (dataUrlMatch) {
        mime = dataUrlMatch[1] || mime;
        const base64 = dataUrlMatch[2] || "";
        const buf = Buffer.from(base64, "base64");
        ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      } else if (/^[A-Za-z0-9+/=]+$/.test(value.trim())) {
        // heuristically treat as base64
        const buf = Buffer.from(value.trim(), "base64");
        ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
      }
      if (ab) {
        return new File([ab], `upload.${mime.split("/")[1] ?? "jpg"}`, { type: mime });
      }
    }
    if (value instanceof ArrayBuffer) {
      return new File([value], "upload.jpg", { type: "image/jpeg" });
    }
    if (ArrayBuffer.isView(value)) {
      const view = value as ArrayBufferView;
      // Copy bytes into a fresh Uint8Array to guarantee an ArrayBuffer (not SharedArrayBuffer)
      const src = new Uint8Array(view.buffer, view.byteOffset, view.byteLength);
      const copy = new Uint8Array(src.length);
      copy.set(src);
      const ab2: ArrayBuffer = copy.buffer; // ensures BlobPart compatibility
      return new File([ab2], "upload.jpg", { type: "image/jpeg" });
    }
  } catch {
    // fall through
  }
  return null;
}

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    // Accept common field names from clients: "file" (our FE), or "image" (Postman habit)
    const candidateKeys = ["file", "image", "image_file", "img", "image_base64", "file_base64"] as const;
    let file: File | null = null;
    for (const key of candidateKeys) {
      const v = formData.get(key);
      const asFile = await coerceToImageFile(v as unknown);
      if (asFile) {
        file = asFile;
        break;
      }
    }

    if (!(file instanceof File)) {
      return NextResponse.json(
        {
          error:
            "File gambar tidak ditemukan pada request. Gunakan field 'file' atau 'image' (form-data).",
        },
        { status: 400 },
      );
    }

    // Optional server-side guardrails (mirror client constraints)
    const maxBytes = 8 * 1024 * 1024; // 8MB
    if (file.size <= 0) {
      return NextResponse.json(
        { error: "File gambar kosong." },
        { status: 400 },
      );
    }
    if (file.size > maxBytes) {
      return NextResponse.json(
        { error: "Ukuran file melebihi batas 8MB." },
        { status: 400 },
      );
    }

    const hfToken = getEnv("HF_TOKEN");
    const spaceId = getEnv("GRADIO_SPACE_ID");
    const spaceUrl = getEnv("GRADIO_SPACE");
    // Provide a safe default pointing to your Space if envs are missing
    const DEFAULT_SPACE = "Taufik2307/plant_disease";
    const spaceTarget = spaceId ?? spaceUrl ?? DEFAULT_SPACE;
      if (!spaceTarget) {
        return NextResponse.json(
          {
            error:
              "Konfigurasi Space tidak ditemukan. Set variabel GRADIO_SPACE_ID atau GRADIO_SPACE.",
          },
          { status: 500 },
        );
      }

    // Initialize client to the target Space
    const app = await Client.connect(
      spaceTarget,
      hfToken ? { token: toHfToken(hfToken) } : undefined,
    );

    // Call the predict API exposed by the Space
    // Prefer array form with File first (most Spaces map by index), then try named inputs
    const ENDPOINT = "/predict" as const;
    let result: unknown | null = null;
    try {
      result = await app.predict(ENDPOINT, [file]);
    } catch {
      // Try a few common input field names when array form fails
      const INPUT_KEYS = ["image", "image_pil", "file"] as const;
      for (const key of INPUT_KEYS) {
        try {
          result = await app.predict(ENDPOINT, { [key]: file } as Record<string, unknown>);
          if (result) break;
        } catch {
          // continue trying keys
        }
      }
    }
    if (!result) {
      return NextResponse.json(
        {
          error:
            "Gagal memanggil model remote: parameter input tidak cocok. Coba periksa nama endpoint/input di Space.",
        },
        { status: 502 },
      );
    }

    if (!result || typeof result !== "object") {
      return NextResponse.json(
        { error: "Format response dari model remote tidak dikenali." },
        { status: 502 },
      );
    }

    // Typical shape expected from the Space based on backend usage:
    // { label: string, percentage: number, probabilities?: Record<string, number> }
    const obj = result as Record<string, unknown>;
    let label = typeof obj["label"] === "string" ? (obj["label"] as string) : undefined;
    const rawPercentage = obj["percentage"];
    let percentage =
      typeof rawPercentage === "number" || typeof rawPercentage === "string"
        ? rawPercentage
        : undefined;

    // Extract probabilities if present
    let probabilities: Probabilities | undefined;
    const maybeProbs = obj["probabilities"] as unknown;
    if (maybeProbs && typeof maybeProbs === "object" && !Array.isArray(maybeProbs)) {
      const map: Probabilities = {};
      for (const [k, v] of Object.entries(maybeProbs as Record<string, unknown>)) {
        const num = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
        if (Number.isFinite(num)) map[k] = num;
      }
      probabilities = map;
    }

    // Fallback: some Spaces return { data: [ { label, percentage, ... } ] }
    if ((!label || percentage === undefined) && "data" in obj) {
      const data = (obj as { data?: unknown }).data;
      if (Array.isArray(data) && data.length > 0 && typeof data[0] === "object" && data[0] !== null) {
        const first = data[0] as Record<string, unknown>;
        if (typeof first["label"] === "string") {
          label = first["label"] as string;
        }
        const maybePerc = first["percentage"];
        if (typeof maybePerc === "number" || typeof maybePerc === "string") {
          percentage = maybePerc;
        }
        const inner = first["probabilities"] as unknown;
        if (!probabilities && inner && typeof inner === "object" && !Array.isArray(inner)) {
          const map: Probabilities = {};
          for (const [k, v] of Object.entries(inner as Record<string, unknown>)) {
            const num = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
            if (Number.isFinite(num)) map[k] = num;
          }
          probabilities = map;
        }
      }
    }

    if (!label || percentage === undefined || percentage === null) {
      return NextResponse.json(
        { error: "Response model tidak mengandung label atau percentage." },
        { status: 502 },
      );
    }

    const confidence = Number(percentage);
    const infoMap = await loadDiseaseInfo();
    const info: DiseaseInfo = infoMap[label] ?? infoMap["DEFAULT"] ?? {};

    // Expand probabilities with labels from class_indices.json
    const labels = await loadLabels();
    const finalProbs: Probabilities = {};
    for (const cls of labels) {
      const val = probabilities?.[cls];
      finalProbs[cls] = Number.isFinite(val as number) ? (val as number) : 0;
    }
    if (!probabilities && Number.isFinite(confidence)) {
      finalProbs[label] = confidence;
    }

    return NextResponse.json(
      {
        prediction: label,
        confidence: Number.isFinite(confidence) ? confidence : 0,
        info,
        probabilities: finalProbs,
      },
      { status: 200 },
    );
  } catch (error: unknown) {
    // Surface a concise error to the client
    const message = error instanceof Error ? error.message : "Terjadi kesalahan saat memproses permintaan.";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

export async function GET() {
  return NextResponse.json({ status: "ok" }, { status: 200 });
}
