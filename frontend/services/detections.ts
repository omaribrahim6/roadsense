import type { SupabaseClient } from "@supabase/supabase-js";
import type { Detection, DetectionBBox } from "@/data/types";
import { getSupabaseClient } from "@/services/supabaseClient";

const STORAGE_BUCKET = "detections-images";
const FALLBACK_IMAGE = "/mock/road-sample.svg";

type DetectionRow = {
  id: string;
  created_at: string;
  captured_at: string | null;
  lat: number;
  lng: number;
  damage_type: Detection["damage_type"] | string;
  confidence: number;
  severity: number;
  bbox: DetectionBBox | string | null;
  image_path: string;
  source_id: string;
  status: Detection["status"] | null;
};

const isAbsoluteUrl = (value: string) => /^https?:\/\//i.test(value);

const toNumber = (value: unknown, fallback: number) => {
  if (typeof value === "number" && !Number.isNaN(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    return Number.isNaN(parsed) ? fallback : parsed;
  }
  return fallback;
};

const normalizeBBox = (value: DetectionRow["bbox"]): DetectionBBox => {
  if (!value) {
    return { x1: 0, y1: 0, x2: 1, y2: 1 };
  }

  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      return normalizeBBox(parsed as DetectionBBox);
    } catch {
      return { x1: 0, y1: 0, x2: 1, y2: 1 };
    }
  }

  const bbox = value as DetectionBBox;
  return {
    x1: toNumber(bbox.x1, 0),
    y1: toNumber(bbox.y1, 0),
    x2: toNumber(bbox.x2, 1),
    y2: toNumber(bbox.y2, 1)
  };
};

const normalizeDamageType = (
  value: DetectionRow["damage_type"]
): Detection["damage_type"] => {
  if (value === "pothole" || value === "crack" || value === "rut" || value === "debris") {
    return value;
  }
  return "pothole";
};

const normalizeStatus = (value: DetectionRow["status"]): Detection["status"] => {
  if (value === "new" || value === "reviewed" || value === "repaired") {
    return value;
  }
  return "new";
};

const resolveImageUrl = (
  supabase: SupabaseClient,
  imagePath: string | null
) => {
  if (!imagePath) return FALLBACK_IMAGE;
  if (isAbsoluteUrl(imagePath)) return imagePath;

  const { data } = supabase.storage.from(STORAGE_BUCKET).getPublicUrl(imagePath);
  return data?.publicUrl || imagePath || FALLBACK_IMAGE;
};

const mapRowToDetection = (
  row: DetectionRow,
  supabase: SupabaseClient
): Detection => {
  return {
    id: String(row.id),
    created_at: String(row.created_at),
    captured_at: String(row.captured_at ?? row.created_at),
    lat: toNumber(row.lat, 0),
    lng: toNumber(row.lng, 0),
    damage_type: normalizeDamageType(row.damage_type),
    confidence: toNumber(row.confidence, 0),
    severity: toNumber(row.severity, 0),
    bbox: normalizeBBox(row.bbox),
    image_path: resolveImageUrl(supabase, row.image_path),
    source_id: String(row.source_id ?? ""),
    status: normalizeStatus(row.status)
  };
};

export async function fetchDetections(limit = 500): Promise<Detection[]> {
  const supabase = getSupabaseClient();
  const { data, error } = await supabase
    .from("detections")
    .select(
      "id, created_at, captured_at, lat, lng, damage_type, confidence, severity, bbox, image_path, source_id, status"
    )
    .order("captured_at", { ascending: false })
    .limit(limit);

  if (error) {
    throw new Error(error.message);
  }

  return (data ?? []).map((row) =>
    mapRowToDetection(row as DetectionRow, supabase)
  );
}
