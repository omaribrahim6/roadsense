export type DetectionBBox = {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
};

export type Detection = {
  id: string;
  created_at: string;
  captured_at: string;
  lat: number;
  lng: number;
  damage_type: "pothole" | "crack" | "rut" | "debris";
  confidence: number;
  severity: number;
  bbox: DetectionBBox;
  image_path: string;
  source_id: string;
  status?: "new" | "reviewed" | "repaired";
};
