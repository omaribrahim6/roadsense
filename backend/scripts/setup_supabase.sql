-- RoadSense Supabase Database Schema
-- Run this in the Supabase SQL Editor

-- create detections table
CREATE TABLE IF NOT EXISTS detections (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    created_at TIMESTAMPTZ DEFAULT now(),
    captured_at TIMESTAMPTZ NOT NULL,
    lat FLOAT8 NOT NULL,
    lng FLOAT8 NOT NULL,
    damage_type TEXT NOT NULL CHECK (damage_type IN ('pothole', 'crack', 'rut', 'debris')),
    confidence FLOAT4 NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    severity FLOAT4 NOT NULL CHECK (severity >= 0 AND severity <= 1),
    bbox JSONB NOT NULL,
    image_path TEXT NOT NULL,
    source_id TEXT NOT NULL,
    status TEXT DEFAULT 'new' CHECK (status IN ('new', 'reviewed', 'repaired'))
);

-- indexes for common queries
CREATE INDEX IF NOT EXISTS idx_detections_source ON detections(source_id);
CREATE INDEX IF NOT EXISTS idx_detections_type ON detections(damage_type);
CREATE INDEX IF NOT EXISTS idx_detections_severity ON detections(severity);
CREATE INDEX IF NOT EXISTS idx_detections_location ON detections(lat, lng);
CREATE INDEX IF NOT EXISTS idx_detections_created ON detections(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_detections_status ON detections(status);

-- optional: sources table for tracking processing runs
CREATE TABLE IF NOT EXISTS sources (
    id TEXT PRIMARY KEY,
    name TEXT,
    description TEXT,
    video_path TEXT,
    frame_count INTEGER,
    detection_count INTEGER,
    created_at TIMESTAMPTZ DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- helpful queries for testing:

-- view recent detections
-- SELECT * FROM detections ORDER BY created_at DESC LIMIT 10;

-- count detections by type
-- SELECT damage_type, COUNT(*) as count FROM detections GROUP BY damage_type;

-- get detections for a source
-- SELECT * FROM detections WHERE source_id = 'your_source_id';

-- delete all detections for a source (careful!)
-- DELETE FROM detections WHERE source_id = 'test_run';
