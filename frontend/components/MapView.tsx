"use client";

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef, useCallback } from "react";
import type { Detection } from "@/data/types";

type MapViewProps = {
  data: Detection[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
};

export default function MapView({ data, selectedId, onSelect }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const mapLoadedRef = useRef(false);
  const onSelectRef = useRef(onSelect);

  // Keep onSelect ref updated
  onSelectRef.current = onSelect;

  // Initialize map once
  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "&copy; OpenStreetMap"
          }
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }]
      },
      center: [-122.4, 37.8],
      zoom: 12
    });

    mapRef.current = map;
    map.addControl(new maplibregl.NavigationControl(), "bottom-right");

    map.on("load", () => {
      mapLoadedRef.current = true;

      // Add detection points source
      map.addSource("detections", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
        cluster: true,
        clusterMaxZoom: 13,
        clusterRadius: 45
      });

      // Clusters
      map.addLayer({
        id: "clusters",
        type: "circle",
        source: "detections",
        filter: ["has", "point_count"],
        paint: {
          "circle-color": ["step", ["get", "point_count"], "#14b8a6", 4, "#38bdf8", 8, "#fb923c"],
          "circle-radius": ["step", ["get", "point_count"], 16, 6, 22, 12, 28],
          "circle-opacity": 0.85
        }
      });

      // Cluster count labels (only when glyphs are available)
      if (map.getStyle()?.glyphs) {
        map.addLayer({
          id: "cluster-count",
          type: "symbol",
          source: "detections",
          filter: ["has", "point_count"],
          layout: {
            "text-field": "{point_count_abbreviated}",
            "text-size": 12
          },
          paint: { "text-color": "#0b1220" }
        });
      }

      // Individual points
      map.addLayer({
        id: "unclustered",
        type: "circle",
        source: "detections",
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-color": [
            "interpolate", ["linear"], ["get", "severity"],
            0, "#22c55e", 0.45, "#38bdf8", 0.7, "#fb923c", 1, "#ef4444"
          ],
          "circle-radius": 8,
          "circle-stroke-color": "#0b1220",
          "circle-stroke-width": 1.2
        }
      });

      // Click handlers
      map.on("click", "clusters", (e) => {
        const features = map.queryRenderedFeatures(e.point, { layers: ["clusters"] });
        const clusterId = features[0]?.properties?.cluster_id;
        if (clusterId == null) return;
        const source = map.getSource("detections") as maplibregl.GeoJSONSource;
        source.getClusterExpansionZoom(clusterId).then((zoom) => {
          map.easeTo({ center: (features[0].geometry as any).coordinates, zoom: zoom ?? 14 });
        });
      });

      map.on("click", "unclustered", (e) => {
        const id = e.features?.[0]?.properties?.id;
        if (id && onSelectRef.current) onSelectRef.current(String(id));
      });

      map.on("mouseenter", "unclustered", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "unclustered", () => { map.getCanvas().style.cursor = ""; });
    });

    return () => {
      mapLoadedRef.current = false;
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Update data when it changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;

    const geojson: GeoJSON.FeatureCollection = {
      type: "FeatureCollection",
      features: data.map((item) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [item.lng, item.lat] },
        properties: { id: item.id, severity: item.severity, damage_type: item.damage_type }
      }))
    };

    const source = map.getSource("detections") as maplibregl.GeoJSONSource;
    if (source) source.setData(geojson);

    // Center on first point if we have data
    if (data.length > 0) {
      map.easeTo({ center: [data[0].lng, data[0].lat], zoom: 13 });
    }
  }, [data]);

  return (
    <div
      ref={containerRef}
      className="map-shell"
    />
  );
}
