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

//AI made this function
function pointNearLine(
  point: [number, number],
  line: [number, number][],
  toleranceMeters = 30
) {
  const R = 6371000;

  const toRad = (d: number) => (d * Math.PI) / 180;

  function toXY([lng, lat]: [number, number]) {
    return {
      x: R * toRad(lng) * Math.cos(toRad(lat)),
      y: R * toRad(lat)
    };
  }

  const p = toXY(point);

  for (let i = 0; i < line.length - 1; i++) {
    const a = toXY(line[i]);
    const b = toXY(line[i + 1]);

    const dx = b.x - a.x;
    const dy = b.y - a.y;

    const t =
      ((p.x - a.x) * dx + (p.y - a.y) * dy) /
      (dx * dx + dy * dy);

    const clamped = Math.max(0, Math.min(1, t));

    const closest = {
      x: a.x + clamped * dx,
      y: a.y + clamped * dy
    };

    const dist = Math.hypot(p.x - closest.x, p.y - closest.y);

    if (dist <= toleranceMeters) return true;
  }

  return false;
}
//End of AI code

export default function MapView({ data, selectedId, onSelect }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const mapLoadedRef = useRef(false);
  const onSelectRef = useRef(onSelect);
  const allDetectionsRef = useRef<Detection[]>([]);

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

      //Add roads source
      map.addSource("roads", {
        type: "geojson",
        data: "/data/roads.geojson",
        promoteId: "@id"
      });

      // Add detection points source
      map.addSource("detections", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] }
      });

      //Roads
      map.addLayer({
        id: "roads-layer",
        type: "line",
        source: "roads",
        paint: {
          "line-color": [
            "case",
            // Selected road (highest priority)
            ["boolean", ["feature-state", "selected"], false],
            "#c48a0e",

            // Unselected + has pins
            ["boolean", ["feature-state", "hasPins"], false],
            "#ef4444",

            // Unselected + no pins
            "#22c55e"
          ],
          "line-width": 4,
          "line-opacity": 0.9
        }
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
      let selectedRoadId: string | number | null = null;

      map.on("click", "roads-layer", (e) => {
        const feature = e.features?.[0];
        if (!feature || feature.id == null) return;

        // Clear previous selection
        if (selectedRoadId !== null) {
          map.setFeatureState(
            { source: "roads", id: selectedRoadId },
            { selected: false }
          );
        }

        // Set new selection
        selectedRoadId = feature.id;

        map.setFeatureState(
          { source: "roads", id: selectedRoadId },
          { selected: true }
        );
        
        const geometry = feature.geometry;
        if (geometry.type !== "LineString") return;

        const roadCoords = geometry.coordinates as [number, number][];

        const filtered = allDetectionsRef.current.filter((d) =>
          pointNearLine([d.lng, d.lat], roadCoords, 50)
        );

        const geojson: GeoJSON.FeatureCollection = {
          type: "FeatureCollection",
          features: filtered.map((item) => ({
            type: "Feature",
            geometry: {
              type: "Point",
              coordinates: [item.lng, item.lat]
            },
            properties: {
              id: item.id,
              severity: item.severity,
              damage_type: item.damage_type
            }
          }))
        };

        const source = map.getSource("detections") as maplibregl.GeoJSONSource;
        source.setData(geojson);

        console.log("Selected road:", feature.properties);
      });

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

      map.on("mouseenter", "roads-layer", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "roads-layer", () => { map.getCanvas().style.cursor = ""; });
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

    allDetectionsRef.current = data;

    // Wait until the map has finished rendering sources
    map.once("idle", () => {
      const roadFeatures = map.querySourceFeatures("roads");

      for (const feature of roadFeatures) {
        if (!feature.id || feature.geometry.type !== "LineString") continue;

        const roadCoords = feature.geometry.coordinates as [number, number][];

        const hasPins = data.some((d) =>
          pointNearLine([d.lng, d.lat], roadCoords, 50)
        );

        map.setFeatureState(
          { source: "roads", id: feature.id },
          { hasPins }
        );
      }
    });

    // Hide detections by default
    const source = map.getSource("detections") as maplibregl.GeoJSONSource;
    if (source) {
      source.setData({
        type: "FeatureCollection",
        features: []
      });
    }

    // Optional camera movement
    if (data.length > 0) {
      map.easeTo({
        center: [data[0].lng, data[0].lat],
        zoom: 13
      });
    }
  }, [data]);


  return (
    <div
      ref={containerRef}
      className="map-shell"
    />
  );
}
