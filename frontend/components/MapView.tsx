"use client";

import maplibregl from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useMemo, useRef } from "react";
import type { Detection } from "@/data/types";

const styleUrl = "https://demotiles.maplibre.org/style.json";

type MapViewProps = {
  data: Detection[];
  selectedId?: string | null;
  onSelect?: (id: string) => void;
};

export default function MapView({ data, selectedId, onSelect }: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const mapLoadedRef = useRef(false);

  const geojson = useMemo(() => {
    return {
      type: "FeatureCollection",
      features: data.map((item) => ({
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
    } as GeoJSON.FeatureCollection;
  }, [data]);

  const route = useMemo(() => {
    const sorted = [...data].sort((a, b) =>
      a.captured_at.localeCompare(b.captured_at)
    );
    const coords = sorted.map((item) => [item.lng, item.lat]);
    return {
      type: "Feature",
      geometry: {
        type: "LineString",
        coordinates: coords
      }
    } as GeoJSON.Feature;
  }, [data]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const center: [number, number] = data.length
      ? [data[0].lng, data[0].lat]
      : [-122.4194, 37.7749];

    const map = new maplibregl.Map({
      container: containerRef.current,
      style: styleUrl,
      center,
      zoom: 12.5,
      pitch: 35
    });

    mapRef.current = map;

    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "bottom-right");

    map.on("load", () => {
      mapLoadedRef.current = true;
      map.addSource("detections", {
        type: "geojson",
        data: geojson,
        cluster: true,
        clusterMaxZoom: 13,
        clusterRadius: 45
      });

      map.addSource("route", {
        type: "geojson",
        data: route
      });

      map.addLayer({
        id: "route-line",
        type: "line",
        source: "route",
        paint: {
          "line-color": "#0ea5e9",
          "line-width": 3,
          "line-opacity": 0.6
        }
      });

      map.addLayer({
        id: "clusters",
        type: "circle",
        source: "detections",
        filter: ["has", "point_count"],
        paint: {
          "circle-color": [
            "step",
            ["get", "point_count"],
            "#14b8a6",
            4,
            "#38bdf8",
            8,
            "#fb923c"
          ],
          "circle-radius": ["step", ["get", "point_count"], 16, 6, 22, 12, 28],
          "circle-opacity": 0.85
        }
      });

      map.addLayer({
        id: "cluster-count",
        type: "symbol",
        source: "detections",
        filter: ["has", "point_count"],
        layout: {
          "text-field": "{point_count_abbreviated}",
          "text-font": ["Open Sans Bold"],
          "text-size": 12
        },
        paint: {
          "text-color": "#0b1220"
        }
      });

      map.addLayer({
        id: "unclustered",
        type: "circle",
        source: "detections",
        filter: ["!", ["has", "point_count"]],
        paint: {
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "severity"],
            0,
            "#22c55e",
            0.45,
            "#38bdf8",
            0.7,
            "#fb923c",
            1,
            "#ef4444"
          ],
          "circle-radius": 8,
          "circle-stroke-color": "#0b1220",
          "circle-stroke-width": 1.2
        }
      });

      map.addLayer({
        id: "selected-point",
        type: "circle",
        source: "detections",
        filter: ["==", ["get", "id"], ""],
        paint: {
          "circle-color": "#111827",
          "circle-radius": 14,
          "circle-opacity": 0.2
        }
      });

      map.on("click", "clusters", (event) => {
        const features = map.queryRenderedFeatures(event.point, {
          layers: ["clusters"]
        });
        const clusterId = features[0]?.properties?.cluster_id;
        if (clusterId === undefined || clusterId === null) return;
        const source = map.getSource("detections") as maplibregl.GeoJSONSource;
        source
          .getClusterExpansionZoom(clusterId)
          .then((zoom) => {
            if (zoom === null || zoom === undefined) return;
            map.easeTo({
              center: (features[0].geometry as any).coordinates,
              zoom
            });
          })
          .catch(() => undefined);
      });

      map.on("click", "unclustered", (event) => {
        const features = map.queryRenderedFeatures(event.point, {
          layers: ["unclustered"]
        });
        const id = features[0]?.properties?.id;
        if (id && onSelect) {
          onSelect(String(id));
        }
      });

      map.on("mouseenter", "unclustered", () => {
        map.getCanvas().style.cursor = "pointer";
      });

      map.on("mouseleave", "unclustered", () => {
        map.getCanvas().style.cursor = "";
      });
    });

    return () => {
      mapLoadedRef.current = false;
      map.remove();
    };
  }, [data, geojson, route, onSelect]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;
    const source = map.getSource("detections") as maplibregl.GeoJSONSource;
    if (source) {
      source.setData(geojson);
    }
    const routeSource = map.getSource("route") as maplibregl.GeoJSONSource;
    if (routeSource) {
      routeSource.setData(route);
    }
  }, [geojson, route]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapLoadedRef.current) return;
    if (map.getLayer("selected-point")) {
      map.setFilter("selected-point", ["==", ["get", "id"], selectedId ?? ""]);
    }
  }, [selectedId]);

  return <div ref={containerRef} className="map-shell" />;
}
