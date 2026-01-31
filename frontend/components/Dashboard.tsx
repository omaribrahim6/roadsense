"use client";

import dynamic from "next/dynamic";
import Image from "next/image";
import { useMemo, useState } from "react";
import { mockDetections } from "@/data/mockDetections";
import type { Detection } from "@/data/types";

const MapView = dynamic(() => import("@/components/MapView"), { ssr: false });

type ViewMode = "map" | "list";

const damageTypes: Array<Detection["damage_type"] | "all"> = [
  "all",
  "pothole",
  "crack",
  "rut",
  "debris"
];

const statusLabels: Record<string, string> = {
  new: "New",
  reviewed: "Reviewed",
  repaired: "Repaired"
};

const severityLabel = (value: number) => {
  if (value >= 0.75) return { label: "High", className: "sev-high" };
  if (value >= 0.45) return { label: "Med", className: "sev-med" };
  return { label: "Low", className: "sev-low" };
};

const formatDate = (value: string) =>
  new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit"
  });

export default function Dashboard() {
  const [view, setView] = useState<ViewMode>("map");
  const [typeFilter, setTypeFilter] = useState<string>("all");
  const [severityFilter, setSeverityFilter] = useState(0.3);
  const [confidenceFilter, setConfidenceFilter] = useState(0.6);
  const [startDate, setStartDate] = useState<string>("");
  const [endDate, setEndDate] = useState<string>("");
  const [selectedId, setSelectedId] = useState<string>(mockDetections[0]?.id ?? "");

  const filtered = useMemo(() => {
    return mockDetections.filter((item) => {
      if (typeFilter !== "all" && item.damage_type !== typeFilter) return false;
      if (item.severity < severityFilter) return false;
      if (item.confidence < confidenceFilter) return false;
      if (startDate) {
        const start = new Date(startDate).getTime();
        if (new Date(item.captured_at).getTime() < start) return false;
      }
      if (endDate) {
        const end = new Date(endDate).getTime();
        if (new Date(item.captured_at).getTime() > end) return false;
      }
      return true;
    });
  }, [typeFilter, severityFilter, confidenceFilter, startDate, endDate]);

  const selected = useMemo(() => {
    return (
      filtered.find((item) => item.id === selectedId) ??
      filtered[0] ??
      mockDetections[0]
    );
  }, [selectedId, filtered]);

  const summary = useMemo(() => {
    const total = filtered.length;
    const avgSeverity = total
      ? filtered.reduce((acc, item) => acc + item.severity, 0) / total
      : 0;
    const avgConfidence = total
      ? filtered.reduce((acc, item) => acc + item.confidence, 0) / total
      : 0;
    return { total, avgSeverity, avgConfidence };
  }, [filtered]);

  const exportCsv = () => {
    if (!filtered.length) return;
    const headers = [
      "id",
      "captured_at",
      "lat",
      "lng",
      "damage_type",
      "confidence",
      "severity",
      "source_id",
      "image_path"
    ];
    const rows = filtered.map((item) => [
      item.id,
      item.captured_at,
      item.lat,
      item.lng,
      item.damage_type,
      item.confidence,
      item.severity,
      item.source_id,
      item.image_path
    ]);
    const csv = [headers, ...rows].map((row) => row.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "nodesense-detections.csv";
    link.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      <aside className="rail">
        <div className="rail-logo">NS</div>
        <div className="rail-nav">
          {[
            "Map",
            "Alerts",
            "Routes",
            "Uploads",
            "Settings"
          ].map((item, index) => (
            <div
              key={item}
              className={`rail-button ${index === 0 ? "active" : ""}`}
              title={item}
            >
              <span>{item.slice(0, 1)}</span>
            </div>
          ))}
        </div>
        <div className="rail-button">?</div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="brand">
            <div>
              <h1>NodeSense</h1>
              <span>Road damage intelligence console</span>
            </div>
          </div>

          <div className="search">
            <span aria-hidden="true">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                <circle cx="11" cy="11" r="7" stroke="#64748b" strokeWidth="2" />
                <path d="M16.5 16.5L21 21" stroke="#64748b" strokeWidth="2" strokeLinecap="round" />
              </svg>
            </span>
            <input placeholder="Search by street, source, or ID" />
          </div>

          <div className="top-actions">
            <div className="pill">Jan 31, 2026</div>
            <div className="toggle-group">
              <button
                className={view === "map" ? "active" : ""}
                onClick={() => setView("map")}
              >
                Map
              </button>
              <button
                className={view === "list" ? "active" : ""}
                onClick={() => setView("list")}
              >
                List
              </button>
            </div>
          </div>
        </header>

        <section className="grid">
          <div className="card">
            <h3>Filters</h3>
            <div className="filter-group">
              <label>Damage type</label>
              <select
                className="select"
                value={typeFilter}
                onChange={(event) => setTypeFilter(event.target.value)}
              >
                {damageTypes.map((type) => (
                  <option key={type} value={type}>
                    {type === "all" ? "All types" : type}
                  </option>
                ))}
              </select>
            </div>
            <div className="filter-group">
              <label>Severity ≥ {severityFilter.toFixed(2)}</label>
              <input
                className="range"
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={severityFilter}
                onChange={(event) => setSeverityFilter(Number(event.target.value))}
              />
            </div>
            <div className="filter-group">
              <label>Confidence ≥ {confidenceFilter.toFixed(2)}</label>
              <input
                className="range"
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={confidenceFilter}
                onChange={(event) => setConfidenceFilter(Number(event.target.value))}
              />
            </div>
            <div className="filter-group">
              <label>Date range</label>
              <input
                className="date-input"
                type="date"
                value={startDate}
                onChange={(event) => setStartDate(event.target.value)}
              />
              <input
                className="date-input"
                type="date"
                value={endDate}
                onChange={(event) => setEndDate(event.target.value)}
              />
            </div>

            <div className="filter-summary">
              <span>Active detections: {summary.total}</span>
              <span>Avg severity: {summary.avgSeverity.toFixed(2)}</span>
              <span>Avg confidence: {summary.avgConfidence.toFixed(2)}</span>
            </div>

            <h3>Latest detections</h3>
            <div className="list">
              {filtered.map((item) => {
                const sev = severityLabel(item.severity);
                return (
                  <div
                    key={item.id}
                    className={`list-item ${
                      selected?.id === item.id ? "active" : ""
                    }`}
                    onClick={() => setSelectedId(item.id)}
                  >
                    <strong>{item.damage_type}</strong>
                    <span>{formatDate(item.captured_at)}</span>
                    <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                      <span className={`badge ${sev.className}`}>Severity {sev.label}</span>
                      <span className="badge">{statusLabels[item.status ?? "new"]}</span>
                    </div>
                  </div>
                );
              })}
              {!filtered.length && <span>No detections match the filters.</span>}
            </div>

            <button className="cta secondary" onClick={exportCsv}>
              Export CSV
            </button>
          </div>

          <div className="card map-card">
            {view === "map" ? (
              <>
                <div className="map-overlay">
                  <span>Clustered detections</span>
                  <strong>{filtered.length}</strong>
                </div>
                <MapView
                  data={filtered.length ? filtered : mockDetections}
                  selectedId={selected?.id}
                  onSelect={(id) => setSelectedId(id)}
                />
                <div className="legend">
                  <span>
                    <i style={{ background: "#22c55e" }} /> Low
                  </span>
                  <span>
                    <i style={{ background: "#38bdf8" }} /> Medium
                  </span>
                  <span>
                    <i style={{ background: "#fb923c" }} /> High
                  </span>
                  <span>
                    <i style={{ background: "#ef4444" }} /> Critical
                  </span>
                </div>
              </>
            ) : (
              <div className="list">
                {filtered.map((item) => {
                  const sev = severityLabel(item.severity);
                  return (
                    <div
                      key={item.id}
                      className={`list-item ${
                        selected?.id === item.id ? "active" : ""
                      }`}
                      onClick={() => setSelectedId(item.id)}
                    >
                      <strong>{item.damage_type}</strong>
                      <span>
                        {item.lat.toFixed(4)}, {item.lng.toFixed(4)}
                      </span>
                      <span>{formatDate(item.captured_at)}</span>
                      <span className={`badge ${sev.className}`}>Severity {sev.label}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <div className="card details-card">
            <div>
              <h3>Detection detail</h3>
              <span style={{ color: "var(--muted)", fontSize: "0.85rem" }}>
                Engineer review panel
              </span>
            </div>

            {selected ? (
              <>
                <div className="preview">
                  <Image
                    src={selected.image_path}
                    alt="Detection preview"
                    fill
                    style={{ objectFit: "cover" }}
                    unoptimized
                  />
                  <div
                    className="bbox"
                    style={{
                      left: `${selected.bbox.x1 * 100}%`,
                      top: `${selected.bbox.y1 * 100}%`,
                      width: `${(selected.bbox.x2 - selected.bbox.x1) * 100}%`,
                      height: `${(selected.bbox.y2 - selected.bbox.y1) * 100}%`
                    }}
                  />
                </div>
                <div className="detail-grid">
                  <span>
                    <strong>Type:</strong> {selected.damage_type}
                  </span>
                  <span>
                    <strong>Confidence:</strong> {selected.confidence.toFixed(2)}
                  </span>
                  <span>
                    <strong>Severity:</strong> {selected.severity.toFixed(2)}
                  </span>
                  <span>
                    <strong>Captured:</strong> {formatDate(selected.captured_at)}
                  </span>
                  <span>
                    <strong>Source:</strong> {selected.source_id}
                  </span>
                </div>

                <div className="stat-row">
                  <span>Status</span>
                  <strong>{statusLabels[selected.status ?? "new"]}</strong>
                </div>

                <div style={{ display: "flex", gap: "10px" }}>
                  <button className="cta">Mark repaired</button>
                  <button className="cta secondary">Flag for QA</button>
                </div>
              </>
            ) : (
              <span>Select a detection to review.</span>
            )}
          </div>
        </section>
      </main>
    </div>
  );
}
