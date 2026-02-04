import SiteHeader from "@/components/SiteHeader";

export default function AboutPage() {
  return (
    <div className="site">
      <SiteHeader active="about" />

      <main className="about-main">
        <section className="about-hero">
          <div>
            <span className="hero-label">About RoadSense</span>
            <h1>RoadSense turns raw road footage into repair-ready decisions.</h1>
            <p>
              The platform ingests dashcam video, runs detection and severity
              scoring, and publishes a live map that maintenance teams can trust.
              Every issue is traceable to an image, a location, and a timestamp.
            </p>
          </div>
          <div className="about-highlights">
            <div>
              <span className="highlight-label">Core focus</span>
              <strong>Road damage intelligence</strong>
            </div>
            <div>
              <span className="highlight-label">Outputs</span>
              <strong>Mapped detections + severity</strong>
            </div>
            <div>
              <span className="highlight-label">Users</span>
              <strong>DOTs, fleets, and ops teams</strong>
            </div>
          </div>
        </section>

        <section className="about-section">
          <div className="section-title">
            <h2>System architecture</h2>
            <p>
              A streamlined pipeline connects capture, AI inference, and the
              operational console used by engineers and planners.
            </p>
          </div>
          <div className="diagram-frame" aria-hidden="true">
            <svg viewBox="0 0 1020 520" className="architecture-diagram">
              <defs>
                <linearGradient id="panel" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#0f172a" />
                  <stop offset="100%" stopColor="#0a0f1a" />
                </linearGradient>
                <linearGradient id="node" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stopColor="#0b111b" />
                  <stop offset="100%" stopColor="#090f18" />
                </linearGradient>
                <linearGradient id="accent" x1="0" y1="0" x2="1" y2="0">
                  <stop offset="0%" stopColor="#8cf0ff" />
                  <stop offset="100%" stopColor="#6bd0ff" />
                </linearGradient>
                <marker
                  id="arrow"
                  markerWidth="10"
                  markerHeight="10"
                  refX="8"
                  refY="3"
                  orient="auto"
                  markerUnits="strokeWidth"
                >
                  <path d="M0 0 L8 3 L0 6" fill="#8cf0ff" />
                </marker>
              </defs>

              <rect x="30" y="60" width="200" height="230" rx="18" fill="url(#panel)" stroke="#1f2a3a" />
              <text x="50" y="90" fill="#e5e7eb" fontSize="14" fontFamily="Sora, sans-serif">
                Input Sources
              </text>
              <rect x="50" y="115" width="160" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="130" y="137" textAnchor="middle" fill="#e2e8f0" fontSize="12">Dashcam Video</text>
              <rect x="50" y="158" width="160" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="130" y="180" textAnchor="middle" fill="#e2e8f0" fontSize="12">Image Frames</text>
              <rect x="50" y="201" width="160" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="130" y="223" textAnchor="middle" fill="#e2e8f0" fontSize="12">GPX Track (opt)</text>

              <rect x="260" y="40" width="280" height="380" rx="20" fill="url(#panel)" stroke="#1f2a3a" />
              <text x="280" y="70" fill="#e5e7eb" fontSize="14" fontFamily="Sora, sans-serif">
                Python Backend Pipeline
              </text>
              <rect x="290" y="95" width="220" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="400" y="117" textAnchor="middle" fill="#e2e8f0" fontSize="12">Frame Extractor</text>
              <rect x="290" y="142" width="220" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="400" y="164" textAnchor="middle" fill="#e2e8f0" fontSize="12">YOLO Inference</text>
              <rect x="290" y="189" width="220" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="400" y="211" textAnchor="middle" fill="#e2e8f0" fontSize="12">Severity Calculator</text>
              <rect x="290" y="236" width="220" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="400" y="258" textAnchor="middle" fill="#e2e8f0" fontSize="12">GPS Resolver</text>
              <rect x="290" y="283" width="220" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="400" y="305" textAnchor="middle" fill="#e2e8f0" fontSize="12">Deduplication</text>
              <rect x="290" y="330" width="220" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="400" y="352" textAnchor="middle" fill="#e2e8f0" fontSize="12">Supabase Uploader</text>

              <rect x="580" y="60" width="190" height="220" rx="18" fill="url(#panel)" stroke="#1f2a3a" />
              <text x="600" y="90" fill="#e5e7eb" fontSize="14" fontFamily="Sora, sans-serif">
                Supabase
              </text>
              <rect x="600" y="120" width="150" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="675" y="142" textAnchor="middle" fill="#e2e8f0" fontSize="12">Storage Bucket</text>
              <rect x="600" y="163" width="150" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="675" y="185" textAnchor="middle" fill="#e2e8f0" fontSize="12">PostgreSQL</text>

              <rect x="800" y="60" width="190" height="220" rx="18" fill="url(#panel)" stroke="#1f2a3a" />
              <text x="820" y="90" fill="#e5e7eb" fontSize="14" fontFamily="Sora, sans-serif">
                Next.js Frontend
              </text>
              <rect x="820" y="120" width="150" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="895" y="142" textAnchor="middle" fill="#e2e8f0" fontSize="12">Map Dashboard</text>
              <rect x="820" y="163" width="150" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="895" y="185" textAnchor="middle" fill="#e2e8f0" fontSize="12">Table View</text>
              <rect x="820" y="206" width="150" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="895" y="228" textAnchor="middle" fill="#e2e8f0" fontSize="12">Filter Controls</text>

              <rect x="580" y="320" width="190" height="120" rx="18" fill="url(#panel)" stroke="#1f2a3a" />
              <text x="600" y="350" fill="#e5e7eb" fontSize="14" fontFamily="Sora, sans-serif">
                Model Source
              </text>
              <rect x="600" y="370" width="150" height="34" rx="10" fill="url(#node)" stroke="#263247" />
              <text x="675" y="392" textAnchor="middle" fill="#e2e8f0" fontSize="12">HuggingFace Hub</text>

              <path d="M210 132 L290 112" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M210 175 L290 159" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M210 218 L290 253" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />

              <path d="M400 129 L400 142" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M400 176 L400 189" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M400 223 L400 236" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M400 270 L400 283" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M400 317 L400 330" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />

              <path d="M510 347 L600 154" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M750 154 L820 137" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />
              <path d="M750 180 L820 180" stroke="url(#accent)" strokeWidth="2" markerEnd="url(#arrow)" />

              <path
                d="M675 370 L510 159"
                stroke="#8cf0ff"
                strokeWidth="2"
                strokeDasharray="6 6"
                markerEnd="url(#arrow)"
              />
            </svg>
          </div>
        </section>

        <section className="about-section about-grid">
          <div className="about-card">
            <h3>Capture</h3>
            <p>
              Any fleet vehicle can collect footage. Optional GPX routes improve
              positioning when GPS data is available.
            </p>
          </div>
          <div className="about-card">
            <h3>Detect + Score</h3>
            <p>
              Each frame runs through YOLO inference, severity scoring, and
              deduplication to keep reports clean and actionable.
            </p>
          </div>
          <div className="about-card">
            <h3>Act</h3>
            <p>
              Teams open a live console to inspect detections, export data, and
              dispatch repairs based on the highest-risk segments.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
