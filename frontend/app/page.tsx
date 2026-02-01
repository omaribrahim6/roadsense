import Image from "next/image";
import Link from "next/link";

export default function Page() {
  return (
    <div className="home">
      <header className="home-header">
        <div className="home-logo">RoadSense</div>
        <nav className="home-nav">
          <Link className="home-nav-link" href="/console">
            Open Console
          </Link>
        </nav>
      </header>

      <main>
        <section className="hero">
          <div className="hero-headline">
            <span className="hero-eyebrow">Road damage intelligence</span>
            <h1>Control your road network like never before.</h1>
            <p>
              RoadSense blends real-time fleet telemetry, computer vision, and
              automated work orders into a single platform that keeps cities and
              logistics teams ahead of infrastructure failures.
            </p>
          </div>

          <div className="hero-visual">
            <Image
              src="/HOME-HERO.png"
              alt="RoadSense fleet intelligence preview"
              fill
              priority
              sizes="100vw"
            />
          </div>
        </section>

        <section className="home-section">
          <div className="section-heading">
            <h2>Useful for every operation.</h2>
            <p>
              From DOTs to private fleets, RoadSense makes inspection data easy
              to action with a clean workflow and automated prioritization.
            </p>
          </div>
          <div className="stat-grid">
            {[
              {
                value: "20%",
                title: "less emergency patching",
                body: "Prevent surprises with automated deterioration alerts."
              },
              {
                value: "50%",
                title: "fewer repeat visits",
                body: "One unified map keeps crews aligned with the right fix."
              },
              {
                value: "30%",
                title: "lower fuel costs",
                body: "Optimize routing with daily confidence scoring."
              }
            ].map((stat) => (
              <div className="stat-card" key={stat.title}>
                <span>{stat.value}</span>
                <h3>{stat.title}</h3>
                <p>{stat.body}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="home-section diagram-section">
          <div className="section-heading">
            <h2>Built on live signals.</h2>
            <p>
              Every camera feed, sensor ping, and inspection route connects into
              a live diagnostic layer with automated risk scoring.
            </p>
          </div>
          <div className="diagram-grid">
            <div className="diagram-card">
              <div className="diagram-header">
                <h3>Signal Flow</h3>
                <span>Ingest -&gt; Verify -&gt; Dispatch</span>
              </div>
              <svg viewBox="0 0 240 120" aria-hidden="true">
                <defs>
                  <linearGradient id="flow" x1="0" y1="0" x2="1" y2="1">
                    <stop offset="0%" stopColor="#1c7f64" />
                    <stop offset="100%" stopColor="#e4a94b" />
                  </linearGradient>
                </defs>
                <path
                  d="M20 70 C60 30, 100 30, 140 70 S200 110, 220 70"
                  stroke="url(#flow)"
                  strokeWidth="4"
                  fill="none"
                />
                <circle cx="30" cy="70" r="8" fill="#111" />
                <circle cx="120" cy="50" r="10" fill="#1c7f64" />
                <circle cx="210" cy="70" r="8" fill="#111" />
              </svg>
              <p>
                Fleet video, accelerometer spikes, and GIS layers merge into a
                single quality score for every segment.
              </p>
            </div>
            <div className="diagram-card">
              <div className="diagram-header">
                <h3>Road Health Index</h3>
                <span>Last 90 days</span>
              </div>
              <svg viewBox="0 0 240 120" aria-hidden="true">
                <defs>
                  <linearGradient id="curve" x1="0" y1="0" x2="1" y2="0">
                    <stop offset="0%" stopColor="#111" />
                    <stop offset="100%" stopColor="#1c7f64" />
                  </linearGradient>
                </defs>
                <path
                  d="M10 90 C60 40, 110 80, 160 40 S220 30, 230 60"
                  stroke="url(#curve)"
                  strokeWidth="4"
                  fill="none"
                />
                <circle cx="10" cy="90" r="6" fill="#e4a94b" />
                <circle cx="120" cy="65" r="6" fill="#111" />
                <circle cx="230" cy="60" r="6" fill="#1c7f64" />
              </svg>
              <p>
                A single KPI blends severity, repeat detections, and traffic
                impact to rank repairs instantly.
              </p>
            </div>
            <div className="diagram-card">
              <div className="diagram-header">
                <h3>Coverage Map</h3>
                <span>Realtime fleet sweep</span>
              </div>
              <div className="coverage-grid" aria-hidden="true">
                {Array.from({ length: 36 }).map((_, index) => (
                  <span key={index} />
                ))}
              </div>
              <p>
                Visualize which blocks were inspected, which need revisits, and
                how coverage shifts by time of day.
              </p>
            </div>
          </div>
        </section>

        <section className="home-section proof-section">
          <div className="section-heading">
            <h2>Results that speak for themselves.</h2>
            <p>
              Teams using RoadSense respond faster, reduce backlog, and keep
              residents updated with transparent reporting.
            </p>
          </div>
          <div className="proof-grid">
            <div className="proof-card">
              <p>
                "Since rolling out RoadSense, our maintenance crew cut repeat
                patching by 27% in six months. The map view and severity alerts
                made it effortless to prioritize the right streets."
              </p>
              <div className="proof-person">
                <span className="avatar" aria-hidden="true">
                  JM
                </span>
                <div>
                  <strong>Jamal M.</strong>
                  <span>City Operations Lead</span>
                </div>
              </div>
            </div>
            <div className="proof-chart">
              <div>
                <h3>Operational efficiency</h3>
                <span>Downward trend in repeat work orders</span>
              </div>
              <div className="chart-lines" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <div className="chart-curve" aria-hidden="true" />
              <div className="chart-labels">
                <span>Week 1</span>
                <span>Week 6</span>
                <span>Week 12</span>
              </div>
            </div>
          </div>
        </section>

        <section className="home-section split-section">
          <div className="split-copy">
            <h2>Compatible with your workflows.</h2>
            <p>
              RoadSense integrates with existing GIS systems, fleet hardware,
              and inspection schedules without disrupting daily operations.
            </p>
            <ul className="feature-list">
              {[
                "Wide range of integrations with ArcGIS, Cityworks, and Cartegraph.",
                "Hardware compatibility for dash cams, phones, and dedicated rigs.",
                "API-first architecture to plug into your existing workflow.",
                "Fast setup with onboarding and calibration in under two weeks."
              ].map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </div>
          <div className="split-visual" aria-hidden="true">
            <div className="visual-card">
              <span>Live route</span>
              <strong>Fleet 08</strong>
              <div className="visual-track">
                <div />
                <div />
                <div />
              </div>
            </div>
            <div className="visual-card accent">
              <span>Priority alerts</span>
              <strong>14 high-severity</strong>
              <div className="visual-dots">
                {Array.from({ length: 12 }).map((_, index) => (
                  <span key={index} />
                ))}
              </div>
            </div>
          </div>
        </section>

        <section className="home-section">
          <div className="section-heading">
            <h2>News and updates.</h2>
            <p>Latest improvements across detection, routing, and reporting.</p>
          </div>
          <div className="news-grid">
            {[
              {
                title: "Integration with traffic volume APIs",
                body: "Blend road health with daily traffic counts for smarter prioritization.",
                date: "Jan 2026"
              },
              {
                title: "EV fleet support packs",
                body: "Optimized capture profiles for electric and hybrid vehicles.",
                date: "Dec 2025"
              },
              {
                title: "Instant resident reporting portal",
                body: "Share public-ready dashboards without exposing internal data.",
                date: "Nov 2025"
              }
            ].map((item) => (
              <article className="news-card" key={item.title}>
                <span>{item.date}</span>
                <h3>{item.title}</h3>
                <p>{item.body}</p>
              </article>
            ))}
          </div>
        </section>

        <section className="home-section faq-section">
          <div className="section-heading">
            <h2>Frequently Asked Questions.</h2>
            <p>
              Everything you need to know before rolling RoadSense into your
              operations.
            </p>
          </div>
          <div className="faq-grid">
            {[
              {
                question: "How quickly can we start seeing detections?",
                answer:
                  "Most teams are live in two weeks with calibration, setup, and training included."
              },
              {
                question: "What vehicle types are compatible?",
                answer:
                  "Any fleet vehicle with a forward-facing camera can be onboarded with a standard kit."
              },
              {
                question: "Can we integrate with existing work orders?",
                answer:
                  "Yes. RoadSense can push tickets into your current system via API."
              },
              {
                question: "How is data quality verified?",
                answer:
                  "Each detection is scored with a confidence model and can be reviewed in the console."
              },
              {
                question: "Do you support private road networks?",
                answer:
                  "Absolutely. Logistics hubs, campuses, and warehouses are common deployments."
              },
              {
                question: "Is there a free trial?",
                answer:
                  "We offer pilots with a limited fleet to demonstrate ROI before full rollout."
              }
            ].map((item) => (
              <div className="faq-item" key={item.question}>
                <h3>{item.question}</h3>
                <p>{item.answer}</p>
              </div>
            ))}
          </div>
        </section>
      </main>

      <section className="cta-banner">
        <div>
          <h2>Increase efficiency, reduce costs, and enhance safety.</h2>
          <p>
            Get a live demo of RoadSense and see how quickly your team can act
            on road issues.
          </p>
        </div>
        <div className="cta-actions">
          <Link className="primary-cta" href="/console">
            Get the demo
          </Link>
          <button className="ghost-cta" type="button">
            Talk to product
          </button>
        </div>
      </section>

      <footer className="home-footer">
        <div>
          <strong>RoadSense</strong>
          <span>Road damage intelligence platform</span>
        </div>
        <div className="footer-links">
          <span>Security</span>
          <span>Support</span>
          <span>API Docs</span>
          <span>Careers</span>
        </div>
      </footer>
    </div>
  );
}
