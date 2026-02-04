import Link from "next/link";
import ParticleField from "@/components/ParticleField";
import SiteHeader from "@/components/SiteHeader";

export default function Page() {
  return (
    <div className="site">
      <SiteHeader active="home" />

      <main className="home-main">
        <section className="hero">
          <ParticleField />
          <div className="hero-content">
            <span className="hero-label">Road damage intelligence</span>
            <h1>ROADSENSE</h1>
            <p>
              Detect, prioritize, and resolve road damage from everyday dashcam
              footage. One clean console. Zero manual surveys.
            </p>
            <div className="hero-actions">
              <Link className="btn primary" href="/console">
                Open Console
              </Link>
              <Link className="btn ghost" href="/about">
                About RoadSense
              </Link>
            </div>
          </div>
        </section>

        <section className="home-summary">
          <div className="summary-card">
            <h2>What it does</h2>
            <p>
              Runs AI on dashcam video to find potholes, cracks, ruts, and debris,
              then maps every issue with severity scores and imagery so teams can
              act fast.
            </p>
          </div>
          <div className="summary-card">
            <h2>What it eliminates</h2>
            <p>
              Replaces slow windshield surveys, scattered spreadsheets, and
              reactive fixes with a single, consistent source of truth for road
              health.
            </p>
          </div>
        </section>
      </main>
    </div>
  );
}
