import Image from "next/image";
import Link from "next/link";

type SiteHeaderProps = {
  active?: "home" | "about";
};

export default function SiteHeader({ active }: SiteHeaderProps) {
  return (
    <header className="site-header">
      <Link className="site-logo" href="/">
        <Image src="/logo_white.png" alt="RoadSense" width={38} height={38} />
        <span>RoadSense</span>
      </Link>

      <nav className="site-nav">
        <div className="nav-pill">
          <Link className={active === "home" ? "nav-link active" : "nav-link"} href="/">
            Home
          </Link>
          <Link className={active === "about" ? "nav-link active" : "nav-link"} href="/about">
            About
          </Link>
          <Link className="nav-cta" href="/console">
            Open Console
            <span aria-hidden="true">-&gt;</span>
          </Link>
        </div>
      </nav>
    </header>
  );
}
