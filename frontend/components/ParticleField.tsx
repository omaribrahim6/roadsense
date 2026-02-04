"use client";

import { useEffect, useRef } from "react";

type Point = {
  x: number;
  y: number;
  offset: number;
};

export default function ParticleField() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const wrapperRef = useRef<HTMLDivElement | null>(null);
  const animationRef = useRef<number | null>(null);
  const pointsRef = useRef<Point[]>([]);
  const reducedMotionRef = useRef(false);
  const sizeRef = useRef({ width: 0, height: 0, dpr: 1 });

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrapper = wrapperRef.current;
    if (!canvas || !wrapper) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const media = window.matchMedia("(prefers-reduced-motion: reduce)");
    reducedMotionRef.current = media.matches;

    const handleReduceMotion = (event: MediaQueryListEvent) => {
      reducedMotionRef.current = event.matches;
    };

    if (media.addEventListener) {
      media.addEventListener("change", handleReduceMotion);
    } else {
      media.addListener(handleReduceMotion);
    }

    const buildGrid = () => {
      const rect = wrapper.getBoundingClientRect();
      const width = Math.max(rect.width, 1);
      const height = Math.max(rect.height, 1);
      const dpr = Math.min(window.devicePixelRatio || 1, 2);

      canvas.width = width * dpr;
      canvas.height = height * dpr;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      sizeRef.current = { width, height, dpr };

      const spacing = Math.max(26, Math.min(width, height) / 16);
      const cols = Math.max(12, Math.floor(width / spacing));
      const rows = Math.max(8, Math.floor(height / spacing));

      const points: Point[] = [];
      const startX = -((cols - 1) * spacing) / 2;
      const startY = -((rows - 1) * spacing) / 2;

      for (let row = 0; row < rows; row += 1) {
        for (let col = 0; col < cols; col += 1) {
          points.push({
            x: startX + col * spacing,
            y: startY + row * spacing,
            offset: Math.random() * Math.PI * 2
          });
        }
      }

      pointsRef.current = points;
    };

    const render = (time: number) => {
      const { width, height, dpr } = sizeRef.current;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);

      const t = reducedMotionRef.current ? 0 : time * 0.0006;
      const amplitude = Math.min(width, height) * 0.12;
      const frequency = 0.025;
      const perspective = 380;

      for (const point of pointsRef.current) {
        const wave =
          Math.sin(point.x * frequency + t + point.offset) +
          Math.cos(point.y * frequency - t * 1.1 + point.offset * 0.5);
        const z = wave * amplitude;
        const scale = perspective / (perspective - z);
        const x = point.x * scale + width / 2;
        const y = point.y * scale + height / 2;
        const radius = 1.3 * scale;
        const alpha = Math.min(0.9, 0.25 + scale * 0.45);

        ctx.beginPath();
        ctx.fillStyle = `rgba(233, 241, 255, ${alpha})`;
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fill();
      }

      animationRef.current = window.requestAnimationFrame(render);
    };

    buildGrid();
    animationRef.current = window.requestAnimationFrame(render);

    const handleResize = () => {
      buildGrid();
    };

    window.addEventListener("resize", handleResize);

    return () => {
      if (animationRef.current) window.cancelAnimationFrame(animationRef.current);
      window.removeEventListener("resize", handleResize);
      if (media.removeEventListener) {
        media.removeEventListener("change", handleReduceMotion);
      } else {
        media.removeListener(handleReduceMotion);
      }
    };
  }, []);

  return (
    <div className="particle-field" ref={wrapperRef} aria-hidden="true">
      <canvas ref={canvasRef} className="particle-canvas" />
    </div>
  );
}
