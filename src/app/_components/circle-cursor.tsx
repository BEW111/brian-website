"use client";

import { useState, useEffect } from "react";

export function CircleCursor() {
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [isHoveringLink, setIsHoveringLink] = useState(false);
  const [isMobile, setIsMobile] = useState(false);

  useEffect(() => {
    const mobileQuery = window.matchMedia(
      "(hover: none) and (pointer: coarse)"
    );
    setIsMobile(mobileQuery.matches);
  }, []);

  useEffect(() => {
    if (isMobile) return;

    const moveCursor = (e: MouseEvent) => {
      setPosition({ x: e.clientX, y: e.clientY });
    };

    const handleMouseOver = (e: MouseEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("a")) {
        setIsHoveringLink(true);
      }
    };

    const handleMouseOut = (e: MouseEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("a")) {
        setIsHoveringLink(false);
      }
    };

    const handleMouseClick = (e: MouseEvent) => {
      if (e.target instanceof HTMLElement && e.target.closest("a")) {
        setIsHoveringLink(false);
      }
    };

    window.addEventListener("mousemove", moveCursor);
    window.addEventListener("mouseover", handleMouseOver);
    window.addEventListener("mouseout", handleMouseOut);
    window.addEventListener("click", handleMouseClick);
    return () => {
      window.removeEventListener("mousemove", moveCursor);
      window.removeEventListener("mouseover", handleMouseOver);
      window.removeEventListener("mouseout", handleMouseOut);
      window.removeEventListener("click", handleMouseClick);
    };
  }, [isMobile]);

  if (isMobile) return null;

  return (
    <div
      style={{
        left: position.x,
        top: position.y,
        transform: "translate(-50%, -50%)",
      }}
      className={`fixed w-6 h-6 rounded-full pointer-events-none z-50 mix-blend-difference ${
        isHoveringLink ? "bg-sky-900" : "bg-sky-100"
      }`}
    />
  );
}
