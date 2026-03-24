import { useEffect } from "react";
import MoleculeViewer3D from "./MoleculeViewer3D";

export default function Molecule3DModal({ atoms, adjacency, onClose }) {
  useEffect(() => {
    const handler = (e) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-black/70" />
      <div
        className="relative w-[80vw] h-[75vh] max-w-[1000px] rounded-xl overflow-hidden"
        style={{ background: "var(--bg-raised)" }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="absolute top-4 left-5 z-10 flex items-center gap-3">
          <span className="text-[14px] text-white font-medium">3D Molecule Viewer</span>
          <span className="text-[12px]" style={{ color: "var(--text-muted)" }}>
            drag to rotate, scroll to zoom
          </span>
        </div>
        <button
          onClick={onClose}
          className="absolute top-4 right-5 z-10 text-[13px] px-3 py-1 rounded-md bg-white/10 text-white hover:bg-white/20 transition-colors"
        >
          Close
        </button>
        <MoleculeViewer3D
          atoms={atoms}
          adjacency={adjacency}
          height="100%"
          autoRotate={true}
        />
      </div>
    </div>
  );
}
