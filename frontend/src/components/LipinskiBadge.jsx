export default function LipinskiBadge({ lipinski }) {
  if (!lipinski) return null;
  const { mw, hbd, hba, heavy_atoms, passes_ro5 } = lipinski;
  return (
    <div className="flex items-center gap-4 text-[13px]">
      <span className={`font-medium ${passes_ro5 ? "text-green-400" : "text-yellow-400"}`}>
        {passes_ro5 ? "Passes" : "Fails"} Ro5
      </span>
      <span style={{ color: "var(--text-muted)" }}>
        MW {mw?.toFixed(0)}
      </span>
      <span style={{ color: "var(--text-muted)" }}>
        HBD {hbd}
      </span>
      <span style={{ color: "var(--text-muted)" }}>
        HBA {hba}
      </span>
      <span style={{ color: "var(--text-muted)" }}>
        Heavy {heavy_atoms}
      </span>
    </div>
  );
}
