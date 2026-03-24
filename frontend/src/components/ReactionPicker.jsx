import { useState } from "react";
import { parseEquation } from "../utils/chemParser";

export default function ReactionPicker({ presets, value, onChange }) {
  const [mode, setMode] = useState("preset");
  const [presetIdx, setPresetIdx] = useState(0);
  const [eqText, setEqText] = useState("CH3Br + OH- -> CH3OH + Br-");
  const [eqError, setEqError] = useState(null);

  const selectPreset = (idx) => {
    setPresetIdx(idx);
    const r = presets?.reactions[idx];
    if (r) onChange({ reactants: r.reactants, label: r.name });
  };

  const applyEquation = () => {
    const parsed = parseEquation(eqText);
    if (!parsed) {
      setEqError("Could not parse. Use: A + B -> C + D");
      return;
    }
    setEqError(null);
    onChange({ reactants: parsed.reactants, products: parsed.products, label: eqText });
  };

  const inputStyle = { background: "var(--bg-input)", border: "1px solid var(--border)" };

  return (
    <div className="mb-6">
      <div className="flex items-center gap-3 mb-3">
        <span className="text-[13px]" style={{ color: "var(--text-muted)" }}>Reaction</span>
        <button
          onClick={() => setMode("preset")}
          className={`px-2 py-1 rounded text-[12px] transition-colors ${mode === "preset" ? "text-white bg-white/[0.08]" : "text-neutral-500 hover:text-neutral-300"}`}
        >
          Preset
        </button>
        <button
          onClick={() => setMode("equation")}
          className={`px-2 py-1 rounded text-[12px] transition-colors ${mode === "equation" ? "text-white bg-white/[0.08]" : "text-neutral-500 hover:text-neutral-300"}`}
        >
          Custom equation
        </button>
      </div>

      {mode === "preset" ? (
        <select
          value={presetIdx}
          onChange={(e) => selectPreset(Number(e.target.value))}
          className="w-full max-w-md rounded-md px-3 py-2 text-[14px] text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
          style={inputStyle}
        >
          {presets?.reactions.map((r, i) => (
            <option key={i} value={i}>{r.name}</option>
          ))}
        </select>
      ) : (
        <div className="flex items-center gap-2 max-w-lg">
          <input
            value={eqText}
            onChange={(e) => { setEqText(e.target.value); setEqError(null); }}
            onKeyDown={(e) => { if (e.key === "Enter") applyEquation(); }}
            placeholder="CH3Br + OH- -> CH3OH + Br-"
            className="flex-1 rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
            style={inputStyle}
          />
          <button
            onClick={applyEquation}
            className="px-3 py-2 rounded-md text-[13px] text-white bg-white/[0.08] hover:bg-white/[0.12] transition-colors"
          >
            Apply
          </button>
        </div>
      )}
      {eqError && <p className="text-red-400 text-[12px] mt-1">{eqError}</p>}
    </div>
  );
}
