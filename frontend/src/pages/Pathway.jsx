import { useState } from "react";
import { api } from "../api";
import MoleculeViewer3D from "../components/MoleculeViewer3D";
import Molecule3DModal from "../components/Molecule3DModal";
import LipinskiBadge from "../components/LipinskiBadge";
import { parseEquation } from "../utils/chemParser";
import { Maximize2 } from "lucide-react";

export default function PathwayPage({ presets }) {
  const [steps, setSteps] = useState([
    { equation: "CH3Br + OH- -> CH3OH + Br-", useProducts: false },
  ]);
  const [config, setConfig] = useState({ hidden_dim: 64, seed: 42, T: 10 });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedStep, setExpandedStep] = useState(null);
  const [show3DModal, setShow3DModal] = useState(null);

  const addStep = () => {
    setSteps((s) => [...s, { equation: "", useProducts: true }]);
  };

  const removeStep = (i) => {
    if (steps.length <= 1) return;
    setSteps((s) => s.filter((_, idx) => idx !== i));
  };

  const updateStep = (i, field, val) => {
    setSteps((s) => s.map((st, idx) => (idx === i ? { ...st, [field]: val } : st)));
  };

  const run = async () => {
    setLoading(true); setError(null); setResult(null);
    try {
      const apiSteps = steps.map((step, i) => {
        if (i > 0 && step.useProducts) {
          return { reactants: null };
        }
        const parsed = parseEquation(step.equation);
        if (!parsed) throw new Error(`Could not parse step ${i + 1}: "${step.equation}"`);
        return { reactants: parsed.reactants };
      });

      const data = await api.runPathway({ steps: apiSteps, ...config });
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const inputStyle = { background: "var(--bg-input)", border: "1px solid var(--border)" };

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Pathways</h1>
      <p className="text-[15px] mb-3" style={{ color: "var(--text-secondary)" }}>
        Chain multiple reactions together where each step's product feeds into the next one.
      </p>
      <p className="text-[13px] mb-8 max-w-[750px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
        Most real synthesis takes more than one reaction. Here you can line up a sequence of
        steps and have the output of each one automatically become the input for the next.
        Every step gets generated and validated on its own, and at the end the tool checks
        that elements are conserved across the whole chain. Useful for planning multi-step
        organic synthesis or just checking if a proposed route makes sense.
      </p>

      <div className="mb-6">
        {steps.map((step, i) => (
          <div key={i} className="flex items-start gap-3 mb-3">
            <span className="text-[13px] font-mono text-white mt-2 w-6 shrink-0">{i + 1}.</span>
            <div className="flex-1">
              {i > 0 && (
                <label className="flex items-center gap-2 mb-2 text-[13px] cursor-pointer" style={{ color: "var(--text-secondary)" }}>
                  <input
                    type="checkbox" checked={step.useProducts}
                    onChange={(e) => updateStep(i, "useProducts", e.target.checked)}
                    className="accent-green-500"
                  />
                  Use previous step's products as reactants
                </label>
              )}
              {(!step.useProducts || i === 0) && (
                <input
                  value={step.equation}
                  onChange={(e) => updateStep(i, "equation", e.target.value)}
                  placeholder="CH3Br + OH- -> CH3OH + Br-"
                  className="w-full rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
                  style={inputStyle}
                />
              )}
            </div>
            {steps.length > 1 && (
              <button onClick={() => removeStep(i)}
                className="mt-2 text-[12px] text-neutral-600 hover:text-red-400 transition-colors">
                remove
              </button>
            )}
          </div>
        ))}
        <button onClick={addStep} className="text-[13px] ml-9 hover:text-green-400 transition-colors" style={{ color: "var(--text-muted)" }}>
          + Add step
        </button>
      </div>

      <div className="flex items-end gap-4 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
        {[
          { key: "hidden_dim", label: "Hidden dim", min: 8, max: 256 },
          { key: "seed", label: "Seed", min: 0, max: 9999 },
          { key: "T", label: "Timesteps", min: 5, max: 50 },
        ].map(({ key, label, min, max }) => (
          <div key={key}>
            <label className="text-[13px] block mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</label>
            <input
              type="number" value={config[key]}
              onChange={(e) => setConfig((p) => ({ ...p, [key]: Number(e.target.value) }))}
              min={min} max={max}
              className="w-24 rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
              style={inputStyle}
            />
          </div>
        ))}
        <button
          onClick={run} disabled={loading}
          className="px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors"
        >
          {loading ? "Running..." : "Run pathway"}
        </button>
      </div>

      {error && <p className="text-red-400 text-[14px] mb-6">{error}</p>}

      {result && (
        <>
          <div className="flex items-center gap-3 mb-8">
            <span className={`text-[24px] font-semibold ${result.chain_valid ? "text-green-400" : "text-red-400"}`}>
              {result.chain_valid ? "Chain valid" : "Chain invalid"}
            </span>
            <span className="text-[15px]" style={{ color: "var(--text-muted)" }}>
              {result.total_steps} steps
            </span>
          </div>

          <div className="mb-10">
            <div className="flex items-stretch gap-0">
              {result.steps.map((step, i) => (
                <div key={i} className="flex items-stretch">
                  <button
                    onClick={() => setExpandedStep(expandedStep === i ? null : i)}
                    className={`px-5 py-4 rounded-lg transition-colors text-left ${
                      expandedStep === i ? "ring-1 ring-white/20" : ""
                    }`}
                    style={{ background: "var(--bg-raised)" }}
                  >
                    <div className="flex items-center gap-2 mb-1">
                      <span className="text-[12px] font-mono" style={{ color: "var(--text-muted)" }}>Step {i + 1}</span>
                      <span className={`w-2 h-2 rounded-full ${step.success ? "bg-green-500" : "bg-red-500"}`} />
                    </div>
                    <span className="text-[14px] text-white block">{step.product.name}</span>
                    <span className="text-[12px] font-mono" style={{ color: "var(--text-muted)" }}>
                      {step.product.total_mass?.toFixed(1)}u &middot; {step.product.atoms.length} atoms
                    </span>
                  </button>
                  {i < result.steps.length - 1 && (
                    <div className="flex items-center px-2">
                      <svg width="24" height="12" viewBox="0 0 24 12">
                        <path d="M0 6 L18 6 M14 2 L20 6 L14 10" stroke="#525252" strokeWidth="1.5" fill="none" />
                      </svg>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {expandedStep !== null && result.steps[expandedStep] && (
            <div className="grid grid-cols-12 gap-6 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
              <div className="col-span-5 rounded-lg overflow-hidden" style={{ background: "var(--bg-raised)" }}>
                <div className="px-4 pt-3 pb-1 flex items-center justify-between">
                  <span className="text-[13px] text-white">Step {expandedStep + 1} product</span>
                  <button
                    onClick={() => setShow3DModal(expandedStep)}
                    className="flex items-center gap-1 text-[11px] hover:text-white transition-colors"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <Maximize2 size={12} /> Expand
                  </button>
                </div>
                <div className="cursor-pointer" onClick={() => setShow3DModal(expandedStep)} title="Click to enlarge">
                  <MoleculeViewer3D
                    atoms={result.steps[expandedStep].product.atoms}
                    adjacency={result.steps[expandedStep].product.adjacency}
                    height={260}
                  />
                </div>
                <div className="px-4 py-3 space-y-2">
                  <div className="flex gap-4 text-[13px]" style={{ color: "var(--text-muted)" }}>
                    <span>{result.steps[expandedStep].product.total_mass?.toFixed(2)} u</span>
                    <span>charge {result.steps[expandedStep].product.total_charge}</span>
                  </div>
                  <LipinskiBadge lipinski={result.steps[expandedStep].product.lipinski} />
                </div>
              </div>
              <div className="col-span-7">
                <div className="flex gap-6 mb-4 text-[14px]">
                  <div>
                    <span className="text-[12px] block mb-0.5" style={{ color: "var(--text-muted)" }}>Result</span>
                    <span className={`font-semibold ${result.steps[expandedStep].success ? "text-green-400" : "text-red-400"}`}>
                      {result.steps[expandedStep].success ? "Valid" : "Invalid"}
                    </span>
                  </div>
                  <div>
                    <span className="text-[12px] block mb-0.5" style={{ color: "var(--text-muted)" }}>Backtracks</span>
                    <span className="font-mono text-white">{result.steps[expandedStep].backtracks}</span>
                  </div>
                  <div>
                    <span className="text-[12px] block mb-0.5" style={{ color: "var(--text-muted)" }}>Corrections</span>
                    <span className="font-mono text-white">{result.steps[expandedStep].corrections}</span>
                  </div>
                  <div>
                    <span className="text-[12px] block mb-0.5" style={{ color: "var(--text-muted)" }}>Time</span>
                    <span className="font-mono text-white">{result.steps[expandedStep].wall_time_s}s</span>
                  </div>
                </div>
                {result.steps[expandedStep].final_check.violations.length > 0 && (
                  <div className="rounded-lg p-4" style={{ background: "rgba(239,68,68,0.05)" }}>
                    <span className="text-[13px] text-red-400 font-medium block mb-2">Violations</span>
                    {result.steps[expandedStep].final_check.violations.map((v, vi) => (
                      <p key={vi} className="text-[13px] text-red-300 mb-1">{v}</p>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {!result.chain_valid && result.chain_check.violations.length > 0 && (
            <div className="rounded-lg p-5" style={{ background: "rgba(239,68,68,0.05)" }}>
              <span className="text-[14px] text-red-400 font-medium block mb-2">Chain-level violations</span>
              {result.chain_check.violations.map((v, i) => (
                <p key={i} className="text-[13px] text-red-300 mb-1">{v}</p>
              ))}
            </div>
          )}
        </>
      )}

      {show3DModal !== null && result?.steps?.[show3DModal] && (
        <Molecule3DModal
          atoms={result.steps[show3DModal].product.atoms}
          adjacency={result.steps[show3DModal].product.adjacency}
          onClose={() => setShow3DModal(null)}
        />
      )}
    </div>
  );
}
