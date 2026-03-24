import { useState, useEffect, useMemo } from "react";
import { api } from "../api";
import MoleculeViewer3D from "../components/MoleculeViewer3D";
import Molecule3DModal from "../components/Molecule3DModal";
import ReactionPicker from "../components/ReactionPicker";
import LipinskiBadge from "../components/LipinskiBadge";
import { ELEMENT_MAP } from "../data/elements";
import { Maximize2 } from "lucide-react";

function elemColor(sym) {
  return ELEMENT_MAP[sym]?.color || "#6b7280";
}

export default function SupervisorPage({ presets }) {
  const [config, setConfig] = useState({
    T: 20, hidden_dim: 64, seed: 42, max_retries: 3, max_backtracks: 5,
  });
  const [reaction, setReaction] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [timelineIdx, setTimelineIdx] = useState(-1);
  const [show3DModal, setShow3DModal] = useState(false);
  const [noiseSchedule, setNoiseSchedule] = useState(null);

  useEffect(() => {
    if (presets && !reaction) {
      setReaction({ reactants: presets.reactions[0].reactants, label: presets.reactions[0].name });
    }
  }, [presets, reaction]);

  useEffect(() => {
    api.getNoiseSchedule(config.T).then(setNoiseSchedule).catch(() => {});
  }, [config.T]);

  const run = async () => {
    const reactants = reaction?.reactants || presets?.reactions[0]?.reactants;
    if (!reactants) return;
    setLoading(true); setError(null); setResult(null); setTimelineIdx(-1);
    try {
      const data = await api.runSupervisor({ ...config, reactants });
      setResult(data);
      if (data.intermediates?.length > 0) setTimelineIdx(data.intermediates.length - 1);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const set = (k, v) => setConfig((p) => ({ ...p, [k]: v }));

  const currentSnapshot = useMemo(() => {
    if (!result) return null;
    const ints = result.intermediates || [];
    if (timelineIdx >= 0 && timelineIdx < ints.length) return ints[timelineIdx];
    return { t: 0, atoms: result.product.atoms, adjacency: result.product.adjacency, total_mass: result.product.total_mass, total_charge: result.product.total_charge };
  }, [result, timelineIdx]);

  const currentNoiseLevel = useMemo(() => {
    if (!noiseSchedule || !currentSnapshot) return null;
    const entry = noiseSchedule.schedule.find((s) => s.t === currentSnapshot.t);
    return entry ? entry.alpha_bar : currentSnapshot.t === 0 ? 1.0 : null;
  }, [noiseSchedule, currentSnapshot]);

  const stepCounts = useMemo(() => {
    if (!result) return null;
    const c = { commit: 0, corrected: 0, backtrack: 0, skip: 0 };
    result.step_log.forEach((s) => { if (c[s.action] !== undefined) c[s.action]++; });
    return c;
  }, [result]);

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Supervisor</h1>
      <p className="text-[15px] mb-3" style={{ color: "var(--text-secondary)" }}>
        Run the constrained diffusion loop and watch molecules get built step by step.
      </p>
      <p className="text-[13px] mb-8 max-w-[750px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
        A graph neural network proposes molecular structures through a diffusion process. At each
        denoising step the constraint engine checks valency limits, charge conservation, and bond
        validity, then fixes anything that breaks the rules before moving on. You can scrub through
        the timeline to see how the molecule changed at each step. Pick a reaction and tweak the
        parameters below to try it out.
      </p>

      {presets && <ReactionPicker presets={presets} value={reaction} onChange={setReaction} />}

      <div className="flex items-end gap-4 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
        {[
          { key: "T", label: "Timesteps", min: 2, max: 100 },
          { key: "hidden_dim", label: "Hidden dim", min: 8, max: 256 },
          { key: "seed", label: "Seed", min: 0, max: 9999 },
          { key: "max_retries", label: "Retries", min: 0, max: 10 },
          { key: "max_backtracks", label: "Backtracks", min: 1, max: 50 },
        ].map(({ key, label, min, max }) => (
          <div key={key}>
            <label className="text-[13px] block mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</label>
            <input
              type="number" value={config[key]}
              onChange={(e) => set(key, Number(e.target.value))}
              min={min} max={max}
              className="w-24 rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
              style={{ background: "var(--bg-input)", border: "1px solid var(--border)" }}
            />
          </div>
        ))}
        <button
          onClick={run} disabled={loading || (!reaction && !presets)}
          className="px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors"
        >
          {loading ? "Running..." : "Run"}
        </button>
      </div>

      {error && <p className="text-red-400 text-[14px] mb-6">{error}</p>}

      {result && (
        <>
          <div className="flex gap-8 mb-8 text-[14px]">
            <Stat label="Result" value={result.success ? "Valid" : "Invalid"} color={result.success ? "text-green-400" : "text-red-400"} />
            <Stat label="Time" value={`${result.wall_time_s}s`} />
            <Stat label="Steps" value={result.step_log.length} />
            <Stat label="Backtracks" value={result.total_backtracks} />
            <Stat label="Corrections" value={result.total_corrections} />
          </div>

          {result.intermediates?.length > 0 && (
            <div className="mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
              <div className="flex items-center justify-between mb-3">
                <span className="text-[14px] text-white">
                  Diffusion timeline
                </span>
                <div className="flex gap-4 text-[13px]" style={{ color: "var(--text-muted)" }}>
                  <span>Step {timelineIdx + 1}/{result.intermediates.length}</span>
                  {currentSnapshot && <span>t={currentSnapshot.t}</span>}
                  {currentNoiseLevel !== null && <span>Signal {(currentNoiseLevel * 100).toFixed(0)}%</span>}
                </div>
              </div>
              <input
                type="range" min={0} max={result.intermediates.length - 1}
                value={timelineIdx >= 0 ? timelineIdx : 0}
                onChange={(e) => setTimelineIdx(Number(e.target.value))}
                className="w-full cursor-pointer"
                style={{ background: `linear-gradient(to right, #22c55e ${(timelineIdx / Math.max(1, result.intermediates.length - 1)) * 100}%, #333 ${(timelineIdx / Math.max(1, result.intermediates.length - 1)) * 100}%)` }}
              />
              <div className="flex justify-between mt-1 text-[11px] font-mono" style={{ color: "var(--text-muted)" }}>
                <span>t={result.intermediates[0]?.t}</span>
                <span>t={result.intermediates[result.intermediates.length - 1]?.t}</span>
              </div>
            </div>
          )}

          <div className="grid grid-cols-12 gap-6 mb-8">
            <div className="col-span-5 rounded-lg overflow-hidden" style={{ background: "var(--bg-raised)" }}>
              <div className="px-5 pt-4 pb-1 flex items-center justify-between">
                <span className="text-[13px] text-white">
                  {currentSnapshot?.t > 0 ? `t=${currentSnapshot.t}` : "Product"}
                </span>
                {currentSnapshot && (
                  <button
                    onClick={() => setShow3DModal(true)}
                    className="flex items-center gap-1 text-[11px] hover:text-white transition-colors"
                    style={{ color: "var(--text-muted)" }}
                  >
                    <Maximize2 size={12} /> Expand
                  </button>
                )}
              </div>
              {currentSnapshot && (
                <div className="cursor-pointer" onClick={() => setShow3DModal(true)} title="Click to enlarge">
                  <MoleculeViewer3D atoms={currentSnapshot.atoms} adjacency={currentSnapshot.adjacency} height={340} />
                </div>
              )}
              {currentSnapshot && (
                <div className="px-5 py-3 space-y-2">
                  <div className="flex gap-5 text-[13px]" style={{ color: "var(--text-muted)" }}>
                    <span>{currentSnapshot.total_mass?.toFixed(2)} u</span>
                    <span>charge {currentSnapshot.total_charge > 0 ? "+" : ""}{currentSnapshot.total_charge}</span>
                    <span>{currentSnapshot.atoms.length} atoms</span>
                  </div>
                  {result?.product?.lipinski && currentSnapshot.t === 0 && (
                    <LipinskiBadge lipinski={result.product.lipinski} />
                  )}
                </div>
              )}
            </div>

            <div className="col-span-3">
              <span className="text-[13px] text-white block mb-3">Actions</span>
              {stepCounts && (
                <div className="space-y-4">
                  {[
                    { label: "Committed", count: stepCounts.commit, color: "#22c55e" },
                    { label: "Corrected", count: stepCounts.corrected, color: "#eab308" },
                    { label: "Backtracked", count: stepCounts.backtrack, color: "#ef4444" },
                    { label: "Skipped", count: stepCounts.skip, color: "#737373" },
                  ].map(({ label, count, color }) => (
                    <div key={label}>
                      <div className="flex justify-between text-[13px] mb-1">
                        <span style={{ color: "var(--text-secondary)" }}>{label}</span>
                        <span className="font-mono text-white">{count}</span>
                      </div>
                      <div className="h-1 rounded-full" style={{ background: "var(--border)" }}>
                        <div className="h-full rounded-full" style={{ width: `${(count / (result.step_log.length || 1)) * 100}%`, background: color }} />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="col-span-4">
              <span className="text-[13px] text-white block mb-3">Atoms</span>
              {currentSnapshot && (
                <div className="space-y-1 max-h-[340px] overflow-y-auto">
                  {currentSnapshot.atoms.map((a, i) => {
                    const ok = a.total_bonds <= a.effective_valency;
                    return (
                      <div key={i} className="flex items-center justify-between py-1.5 px-3 rounded-md text-[13px]" style={{ background: ok ? "transparent" : "rgba(239,68,68,0.06)" }}>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-medium" style={{ color: elemColor(a.element) }}>{a.element}</span>
                          <span className="font-mono" style={{ color: "var(--text-muted)" }}>{a.bonds}b +{a.implicit_h}H</span>
                        </div>
                        <span className={`text-[12px] font-mono ${ok ? "text-green-500" : "text-red-400"}`}>{ok ? "ok" : "over"}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>

          <div className="mb-8">
            <span className="text-[13px] text-white block mb-3">
              Step log ({result.step_log.length})
            </span>
            <div className="space-y-px max-h-[280px] overflow-y-auto rounded-lg" style={{ background: "var(--border)" }}>
              {result.step_log.map((s, i) => {
                const active = currentSnapshot && s.t === currentSnapshot.t && s.action !== "backtrack";
                return (
                  <div key={i} className={`flex items-center gap-4 px-4 py-2 text-[13px] ${active ? "ring-1 ring-green-500/30" : ""}`} style={{ background: "var(--bg)" }}>
                    <span className={`w-2 h-2 rounded-full shrink-0 ${
                      s.action === "commit" ? "bg-green-500" :
                      s.action === "corrected" ? "bg-yellow-500" :
                      s.action === "backtrack" ? "bg-red-500" : "bg-neutral-600"
                    }`} />
                    <span className="font-mono w-12" style={{ color: "var(--text-muted)" }}>t={s.t}</span>
                    <span className="w-20 text-white">{s.action}</span>
                    <span className="flex-1 truncate" style={{ color: "var(--text-muted)" }}>
                      {s.constraint_sat ? "Satisfied" : s.violations?.[0] || s.constraint_reason}
                    </span>
                    <span className="font-mono" style={{ color: "var(--text-muted)" }}>{s.elapsed_ms}ms</span>
                  </div>
                );
              })}
            </div>
          </div>

          {result.final_check.violations.length > 0 && (
            <div className="rounded-lg p-5" style={{ background: "rgba(239,68,68,0.05)" }}>
              <span className="text-[14px] text-red-400 font-medium block mb-2">Violations</span>
              {result.final_check.violations.map((v, i) => (
                <p key={i} className="text-[13px] text-red-300 mb-1">{v}</p>
              ))}
            </div>
          )}
        </>
      )}

      {show3DModal && currentSnapshot && (
        <Molecule3DModal
          atoms={currentSnapshot.atoms}
          adjacency={currentSnapshot.adjacency}
          onClose={() => setShow3DModal(false)}
        />
      )}
    </div>
  );
}

function Stat({ label, value, color = "text-white" }) {
  return (
    <div>
      <span className="text-[12px] block mb-0.5" style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className={`text-[20px] font-semibold font-mono ${color}`}>{value}</span>
    </div>
  );
}
