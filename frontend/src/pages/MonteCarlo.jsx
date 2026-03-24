import { useState, useEffect, useMemo } from "react";
import { api } from "../api";
import ReactionPicker from "../components/ReactionPicker";

export default function MonteCarloPage({ presets }) {
  const [config, setConfig] = useState({ n_samples: 50, hidden_dim: 32, T: 10 });
  const [reaction, setReaction] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    if (presets && !reaction) {
      setReaction({ reactants: presets.reactions[0].reactants, label: presets.reactions[0].name });
    }
  }, [presets, reaction]);

  const set = (k, v) => setConfig((p) => ({ ...p, [k]: v }));

  const run = async () => {
    const reactants = reaction?.reactants || presets?.reactions[0]?.reactants;
    if (!reactants) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.runMonteCarlo({ ...config, reactants });
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const filteredRuns = useMemo(() => {
    if (!result) return [];
    if (filter === "valid") return result.runs.filter((r) => r.valid);
    if (filter === "invalid") return result.runs.filter((r) => !r.valid);
    return result.runs;
  }, [result, filter]);

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Simulation</h1>
      <p className="text-[15px] mb-3" style={{ color: "var(--text-secondary)" }}>
        Run hundreds of generations at once and look at the results statistically.
      </p>
      <p className="text-[13px] mb-8 max-w-[750px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
        This generates a big batch of molecules using different random seeds and then aggregates
        everything so you can see patterns. You get the overall validity rate, a breakdown of
        which violations come up most often, the distribution of molecular masses, and how many
        molecules pass Lipinski drug-likeness filters. Good for stress-testing the model and
        catching edge cases you wouldn't see from individual runs.
      </p>

      {presets && <ReactionPicker presets={presets} value={reaction} onChange={setReaction} />}

      <div className="flex items-end gap-4 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
        {[
          { key: "n_samples", label: "Samples", min: 10, max: 500 },
          { key: "hidden_dim", label: "Hidden dim", min: 8, max: 128 },
          { key: "T", label: "Timesteps", min: 5, max: 50 },
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
          {loading ? `Running ${config.n_samples}...` : "Run"}
        </button>
      </div>

      {error && <p className="text-red-400 text-[14px] mb-6">{error}</p>}

      {result && (
        <>
          <div className="flex items-baseline gap-3 mb-10">
            <span className="text-[48px] font-semibold font-mono text-green-400">
              {result.summary.validity_rate}%
            </span>
            <span className="text-[15px]" style={{ color: "var(--text-secondary)" }}>
              valid across {result.n} generations
            </span>
          </div>

          <div className="grid grid-cols-4 gap-8 mb-10">
            <KV label="Avg backtracks" value={result.summary.avg_backtracks} />
            <KV label="Avg corrections" value={result.summary.avg_corrections} />
            <KV label="Avg mass" value={`${result.summary.avg_mass} u`} />
            <KV label="Lipinski pass rate" value={`${result.summary.lipinski_pass_rate}%`} />
          </div>

          <div className="grid grid-cols-2 gap-10 mb-10">
            <div>
              <span className="text-[14px] text-white block mb-4">Violation breakdown</span>
              <ViolationChart breakdown={result.summary.violation_breakdown} total={result.n} />
            </div>
            <div>
              <span className="text-[14px] text-white block mb-4">Mass distribution</span>
              <MassHistogram runs={result.runs} />
            </div>
          </div>

          <div className="mb-10">
            <span className="text-[14px] text-white block mb-4">Validity by seed</span>
            <ValidityStrip runs={result.runs} />
          </div>

          <div>
            <div className="flex items-center justify-between mb-4">
              <span className="text-[14px] text-white">
                All runs ({filteredRuns.length})
              </span>
              <div className="flex gap-2">
                {["all", "valid", "invalid"].map((f) => (
                  <button key={f} onClick={() => setFilter(f)}
                    className={`px-2 py-1 rounded text-[12px] transition-colors capitalize ${filter === f ? "text-white bg-white/[0.08]" : "text-neutral-500 hover:text-neutral-300"}`}>
                    {f}
                  </button>
                ))}
              </div>
            </div>
            <div className="max-h-[400px] overflow-y-auto">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="border-b" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                    <th className="text-left py-2 font-normal">Seed</th>
                    <th className="text-left py-2 font-normal">Status</th>
                    <th className="text-right py-2 font-normal">Atoms</th>
                    <th className="text-right py-2 font-normal">Mass</th>
                    <th className="text-right py-2 font-normal">BT</th>
                    <th className="text-right py-2 font-normal">Corr</th>
                    <th className="text-right py-2 font-normal">Time</th>
                    <th className="text-right py-2 font-normal">Ro5</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRuns.map((r) => (
                    <tr key={r.seed} className="border-b" style={{ borderColor: "var(--border)" }}>
                      <td className="py-1.5 font-mono">{r.seed}</td>
                      <td className="py-1.5">
                        <span className={`text-[12px] ${r.valid ? "text-green-400" : "text-red-400"}`}>
                          {r.valid ? "valid" : "invalid"}
                        </span>
                      </td>
                      <td className="py-1.5 text-right font-mono">{r.atom_count}</td>
                      <td className="py-1.5 text-right font-mono">{r.mass}</td>
                      <td className="py-1.5 text-right font-mono">{r.backtracks}</td>
                      <td className="py-1.5 text-right font-mono">{r.corrections}</td>
                      <td className="py-1.5 text-right font-mono">{r.wall_time_ms}ms</td>
                      <td className="py-1.5 text-right">
                        <span className={r.lipinski_pass ? "text-green-400" : "text-neutral-600"}>
                          {r.lipinski_pass ? "yes" : "no"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function KV({ label, value }) {
  return (
    <div>
      <span className="text-[12px] block mb-0.5" style={{ color: "var(--text-muted)" }}>{label}</span>
      <span className="text-[18px] font-semibold font-mono text-white">{value}</span>
    </div>
  );
}

function ViolationChart({ breakdown, total }) {
  const entries = Object.entries(breakdown).sort((a, b) => b[1] - a[1]);
  if (entries.length === 0) {
    return <p className="text-[13px]" style={{ color: "var(--text-muted)" }}>No violations recorded.</p>;
  }
  const maxCount = Math.max(...entries.map(([, c]) => c));
  return (
    <div className="space-y-2">
      {entries.map(([type, count]) => (
        <div key={type}>
          <div className="flex justify-between text-[13px] mb-1">
            <span style={{ color: "var(--text-secondary)" }}>{type}</span>
            <span className="font-mono text-white">{count}</span>
          </div>
          <div className="h-1.5 rounded-full" style={{ background: "var(--border)" }}>
            <div className="h-full bg-red-500 rounded-full" style={{ width: `${(count / maxCount) * 100}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

function MassHistogram({ runs }) {
  const masses = runs.map((r) => r.mass);
  const min = Math.min(...masses);
  const max = Math.max(...masses);
  const range = max - min || 1;
  const numBins = 12;
  const bins = Array(numBins).fill(0);
  masses.forEach((m) => {
    const idx = Math.min(numBins - 1, Math.floor(((m - min) / range) * numBins));
    bins[idx]++;
  });
  const maxBin = Math.max(...bins);

  const pad = { top: 8, right: 4, bottom: 20, left: 4 };
  const w = 400, h = 100;
  const bw = (w - pad.left - pad.right) / numBins;

  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="w-full">
      {bins.map((count, i) => {
        const barH = maxBin > 0 ? ((count / maxBin) * (h - pad.top - pad.bottom)) : 0;
        const x = pad.left + i * bw;
        return (
          <rect key={i} x={x + 1} y={h - pad.bottom - barH} width={bw - 2} height={barH}
            fill="#22c55e" fillOpacity={0.6} rx={1}>
            <title>{(min + (i / numBins) * range).toFixed(1)} - {(min + ((i + 1) / numBins) * range).toFixed(1)} u: {count} runs</title>
          </rect>
        );
      })}
      <text x={pad.left} y={h - 4} fill="#525252" fontSize={9} fontFamily="JetBrains Mono, monospace">
        {min.toFixed(0)}u
      </text>
      <text x={w - pad.right} y={h - 4} textAnchor="end" fill="#525252" fontSize={9} fontFamily="JetBrains Mono, monospace">
        {max.toFixed(0)}u
      </text>
    </svg>
  );
}

function ValidityStrip({ runs }) {
  return (
    <div className="flex gap-px flex-wrap">
      {runs.map((r) => (
        <div
          key={r.seed}
          className="w-2.5 h-2.5 rounded-[2px]"
          style={{ background: r.valid ? "#22c55e" : "#ef4444", opacity: 0.7 }}
          title={`Seed ${r.seed}: ${r.valid ? "valid" : "invalid"}`}
        />
      ))}
    </div>
  );
}
