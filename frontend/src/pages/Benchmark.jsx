import { useState, useEffect } from "react";
import { api } from "../api";
import ReactionPicker from "../components/ReactionPicker";

export default function BenchmarkPage({ presets }) {
  const [n, setN] = useState(20);
  const [reaction, setReaction] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (presets && !reaction) {
      setReaction({ reactants: presets.reactions[0].reactants, label: presets.reactions[0].name });
    }
  }, [presets, reaction]);

  const run = async () => {
    const reactants = reaction?.reactants || presets?.reactions[0]?.reactants;
    if (!reactants) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.runBenchmark({ reactants, n });
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  const valDelta = result
    ? (result.summary.supervised_valency_pct - result.summary.unsupervised_valency_pct).toFixed(1)
    : 0;

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Benchmark</h1>
      <p className="text-[15px] mb-3" style={{ color: "var(--text-secondary)" }}>
        See how much the constraint supervisor actually helps by comparing it against raw generation.
      </p>
      <p className="text-[13px] mb-8 max-w-[750px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
        Each trial generates the same molecule twice: once with the constraint supervisor turned on
        and once without it. The results go side by side so you can compare valency correctness,
        charge conservation, bond validity, and mass accuracy. The summary stats at the top show
        the overall difference across all trials.
      </p>

      {presets && <ReactionPicker presets={presets} value={reaction} onChange={setReaction} />}

      <div className="flex items-end gap-4 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
        <div>
          <label className="text-[13px] block mb-1.5" style={{ color: "var(--text-muted)" }}>Trials</label>
          <input
            type="number" value={n}
            onChange={(e) => setN(Math.min(100, Math.max(1, Number(e.target.value))))}
            min={1} max={100}
            className="w-24 rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
            style={{ background: "var(--bg-input)", border: "1px solid var(--border)" }}
          />
        </div>
        <button
          onClick={run} disabled={loading || !presets}
          className="px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors"
        >
          {loading ? `Running ${n} trials...` : "Run"}
        </button>
      </div>

      {error && <p className="text-red-400 text-[14px] mb-6">{error}</p>}

      {result && (
        <>
          <div className="mb-10">
            <span className="text-[36px] font-semibold text-green-400 font-mono">
              +{valDelta}pp
            </span>
            <span className="text-[15px] ml-3" style={{ color: "var(--text-secondary)" }}>
              valency improvement with supervision across {result.n} trials
            </span>
          </div>

          <div className="grid grid-cols-2 gap-12 mb-10">
            <BarPair
              label="Valency validity"
              supervised={result.summary.supervised_valency_pct}
              unsupervised={result.summary.unsupervised_valency_pct}
            />
            <BarPair
              label="Full conservation"
              supervised={result.summary.supervised_full_pct}
              unsupervised={result.summary.unsupervised_full_pct}
            />
          </div>

          {result.runs && (
            <div className="mb-10">
              <span className="text-[15px] text-white block mb-4">Per-trial results</span>
              <RunScatter runs={result.runs} />
            </div>
          )}

          <table className="w-full text-[14px] mb-10">
            <thead>
              <tr className="border-b" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                <th className="text-left py-3 font-normal">Metric</th>
                <th className="text-right py-3 font-normal">Supervised</th>
                <th className="text-right py-3 font-normal">Unsupervised</th>
                <th className="text-right py-3 font-normal">Delta</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                <td className="py-3 text-white">Valency validity</td>
                <td className="py-3 text-right font-mono text-green-400">{result.summary.supervised_valency_pct}%</td>
                <td className="py-3 text-right font-mono text-red-400">{result.summary.unsupervised_valency_pct}%</td>
                <td className="py-3 text-right font-mono text-white">+{valDelta}pp</td>
              </tr>
              <tr>
                <td className="py-3 text-white">Full conservation</td>
                <td className="py-3 text-right font-mono text-green-400">{result.summary.supervised_full_pct}%</td>
                <td className="py-3 text-right font-mono text-red-400">{result.summary.unsupervised_full_pct}%</td>
                <td className="py-3 text-right font-mono text-white">
                  +{(result.summary.supervised_full_pct - result.summary.unsupervised_full_pct).toFixed(1)}pp
                </td>
              </tr>
            </tbody>
          </table>

          <p className="text-[13px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
            These benchmarks use random GNN weights. Trained models produce higher
            conservation rates while the supervisor provides an additional safety layer.
          </p>
        </>
      )}
    </div>
  );
}

function BarPair({ label, supervised, unsupervised }) {
  return (
    <div>
      <span className="text-[14px] text-white block mb-4">{label}</span>
      <div className="space-y-3">
        <div>
          <div className="flex justify-between text-[13px] mb-1.5">
            <span style={{ color: "var(--text-secondary)" }}>Supervised</span>
            <span className="font-mono text-white">{supervised}%</span>
          </div>
          <div className="h-2 rounded-full" style={{ background: "var(--border)" }}>
            <div className="h-full bg-green-500 rounded-full transition-all duration-700" style={{ width: `${supervised}%` }} />
          </div>
        </div>
        <div>
          <div className="flex justify-between text-[13px] mb-1.5">
            <span style={{ color: "var(--text-secondary)" }}>Unsupervised</span>
            <span className="font-mono text-white">{unsupervised}%</span>
          </div>
          <div className="h-2 rounded-full" style={{ background: "var(--border)" }}>
            <div className="h-full bg-red-500 rounded-full transition-all duration-700" style={{ width: `${unsupervised}%` }} />
          </div>
        </div>
      </div>
    </div>
  );
}

function RunScatter({ runs }) {
  const n = runs.length;
  const pad = { top: 16, right: 8, bottom: 24, left: 80 };
  const dotR = Math.min(5, Math.max(3, 180 / n));
  const w = Math.max(300, n * (dotR * 2 + 2) + pad.left + pad.right);
  const rowH = 32;
  const h = pad.top + rowH * 4 + pad.bottom;

  const rows = [
    { key: "supervised_valency", label: "Sup. valency" },
    { key: "supervised_conservation", label: "Sup. full" },
    { key: "unsupervised_valency", label: "Unsup. valency" },
    { key: "unsupervised_conservation", label: "Unsup. full" },
  ];

  return (
    <div className="overflow-x-auto">
      <svg width={w} height={h}>
        {rows.map((row, ri) => {
          const y = pad.top + ri * rowH + rowH / 2;
          return (
            <g key={row.key}>
              <text x={pad.left - 8} y={y + 1} textAnchor="end" dominantBaseline="middle" fill="#737373" fontSize={11} fontFamily="Inter, sans-serif">
                {row.label}
              </text>
              <line x1={pad.left} y1={y} x2={w - pad.right} y2={y} stroke="#262626" strokeWidth={1} />
              {runs.map((r, i) => {
                const x = pad.left + (i / Math.max(1, n - 1)) * (w - pad.left - pad.right);
                const valid = r[row.key];
                return (
                  <circle key={i} cx={x} cy={y} r={dotR}
                    fill={valid ? "#22c55e" : "#ef4444"} fillOpacity={0.75}>
                    <title>Seed {r.seed}: {valid ? "valid" : "invalid"}</title>
                  </circle>
                );
              })}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
