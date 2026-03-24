import { useState, useEffect } from "react";
import { api } from "../api";
import ReactionPicker from "../components/ReactionPicker";

export default function TrainingPage({ presets }) {
  const [config, setConfig] = useState({
    population: 12, generations: 8, hidden_dim: 64, T: 10,
  });
  const [gradConfig, setGradConfig] = useState({
    epochs: 40, lr: 0.001, hidden_dim: 64, T: 20, steps_per_epoch: 80, seed: 42,
  });
  const [reaction, setReaction] = useState(null);
  const [result, setResult] = useState(null);
  const [gradResult, setGradResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [gradLoading, setGradLoading] = useState(false);
  const [error, setError] = useState(null);
  const [gradError, setGradError] = useState(null);

  useEffect(() => {
    if (presets && !reaction) {
      setReaction({ reactants: presets.reactions[0].reactants, label: presets.reactions[0].name });
    }
  }, [presets, reaction]);

  const set = (k, v) => setConfig((p) => ({ ...p, [k]: v }));
  const setG = (k, v) => setGradConfig((p) => ({ ...p, [k]: v }));

  const runGrad = async () => {
    setGradLoading(true); setGradError(null); setGradResult(null);
    try {
      const data = await api.trainWeights({ ...gradConfig });
      setGradResult(data);
      window.dispatchEvent(new CustomEvent("chemcsp-reload-info"));
    } catch (e) { setGradError(e.message); }
    finally { setGradLoading(false); }
  };

  const run = async () => {
    const reactants = reaction?.reactants || presets?.reactions[0]?.reactants;
    if (!reactants) return;
    setLoading(true); setError(null); setResult(null);
    try {
      const data = await api.trainModel({ ...config, reactants });
      setResult(data);
    } catch (e) { setError(e.message); }
    finally { setLoading(false); }
  };

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10">
      <h1 className="text-[28px] font-semibold text-white mb-2">Model Training</h1>
      <p className="text-[15px] mb-3" style={{ color: "var(--text-secondary)" }}>
        Train the GNN weights with gradient descent, or search for better random seeds without changing weights.
      </p>
      <p className="text-[13px] mb-10 max-w-[750px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
        The generative core is a graph neural network. Use <span className="text-white">Train weights</span> to run
        PyTorch Adam on a denoising objective over preset molecules and reaction products. The checkpoint is saved
        under <span className="font-mono text-neutral-400">checkpoints/diffusion_weights.pt</span> and is loaded
        automatically for Supervisor, Benchmark, Simulation, and Pathways when <span className="font-mono">hidden_dim</span> matches.
        Evolutionary training below only mutates RNG seeds and does not update weights.
      </p>

      <div className="mb-12 pb-10 border-b" style={{ borderColor: "var(--border)" }}>
        <h2 className="text-[18px] font-medium text-white mb-2">Train GNN weights (gradient descent)</h2>
        <p className="text-[13px] mb-6 max-w-[700px]" style={{ color: "var(--text-muted)" }}>
          Uses every preset molecule and reaction product as training graphs. After training, refresh the app header
          or reload the page to see the GNN badge.
        </p>
        <div className="flex items-end gap-4 flex-wrap mb-4">
          {[
            { key: "epochs", label: "Epochs", min: 5, max: 200 },
            { key: "lr", label: "Learning rate", min: 0.0001, max: 0.05, step: 0.0001 },
            { key: "hidden_dim", label: "Hidden dim", min: 8, max: 128 },
            { key: "T", label: "Diffusion T", min: 5, max: 50 },
            { key: "steps_per_epoch", label: "Steps / epoch", min: 20, max: 300 },
            { key: "seed", label: "Seed", min: 0, max: 99999 },
          ].map(({ key, label, min, max, step }) => (
            <div key={key}>
              <label className="text-[13px] block mb-1.5" style={{ color: "var(--text-muted)" }}>{label}</label>
              <input
                type="number"
                value={gradConfig[key]}
                onChange={(e) => setG(key, Number(e.target.value))}
                min={min} max={max} step={step || 1}
                className="w-28 rounded-md px-3 py-2 text-[14px] font-mono text-white focus:outline-none focus:ring-1 focus:ring-neutral-500"
                style={{ background: "var(--bg-input)", border: "1px solid var(--border)" }}
              />
            </div>
          ))}
          <button
            onClick={runGrad}
            disabled={gradLoading}
            className="px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors"
          >
            {gradLoading ? "Training…" : "Train weights"}
          </button>
        </div>
        {gradError && <p className="text-red-400 text-[14px] mb-4">{gradError}</p>}
        {gradLoading && (
          <p className="text-[13px] mb-4 animate-pulse-slow" style={{ color: "var(--text-muted)" }}>
            Running PyTorch on preset molecules (may take a minute)…
          </p>
        )}
        {gradResult && (
          <div className="space-y-4">
            <p className="text-[14px]" style={{ color: "var(--text-secondary)" }}>
              Saved <span className="font-mono text-white">{gradResult.checkpoint_path}</span>
              {" "}({gradResult.n_molecules} graphs, final loss {gradResult.final_loss}, {gradResult.training_wall_s}s)
            </p>
            <LossCurve losses={gradResult.loss_history} />
          </div>
        )}
      </div>

      <h2 className="text-[18px] font-medium text-white mb-2">Evolutionary seed search (no weight updates)</h2>
      <p className="text-[13px] mb-6 max-w-[700px]" style={{ color: "var(--text-muted)" }}>
        Spawns models with different random seeds, scores them through the supervised pipeline, and keeps the best seeds.
        This does not change network weights; use Train weights above for that.
      </p>

      {presets && <ReactionPicker presets={presets} value={reaction} onChange={setReaction} />}

      <div className="grid grid-cols-12 gap-10 mb-8 pb-8 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="col-span-7">
          <div className="flex items-end gap-4 flex-wrap">
            {[
              { key: "population", label: "Population", min: 4, max: 32 },
              { key: "generations", label: "Generations", min: 1, max: 20 },
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
              onClick={run} disabled={loading || !presets}
              className="px-5 py-2 bg-white text-black text-[14px] font-medium rounded-md hover:bg-neutral-200 disabled:opacity-30 transition-colors"
            >
              {loading ? "Training..." : "Start"}
            </button>
          </div>
          {loading && (
            <p className="mt-3 text-[13px] animate-pulse-slow" style={{ color: "var(--text-muted)" }}>
              Evaluating {config.population * config.generations} configurations...
            </p>
          )}
        </div>
        <div className="col-span-5 text-[13px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
          <p className="mb-2">
            <span className="text-white">Initialize</span> a population of GNN models with random seeds.
            <span className="text-white"> Evaluate</span> each through the supervised pipeline.
          </p>
          <p>
            <span className="text-white">Select</span> top 25% as parents.
            <span className="text-white"> Mutate</span> to fill the next generation.
            <span className="text-white"> Repeat</span> until convergence.
          </p>
        </div>
      </div>

      {error && <p className="text-red-400 text-[14px] mb-6">{error}</p>}

      {result && (
        <>
          <div className="mb-10">
            <span className="text-[13px] block mb-1" style={{ color: "var(--text-muted)" }}>Best model</span>
            <span className="text-[32px] font-semibold font-mono text-white">
              seed {result.best_seed}
            </span>
            <span className="text-[15px] ml-3" style={{ color: "var(--text-secondary)" }}>
              fitness {result.best_score} over {result.config.generations} generations
            </span>
          </div>

          <div className="grid grid-cols-2 gap-8 mb-10">
            <Curve history={result.history} metric="best_score" label="Best fitness" color="#22c55e" />
            <Curve history={result.history} metric="avg_score" label="Average fitness" color="#3b82f6" />
            <Curve history={result.history} metric="valency_rate" label="Valency rate %" color="#eab308" max={100} />
            <Curve history={result.history} metric="conservation_rate" label="Conservation rate %" color="#a855f7" max={100} />
          </div>

          <div className="mb-8">
            <span className="text-[15px] text-white block mb-4">Generation log</span>
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                  <th className="text-left py-2 font-normal">Gen</th>
                  <th className="text-right py-2 font-normal">Best seed</th>
                  <th className="text-right py-2 font-normal">Score</th>
                  <th className="text-right py-2 font-normal">Avg</th>
                  <th className="text-right py-2 font-normal">Valency</th>
                  <th className="text-right py-2 font-normal">Conservation</th>
                </tr>
              </thead>
              <tbody>
                {result.history.map((h) => (
                  <tr key={h.generation} className="border-b" style={{ borderColor: "var(--border)" }}>
                    <td className="py-2 font-mono text-white">{h.generation}</td>
                    <td className="py-2 text-right font-mono text-green-400">{h.best_seed}</td>
                    <td className="py-2 text-right font-mono text-white">{h.best_score}</td>
                    <td className="py-2 text-right font-mono" style={{ color: "var(--text-muted)" }}>{h.avg_score}</td>
                    <td className="py-2 text-right font-mono text-yellow-400">{h.valency_rate}%</td>
                    <td className="py-2 text-right font-mono text-purple-400">{h.conservation_rate}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}

function LossCurve({ losses }) {
  if (!losses?.length) return null;
  const history = losses.map((loss, generation) => ({ generation, loss }));
  return (
    <Curve
      history={history}
      metric="loss"
      label="Training loss (per epoch)"
      color="#22c55e"
      xKey="generation"
    />
  );
}

function Curve({ history, metric, label, color, max, xKey = "generation" }) {
  if (!history?.length) return null;
  const pad = { top: 16, right: 12, bottom: 24, left: 40 };
  const w = 480, h = 160;
  const iw = w - pad.left - pad.right;
  const ih = h - pad.top - pad.bottom;

  const vals = history.map((h) => h[metric]);
  const minV = Math.min(...vals);
  const maxV = max !== undefined ? max : Math.max(...vals, minV + 1e-6);
  const range = maxV - minV || 1;

  const pts = vals.map((v, i) => {
    const x = pad.left + (i / Math.max(1, vals.length - 1)) * iw;
    const y = pad.top + ih - ((v - minV) / range) * ih;
    return `${x},${y}`;
  }).join(" ");

  return (
    <div>
      <span className="text-[13px] block mb-2" style={{ color: "var(--text-secondary)" }}>{label}</span>
      <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="w-full">
        {[0, 0.5, 1].map((f) => {
          const y = pad.top + ih * (1 - f);
          return (
            <g key={f}>
              <line x1={pad.left} y1={y} x2={w - pad.right} y2={y} stroke="#262626" strokeWidth={0.5} />
              <text x={pad.left - 6} y={y + 3} textAnchor="end" fill="#525252" fontSize={9} fontFamily="JetBrains Mono, monospace">
                {(minV + range * f).toFixed(range > 10 ? 0 : 2)}
              </text>
            </g>
          );
        })}
        <polygon
          points={`${pad.left},${pad.top + ih} ${pts} ${pad.left + iw},${pad.top + ih}`}
          fill={color} fillOpacity={0.06}
        />
        <polyline points={pts} fill="none" stroke={color} strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round" />
        {vals.map((v, i) => {
          const x = pad.left + (i / Math.max(1, vals.length - 1)) * iw;
          const y = pad.top + ih - ((v - minV) / range) * ih;
          const row = history[i];
          return <circle key={i} cx={x} cy={y} r={2.5} fill={color}><title>{xKey} {row[xKey] ?? i}: {v}</title></circle>;
        })}
      </svg>
    </div>
  );
}
