import { ArrowRight } from "lucide-react";
import { ELEMENTS, GROUP_COLORS } from "../data/elements";

export default function Overview({ info, onNavigate }) {
  return (
    <div className="max-w-[1200px] mx-auto px-6">
      <div className="pt-20 pb-16 border-b" style={{ borderColor: "var(--border)" }}>
        <h1 className="text-[48px] font-semibold text-white leading-[1.08] tracking-tight mb-5">
          Constraint satisfaction<br />
          for molecular generation
        </h1>
        <p className="text-lg leading-relaxed max-w-[520px]" style={{ color: "var(--text-secondary)" }}>
          A runtime supervisor that wraps AI-driven diffusion models,
          enforcing mass conservation, charge balance, and valency bounds
          at every denoising step.
        </p>
      </div>

      <div className="grid grid-cols-3 gap-px mt-px" style={{ background: "var(--border)" }}>
        {[
          { label: "Mass Conservation", value: "\u03A3 m(R) = \u03A3 m(P)" },
          { label: "Charge Conservation", value: "\u03A3 q(R) = \u03A3 q(P)" },
          { label: "Valency Bound", value: "bonds(a) + H(a) \u2264 v(a)" },
        ].map(({ label, value }) => (
          <div key={label} className="py-6 px-1" style={{ background: "var(--bg)" }}>
            <span className="text-[13px] block mb-1" style={{ color: "var(--text-muted)" }}>
              {label}
            </span>
            <span className="text-[15px] font-mono text-white">{value}</span>
          </div>
        ))}
      </div>

      <div className="mt-16 mb-16">
        <h2 className="text-[22px] font-semibold text-white mb-8">Tools</h2>
        <div className="space-y-0 border-t" style={{ borderColor: "var(--border)" }}>
          {[
            { id: "lab", label: "Molecule Lab", desc: "Build molecules by hand with live valency checks and a 3D preview." },
            { id: "checker", label: "Constraint Checker", desc: "Type a reaction or pick a preset and check it against the conservation rules." },
            { id: "supervisor", label: "Supervisor", desc: "Run the diffusion loop and scrub through each generation step." },
            { id: "training", label: "Model Training", desc: "Evolve model seeds over generations and track which ones produce valid molecules." },
            { id: "benchmark", label: "Benchmark", desc: "Run the same reaction with and without the constraint supervisor and compare." },
            { id: "simulation", label: "Simulation", desc: "Generate hundreds of molecules at once and look at the results statistically." },
            { id: "pathways", label: "Pathways", desc: "Line up multiple reactions where each step's product becomes the next step's input." },
          ].map(({ id, label, desc }) => (
            <button
              key={id}
              onClick={() => onNavigate?.(id)}
              className="w-full flex items-center justify-between py-5 border-b group text-left"
              style={{ borderColor: "var(--border)" }}
            >
              <div>
                <span className="text-[16px] font-medium text-white group-hover:text-green-400 transition-colors">
                  {label}
                </span>
                <span className="text-[14px] ml-4" style={{ color: "var(--text-muted)" }}>
                  {desc}
                </span>
              </div>
              <ArrowRight
                className="w-4 h-4 shrink-0 ml-4 text-neutral-600 group-hover:text-green-400 group-hover:translate-x-1 transition-all"
              />
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-16 mb-16">
        <div>
          <h3 className="text-[15px] font-medium text-white mb-3">Diffusion Model</h3>
          <p className="text-[14px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            NumPy-based graph neural network with two message-passing layers.
            Predicts atom features and bond adjacency. No PyTorch required.
          </p>
        </div>
        <div>
          <h3 className="text-[15px] font-medium text-white mb-3">Constraint Verifier</h3>
          <p className="text-[14px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            Z3 SMT solver for formal proofs with automatic pure-Python fallback.
            Sub-millisecond per check.
          </p>
        </div>
        <div>
          <h3 className="text-[15px] font-medium text-white mb-3">Supervisor Loop</h3>
          <p className="text-[14px] leading-relaxed" style={{ color: "var(--text-secondary)" }}>
            Per-step validation with targeted corrections: valency reduction,
            hydrogen adjustment, or full state backtracking.
          </p>
        </div>
      </div>

      <div className="mb-16">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-[22px] font-semibold text-white">
            {info ? `${info.elements.length} elements` : "Elements"}
          </h2>
          <span className="text-[13px]" style={{ color: "var(--text-muted)" }}>
            Full constraint support for Z=1 through Z=86
          </span>
        </div>
        <div className="flex flex-wrap gap-[2px]">
          {ELEMENTS.map((el) => {
            const gc = GROUP_COLORS[el.group] || "#6b7280";
            return (
              <div
                key={el.sym}
                className="w-[28px] h-[28px] rounded-[3px] flex items-center justify-center text-[9px] font-medium"
                style={{ background: gc + "14", color: gc }}
                title={`${el.name} - Z=${el.z}, ${el.mass}u, valency ${el.maxVal}`}
              >
                {el.sym}
              </div>
            );
          })}
        </div>
        <div className="flex gap-5 mt-4">
          {Object.entries(GROUP_COLORS).map(([group, color]) => (
            <div key={group} className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-sm" style={{ background: color }} />
              <span className="text-[11px] capitalize" style={{ color: "var(--text-muted)" }}>
                {group.replace("-", " ")}
              </span>
            </div>
          ))}
        </div>
      </div>

      {info && (
        <div className="pb-16 flex gap-12 text-[13px]" style={{ color: "var(--text-muted)" }}>
          <span>Engine v{info.version}</span>
          <span>Solver: {info.z3_available ? "Z3 SMT" : "Pure Python"}</span>
          <span>Stack: NumPy + Flask + React + Three.js</span>
        </div>
      )}
    </div>
  );
}
