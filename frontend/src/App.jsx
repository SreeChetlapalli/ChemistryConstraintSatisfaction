import { useState, useEffect } from "react";
import { api } from "./api";
import Overview from "./pages/Overview";
import ConstraintChecker from "./pages/ConstraintChecker";
import SupervisorPage from "./pages/Supervisor";
import BenchmarkPage from "./pages/Benchmark";
import MoleculeLab from "./pages/MoleculeLab";
import TrainingPage from "./pages/Training";
import MonteCarloPage from "./pages/MonteCarlo";
import PathwayPage from "./pages/Pathway";

const TABS = [
  { id: "overview", label: "Overview" },
  { id: "lab", label: "Lab" },
  { id: "checker", label: "Constraints" },
  { id: "supervisor", label: "Supervisor" },
  { id: "training", label: "Training" },
  { id: "benchmark", label: "Benchmark" },
  { id: "simulation", label: "Simulation" },
  { id: "pathways", label: "Pathways" },
];

export default function App() {
  const [page, setPage] = useState("overview");
  const [info, setInfo] = useState(null);
  const [presets, setPresets] = useState(null);

  useEffect(() => {
    api.getInfo().then(setInfo).catch(() => {});
    api.getPresets().then(setPresets).catch(() => {});
  }, []);

  const renderPage = () => {
    switch (page) {
      case "overview":
        return <Overview info={info} onNavigate={setPage} />;
      case "lab":
        return <MoleculeLab presets={presets} />;
      case "checker":
        return <ConstraintChecker presets={presets} />;
      case "supervisor":
        return <SupervisorPage presets={presets} />;
      case "training":
        return <TrainingPage presets={presets} />;
      case "benchmark":
        return <BenchmarkPage presets={presets} />;
      case "simulation":
        return <MonteCarloPage presets={presets} />;
      case "pathways":
        return <PathwayPage presets={presets} />;
      default:
        return null;
    }
  };

  return (
    <div className="h-screen flex flex-col" style={{ background: "var(--bg)" }}>
      <header className="shrink-0 border-b" style={{ borderColor: "var(--border)" }}>
        <div className="max-w-[1200px] mx-auto px-6 flex items-center justify-between h-14">
          <div className="flex items-center gap-8">
            <span className="text-[15px] font-semibold tracking-tight text-white">
              ChemCSP
            </span>
            <nav className="flex items-center gap-1">
              {TABS.map(({ id, label }) => (
                <button
                  key={id}
                  onClick={() => setPage(id)}
                  className={`px-3 py-1.5 rounded-md text-[13px] transition-colors ${
                    page === id
                      ? "text-white bg-white/[0.08]"
                      : "text-neutral-500 hover:text-neutral-300"
                  }`}
                >
                  {label}
                </button>
              ))}
            </nav>
          </div>
          {info && (
            <div className="flex items-center gap-4 text-[13px] text-neutral-500">
              <span>v{info.version}</span>
              <span className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${info.z3_available ? "bg-green-500" : "bg-yellow-500"}`} />
                {info.z3_available ? "Z3" : "Python"}
              </span>
            </div>
          )}
        </div>
      </header>
      <main className="flex-1 overflow-auto page-transition">
        {renderPage()}
      </main>
    </div>
  );
}
