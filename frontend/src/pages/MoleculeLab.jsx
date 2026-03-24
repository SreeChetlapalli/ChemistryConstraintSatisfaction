import { useState, useRef, useCallback, useMemo } from "react";
import { api } from "../api";
import MoleculeViewer3D from "../components/MoleculeViewer3D";
import Molecule3DModal from "../components/Molecule3DModal";
import { ELEMENTS, ELEMENT_MAP, GROUP_COLORS } from "../data/elements";
import {
  MousePointer2,
  Circle,
  Minus,
  Trash2,
  RotateCcw,
  Zap,
  CheckCircle2,
  XCircle,
  Search,
  Maximize2,
} from "lucide-react";

const COMMON_ELEMENTS = ["H", "C", "N", "O", "F", "P", "S", "Cl", "Br", "I"];

function effValency(element, charge) {
  const base = ELEMENT_MAP[element]?.maxVal || 4;
  const delta = { N: {1:1,"-1":-1}, O: {"-1":-1}, S: {1:1,2:2}, P: {1:1} };
  return base + (delta[element]?.[charge] || 0);
}

let _nextId = 1;
function uid() {
  return `n${_nextId++}`;
}

export default function MoleculeLab({ presets }) {
  const [tool, setTool] = useState("atom");
  const [elemSearch, setElemSearch] = useState("");
  const [selElem, setSelElem] = useState("C");
  const [bondOrd, setBondOrd] = useState(1);

  const [atoms, setAtoms] = useState([]);
  const [bonds, setBonds] = useState([]);
  const [selAtom, setSelAtom] = useState(null);
  const [dragBond, setDragBond] = useState(null);
  const [dragging, setDragging] = useState(null);

  const [checkResult, setCheckResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [show3DModal, setShow3DModal] = useState(false);

  const svgRef = useRef(null);

  const bondTotals = useMemo(() => {
    const t = {};
    atoms.forEach((a) => (t[a.id] = 0));
    bonds.forEach((b) => {
      if (t[b.from] !== undefined) t[b.from] += b.order;
      if (t[b.to] !== undefined) t[b.to] += b.order;
    });
    return t;
  }, [atoms, bonds]);

  const props = useMemo(() => {
    let mass = 0,
      charge = 0,
      implH = 0;
    atoms.forEach((a) => {
      const el = ELEMENT_MAP[a.element];
      mass += el?.mass || 0;
      charge += a.charge;
      const ev = effValency(a.element, a.charge);
      const iH = Math.max(0, ev - (bondTotals[a.id] || 0));
      implH += iH;
      mass += iH * 1.008;
    });
    return {
      mass,
      charge,
      implH,
      atomCount: atoms.length,
      bondCount: bonds.length,
    };
  }, [atoms, bonds, bondTotals]);

  const viewerAtoms = useMemo(() =>
    atoms.map((a) => ({
      element: a.element,
      bonds: bondTotals[a.id] || 0,
      formal_charge: a.charge,
      implicit_h: 0,
      effective_valency: effValency(a.element, a.charge),
      total_bonds: bondTotals[a.id] || 0,
    })),
  [atoms, bondTotals]);

  const viewerAdj = useMemo(() => {
    const n = atoms.length;
    if (n === 0) return [];
    const idxOf = {};
    atoms.forEach((a, i) => { idxOf[a.id] = i; });
    const m = Array.from({ length: n }, () => new Float32Array(n));
    bonds.forEach((b) => {
      const fi = idxOf[b.from], ti = idxOf[b.to];
      if (fi !== undefined && ti !== undefined) {
        m[fi][ti] = b.order;
        m[ti][fi] = b.order;
      }
    });
    return m;
  }, [atoms, bonds]);

  const svgPt = useCallback(
    (e) => {
      const svg = svgRef.current;
      if (!svg) return { x: 0, y: 0 };
      const r = svg.getBoundingClientRect();
      return { x: e.clientX - r.left, y: e.clientY - r.top };
    },
    [],
  );

  const onCanvasDown = useCallback(
    (e) => {
      if (e.target !== svgRef.current && e.target.tagName !== "rect") return;
      if (tool === "atom") {
        const pt = svgPt(e);
        setAtoms((p) => [
          ...p,
          { id: uid(), element: selElem, x: pt.x, y: pt.y, charge: 0 },
        ]);
      } else if (tool === "select") {
        setSelAtom(null);
      }
    },
    [tool, selElem, svgPt],
  );

  const onAtomDown = useCallback(
    (e, id) => {
      e.stopPropagation();
      if (tool === "erase") {
        setAtoms((p) => p.filter((a) => a.id !== id));
        setBonds((p) => p.filter((b) => b.from !== id && b.to !== id));
        if (selAtom === id) setSelAtom(null);
        return;
      }
      if (tool === "bond") {
        const pt = svgPt(e);
        setDragBond({ fromId: id, mx: pt.x, my: pt.y });
        return;
      }
      if (tool === "select" || tool === "atom") {
        setSelAtom(id);
        const atom = atoms.find((a) => a.id === id);
        if (atom) {
          const pt = svgPt(e);
          setDragging({
            id,
            ox: pt.x - atom.x,
            oy: pt.y - atom.y,
          });
        }
      }
    },
    [tool, svgPt, atoms, selAtom],
  );

  const onMove = useCallback(
    (e) => {
      const pt = svgPt(e);
      if (dragBond)
        setDragBond((p) => (p ? { ...p, mx: pt.x, my: pt.y } : null));
      if (dragging)
        setAtoms((p) =>
          p.map((a) =>
            a.id === dragging.id
              ? { ...a, x: pt.x - dragging.ox, y: pt.y - dragging.oy }
              : a,
          ),
        );
    },
    [dragBond, dragging, svgPt],
  );

  const onUp = useCallback(
    (e) => {
      if (dragging) {
        setDragging(null);
        return;
      }
      if (dragBond) {
        const pt = svgPt(e);
        const target = atoms.find((a) => {
          const dx = a.x - pt.x,
            dy = a.y - pt.y;
          return Math.sqrt(dx * dx + dy * dy) < 28 && a.id !== dragBond.fromId;
        });
        if (target) {
          const existing = bonds.find(
            (b) =>
              (b.from === dragBond.fromId && b.to === target.id) ||
              (b.from === target.id && b.to === dragBond.fromId),
          );
          if (existing) {
            setBonds((p) =>
              p.map((b) =>
                b.id === existing.id
                  ? { ...b, order: (b.order % 3) + 1 }
                  : b,
              ),
            );
          } else {
            setBonds((p) => [
              ...p,
              { id: uid(), from: dragBond.fromId, to: target.id, order: bondOrd },
            ]);
          }
        }
        setDragBond(null);
      }
    },
    [dragBond, atoms, bonds, bondOrd, svgPt, dragging],
  );

  const onBondClick = useCallback(
    (e, id) => {
      e.stopPropagation();
      if (tool === "erase") setBonds((p) => p.filter((b) => b.id !== id));
      else if (tool === "bond")
        setBonds((p) =>
          p.map((b) => (b.id === id ? { ...b, order: (b.order % 3) + 1 } : b)),
        );
    },
    [tool],
  );

  const adjCharge = useCallback((id, d) => {
    setAtoms((p) =>
      p.map((a) =>
        a.id === id
          ? { ...a, charge: Math.max(-3, Math.min(3, a.charge + d)) }
          : a,
      ),
    );
  }, []);

  const checkConstraints = async () => {
    if (atoms.length === 0) return;
    setLoading(true);
    setCheckResult(null);
    try {
      const mol = {
        name: "custom",
        atoms: atoms.map((a) => ({
          element: a.element,
          bonds: bondTotals[a.id] || 0,
          formal_charge: a.charge,
          implicit_h: Math.max(
            0,
            effValency(a.element, a.charge) - (bondTotals[a.id] || 0),
          ),
        })),
      };
      const data = await api.checkIntermediate({ molecule: mol });
      setCheckResult(data);
    } catch (err) {
      setCheckResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const importPreset = useCallback((mol) => {
    const n = mol.atoms.length;
    const cx = 350,
      cy = 250;
    const rad = n > 1 ? Math.min(150, 40 * n) : 0;
    const na = mol.atoms.map((a, i) => ({
      id: uid(),
      element: a.element,
      x: cx + rad * Math.cos((2 * Math.PI * i) / n - Math.PI / 2),
      y: cy + rad * Math.sin((2 * Math.PI * i) / n - Math.PI / 2),
      charge: a.formal_charge || 0,
    }));
    const rem = mol.atoms.map((a) => a.bonds);
    const nb = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        if (rem[i] <= 0 || rem[j] <= 0) continue;
        const ord = Math.min(rem[i], rem[j]);
        nb.push({ id: uid(), from: na[i].id, to: na[j].id, order: ord });
        rem[i] -= ord;
        rem[j] -= ord;
      }
    }
    setAtoms(na);
    setBonds(nb);
    setSelAtom(null);
    setCheckResult(null);
  }, []);

  const clearAll = useCallback(() => {
    setAtoms([]);
    setBonds([]);
    setSelAtom(null);
    setCheckResult(null);
  }, []);

  const selData = atoms.find((a) => a.id === selAtom);

  return (
    <div className="max-w-[1200px] mx-auto px-6 py-10 h-full flex flex-col">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-[28px] font-semibold text-white mb-1">Molecule Lab</h1>
          <p className="text-[15px]" style={{ color: "var(--text-secondary)" }}>
            Build molecules interactively with real-time constraint validation.
          </p>
          <p className="text-[13px] mt-1 max-w-[700px] leading-relaxed" style={{ color: "var(--text-muted)" }}>
            Pick elements, draw bonds, and adjust charges. The constraint engine runs in real time
            so you can see right away if something is over-bonded. Click the 3D preview on the
            right to open a bigger interactive view.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {presets && (
            <select
              onChange={(e) => {
                const idx = Number(e.target.value);
                if (idx >= 0) importPreset(presets.molecules[idx]);
                e.target.value = "-1";
              }}
              defaultValue="-1"
              className="bg-[#1a1a1a] border border-[#262626] rounded-md px-3 py-2 text-sm text-neutral-300 focus:outline-none focus:ring-1 focus:ring-neutral-500"
            >
              <option value="-1" disabled>
                Import preset...
              </option>
              {presets.molecules.map((m, i) => (
                <option key={i} value={i}>
                  {m.name}
                </option>
              ))}
            </select>
          )}
          <button
            onClick={clearAll}
            className="p-2 text-neutral-400 hover:text-red-400 transition-colors"
            title="Clear all"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 flex gap-4 min-h-0">
        {/* Toolbar */}
        <div className="w-48 shrink-0 space-y-3">
          <div className="bg-[#171717] border border-[#262626] rounded-lg p-3">
            <span className="text-[12px] text-neutral-500 block mb-2">
              Tools
            </span>
            <div className="grid grid-cols-2 gap-1.5">
              {[
                { id: "select", label: "Select", Icon: MousePointer2 },
                { id: "atom", label: "Atom", Icon: Circle },
                { id: "bond", label: "Bond", Icon: Minus },
                { id: "erase", label: "Erase", Icon: Trash2 },
              ].map(({ id, label, Icon }) => (
                <button
                  key={id}
                  onClick={() => setTool(id)}
                  className={`flex items-center gap-1.5 px-2 py-1.5 rounded text-[11px] transition-all ${
                    tool === id
                      ? "bg-white/10 text-white border border-white/20"
                      : "text-neutral-400 hover:text-neutral-200 border border-transparent hover:bg-white/5"
                  }`}
                >
                  <Icon className="w-3 h-3" />
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="bg-[#171717] border border-[#262626] rounded-lg p-3">
            <span className="text-[12px] text-neutral-500 block mb-2">
              Element
            </span>
            <div className="relative mb-2">
              <Search className="w-3 h-3 absolute left-2 top-1/2 -translate-y-1/2 text-neutral-600" />
              <input
                type="text"
                value={elemSearch}
                onChange={(e) => setElemSearch(e.target.value)}
                placeholder="Search..."
                className="w-full bg-[#1a1a1a] border border-[#262626]/50 rounded px-2 py-1 pl-6 text-[10px] text-neutral-300 focus:outline-none focus:ring-1 focus:ring-neutral-500"
              />
            </div>
            <div className="grid grid-cols-5 gap-1 max-h-[200px] overflow-y-auto">
              {(elemSearch.trim()
                ? ELEMENTS.filter(
                    (e) =>
                      e.sym.toLowerCase().startsWith(elemSearch.toLowerCase()) ||
                      e.name.toLowerCase().startsWith(elemSearch.toLowerCase()),
                  )
                : ELEMENTS.filter((e) => COMMON_ELEMENTS.includes(e.sym))
              ).map((el) => {
                const gc = GROUP_COLORS[el.group] || "#6b7280";
                return (
                  <button
                    key={el.sym}
                    onClick={() => { setSelElem(el.sym); setTool("atom"); }}
                    className={`w-8 h-8 rounded text-[11px] font-bold transition-all ${
                      selElem === el.sym && tool === "atom"
                        ? "ring-1 ring-white/40 scale-110"
                        : "hover:scale-105"
                    }`}
                    style={{
                      backgroundColor: gc + "18",
                      color: gc,
                      borderColor: selElem === el.sym && tool === "atom" ? gc : "transparent",
                      borderWidth: "1px",
                    }}
                    title={`${el.name} (${el.sym}) - valency ${el.maxVal}`}
                  >
                    {el.sym}
                  </button>
                );
              })}
            </div>
            {!elemSearch.trim() && (
              <p className="text-[9px] text-neutral-600 mt-1.5">
                Type to search all 86 elements
              </p>
            )}
          </div>

          {tool === "bond" && (
            <div className="bg-[#171717] border border-[#262626] rounded-lg p-3">
              <span className="text-[12px] text-neutral-500 block mb-2">
                Bond Order
              </span>
              <div className="flex gap-1">
                {[1, 2, 3].map((o) => (
                  <button
                    key={o}
                    onClick={() => setBondOrd(o)}
                    className={`flex-1 py-1.5 rounded text-[10px] font-mono transition-all ${
                      bondOrd === o
                        ? "bg-white/10 text-white border border-white/20"
                        : "text-neutral-400 hover:text-neutral-200 border border-[#262626]"
                    }`}
                  >
                    {o === 1 ? "Single" : o === 2 ? "Double" : "Triple"}
                  </button>
                ))}
              </div>
            </div>
          )}

          {selData && (
            <div className="bg-[#171717] border border-[#262626] rounded-lg p-3">
              <span className="text-[12px] text-neutral-500 block mb-2">
                Selected Atom
              </span>
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-neutral-400">Element</span>
                  <span className="text-xs font-mono text-white">
                    {selData.element}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-neutral-400">Bonds</span>
                  <span className="text-xs font-mono text-white">
                    {bondTotals[selAtom] || 0} /{" "}
                    {effValency(selData.element, selData.charge)}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-neutral-400">Charge</span>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() => adjCharge(selAtom, -1)}
                      className="w-5 h-5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-300 text-xs flex items-center justify-center"
                    >
                      -
                    </button>
                    <span className="text-xs font-mono text-white w-6 text-center">
                      {selData.charge > 0 ? "+" : ""}
                      {selData.charge}
                    </span>
                    <button
                      onClick={() => adjCharge(selAtom, 1)}
                      className="w-5 h-5 rounded bg-neutral-700 hover:bg-neutral-600 text-neutral-300 text-xs flex items-center justify-center"
                    >
                      +
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Canvas */}
        <div className="flex-1 bg-[#131313] border border-[#262626] rounded-lg overflow-hidden relative">
          <svg
            ref={svgRef}
            className="w-full h-full"
            style={{
              cursor:
                tool === "atom" || tool === "bond"
                  ? "crosshair"
                  : tool === "erase"
                    ? "not-allowed"
                    : "default",
            }}
            onMouseDown={onCanvasDown}
            onMouseMove={onMove}
            onMouseUp={onUp}
            onMouseLeave={() => {
              setDragBond(null);
              setDragging(null);
            }}
          >
            <defs>
              <pattern
                id="labgrid"
                width="30"
                height="30"
                patternUnits="userSpaceOnUse"
              >
                <path
                  d="M 30 0 L 0 0 0 30"
                  fill="none"
                  stroke="#1e293b"
                  strokeWidth="0.5"
                  opacity="0.3"
                />
              </pattern>
            </defs>
            <rect width="100%" height="100%" fill="url(#labgrid)" />

            {bonds.map((bond) => {
              const fa = atoms.find((a) => a.id === bond.from);
              const ta = atoms.find((a) => a.id === bond.to);
              if (!fa || !ta) return null;
              const dx = ta.x - fa.x,
                dy = ta.y - fa.y;
              const len = Math.sqrt(dx * dx + dy * dy) || 1;
              const nx = -dy / len,
                ny = dx / len;
              const offsets =
                bond.order === 1
                  ? [0]
                  : bond.order === 2
                    ? [-3.5, 3.5]
                    : [-5, 0, 5];
              return (
                <g
                  key={bond.id}
                  onClick={(e) => onBondClick(e, bond.id)}
                  style={{
                    cursor:
                      tool === "erase" || tool === "bond"
                        ? "pointer"
                        : "default",
                  }}
                >
                  <line
                    x1={fa.x}
                    y1={fa.y}
                    x2={ta.x}
                    y2={ta.y}
                    stroke="transparent"
                    strokeWidth={12}
                  />
                  {offsets.map((off, k) => (
                    <line
                      key={k}
                      x1={fa.x + nx * off}
                      y1={fa.y + ny * off}
                      x2={ta.x + nx * off}
                      y2={ta.y + ny * off}
                      stroke="#334155"
                      strokeWidth={2}
                      strokeLinecap="round"
                    />
                  ))}
                </g>
              );
            })}

            {dragBond &&
              (() => {
                const fa = atoms.find((a) => a.id === dragBond.fromId);
                if (!fa) return null;
                return (
                  <line
                    x1={fa.x}
                    y1={fa.y}
                    x2={dragBond.mx}
                    y2={dragBond.my}
                    stroke="#22c55e"
                    strokeWidth={1.5}
                    strokeDasharray="5,5"
                    opacity={0.6}
                    pointerEvents="none"
                  />
                );
              })()}

            {atoms.map((atom) => {
              const elem = ELEMENT_MAP[atom.element];
              const color = elem ? (GROUP_COLORS[elem.group] || "#6b7280") : "#6b7280";
              const tb = bondTotals[atom.id] || 0;
              const ev = effValency(atom.element, atom.charge);
              const over = tb > ev;
              const isSel = selAtom === atom.id;
              const R = 22;
              return (
                <g
                  key={atom.id}
                  onMouseDown={(e) => onAtomDown(e, atom.id)}
                  style={{
                    cursor: tool === "select" ? "grab" : "pointer",
                  }}
                >
                  {over && (
                    <circle
                      cx={atom.x}
                      cy={atom.y}
                      r={R + 6}
                      fill="#ef4444"
                      fillOpacity={0.12}
                    />
                  )}
                  {isSel && (
                    <circle
                      cx={atom.x}
                      cy={atom.y}
                      r={R + 4}
                      fill="none"
                      stroke="#22c55e"
                      strokeWidth={1.5}
                      strokeDasharray="3,2"
                    />
                  )}
                  <circle
                    cx={atom.x}
                    cy={atom.y}
                    r={R}
                    fill={color + "15"}
                    stroke={over ? "#ef4444" : color}
                    strokeWidth={isSel ? 2 : 1.5}
                  />
                  <text
                    x={atom.x}
                    y={atom.y - 1}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    fill="white"
                    fontSize={13}
                    fontWeight="700"
                    fontFamily="Inter, sans-serif"
                    pointerEvents="none"
                  >
                    {atom.element}
                  </text>
                  <text
                    x={atom.x}
                    y={atom.y + 12}
                    textAnchor="middle"
                    fill={over ? "#ef4444" : "#64748b"}
                    fontSize={8}
                    fontFamily="JetBrains Mono, monospace"
                    pointerEvents="none"
                  >
                    {tb}/{ev}
                  </text>
                  {atom.charge !== 0 && (
                    <>
                      <circle
                        cx={atom.x + R - 2}
                        cy={atom.y - R + 2}
                        r={8}
                        fill="#1e293b"
                        stroke="#475569"
                        strokeWidth={0.5}
                      />
                      <text
                        x={atom.x + R - 2}
                        y={atom.y - R + 3}
                        textAnchor="middle"
                        dominantBaseline="middle"
                        fill="#fbbf24"
                        fontSize={9}
                        fontWeight="700"
                        fontFamily="JetBrains Mono, monospace"
                        pointerEvents="none"
                      >
                        {atom.charge > 0 ? "+" : ""}
                        {atom.charge}
                      </text>
                    </>
                  )}
                </g>
              );
            })}

            {atoms.length === 0 && (
              <text
                x="50%"
                y="50%"
                textAnchor="middle"
                dominantBaseline="middle"
                fill="#334155"
                fontSize={14}
                fontFamily="Inter, sans-serif"
              >
                Click to place atoms. Drag between atoms to draw bonds.
              </text>
            )}
          </svg>
        </div>

        {/* 3D Preview + Right panel */}
        <div className="w-64 shrink-0 overflow-y-auto space-y-3">
          <div className="bg-[#171717] border border-[#262626] rounded-lg overflow-hidden">
            <div className="px-3 pt-3 pb-1 flex items-center justify-between">
              <span className="text-[12px] text-neutral-500">
                3D Preview
              </span>
              {atoms.length > 0 ? (
                <button
                  onClick={() => setShow3DModal(true)}
                  className="flex items-center gap-1 text-[10px] text-neutral-500 hover:text-white transition-colors"
                  title="Expand to full view"
                >
                  <Maximize2 size={11} /> Expand
                </button>
              ) : (
                <span className="text-[9px] text-neutral-600">drag to rotate</span>
              )}
            </div>
            {atoms.length > 0 ? (
              <div
                className="cursor-pointer"
                onClick={() => setShow3DModal(true)}
                title="Click to enlarge"
              >
                <MoleculeViewer3D
                  atoms={viewerAtoms}
                  adjacency={viewerAdj}
                  height={240}
                  autoRotate={true}
                />
              </div>
            ) : (
              <div className="h-[240px] flex items-center justify-center text-neutral-700 text-xs">
                Build a molecule to preview
              </div>
            )}
          </div>

          <div className="bg-[#171717] border border-[#262626] rounded-lg p-3">
            <span className="text-[12px] text-neutral-500 block mb-2">
              Properties
            </span>
            <div className="space-y-1.5">
              {[
                { l: "Atoms", v: props.atomCount },
                { l: "Bonds", v: props.bondCount },
                { l: "Mass", v: `${props.mass.toFixed(3)} u` },
                {
                  l: "Charge",
                  v: (props.charge > 0 ? "+" : "") + props.charge,
                },
                { l: "Implicit H", v: props.implH },
              ].map(({ l, v }) => (
                <div key={l} className="flex items-center justify-between">
                  <span className="text-[11px] text-neutral-500">{l}</span>
                  <span className="text-xs font-mono text-neutral-200">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="bg-[#171717] border border-[#262626] rounded-lg p-3">
            <span className="text-[12px] text-neutral-500 block mb-2">
              Valency
            </span>
            {atoms.length === 0 ? (
              <p className="text-[11px] text-neutral-600">No atoms</p>
            ) : (
              <div className="space-y-1 max-h-[220px] overflow-y-auto pr-1">
                {atoms.map((a) => {
                  const tb = bondTotals[a.id] || 0;
                  const ev = effValency(a.element, a.charge);
                  const ok = tb <= ev;
                  const emData = ELEMENT_MAP[a.element];
                  const col = emData ? (GROUP_COLORS[emData.group] || "#6b7280") : "#6b7280";
                  return (
                    <div
                      key={a.id}
                      className={`flex items-center justify-between px-2 py-1 rounded text-[10px] ${ok ? "bg-green-500/5" : "bg-red-500/10"}`}
                    >
                      <span className="font-bold" style={{ color: col }}>
                        {a.element}
                      </span>
                      <span
                        className={`font-mono ${ok ? "text-green-400" : "text-red-400"}`}
                      >
                        {tb}/{ev}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          <button
            onClick={checkConstraints}
            disabled={loading || atoms.length === 0}
            className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-white text-black disabled:opacity-30 text-sm font-medium rounded-lg hover:bg-neutral-200 transition-colors"
          >
            <Zap className="w-4 h-4" />
            {loading ? "Checking..." : "Check Constraints"}
          </button>

          {checkResult && !checkResult.error && (
            <div
              className={`rounded-lg p-3 ${checkResult.result.sat ? "bg-green-500/5" : "bg-red-500/5"}`}
            >
              <div className="flex items-center gap-2 mb-1">
                {checkResult.result.sat ? (
                  <CheckCircle2 className="w-4 h-4 text-green-500" />
                ) : (
                  <XCircle className="w-4 h-4 text-red-500" />
                )}
                <span
                  className={`text-xs font-semibold ${checkResult.result.sat ? "text-green-400" : "text-red-400"}`}
                >
                  {checkResult.result.sat ? "Valid" : "Violations"}
                </span>
              </div>
              {checkResult.result.violations.length > 0 && (
                <div className="mt-2 space-y-1">
                  {checkResult.result.violations.map((v, i) => (
                    <p key={i} className="text-[10px] text-red-300 leading-snug">
                      {v}
                    </p>
                  ))}
                </div>
              )}
              <div className="mt-2 flex gap-3 text-[10px] text-neutral-500">
                <span>Mass: {checkResult.total_mass?.toFixed(2)} u</span>
                <span>{checkResult.elapsed_ms}ms</span>
              </div>
              {checkResult.lipinski && (
                <div className="mt-1.5 flex gap-2 text-[10px]">
                  <span className={checkResult.lipinski.passes_ro5 ? "text-green-400" : "text-yellow-400"}>
                    {checkResult.lipinski.passes_ro5 ? "Passes" : "Fails"} Ro5
                  </span>
                  <span className="text-neutral-500">HBD {checkResult.lipinski.hbd} HBA {checkResult.lipinski.hba}</span>
                </div>
              )}
            </div>
          )}
          {checkResult?.error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-3 text-[11px] text-red-400">
              {checkResult.error}
            </div>
          )}
        </div>
      </div>

      {show3DModal && atoms.length > 0 && (
        <Molecule3DModal
          atoms={viewerAtoms}
          adjacency={viewerAdj}
          onClose={() => setShow3DModal(false)}
        />
      )}
    </div>
  );
}
