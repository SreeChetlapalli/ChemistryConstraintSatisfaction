import { ELEMENTS as ALL_ELEMENTS } from "../data/elements";

const MAX_VAL = Object.fromEntries(ALL_ELEMENTS.map((e) => [e.sym, e.maxVal]));

export function parseFormula(formula) {
  const str = formula.trim();
  let charge = 0;
  let core = str;

  const chargeMatch = core.match(/([+-])(\d*)$/);
  if (chargeMatch) {
    const sign = chargeMatch[1] === "+" ? 1 : -1;
    const mag = chargeMatch[2] ? parseInt(chargeMatch[2]) : 1;
    charge = sign * mag;
    core = core.slice(0, -chargeMatch[0].length);
  }

  const atoms = [];
  const re = /([A-Z][a-z]?)(\d*)/g;
  let m;
  while ((m = re.exec(core)) !== null) {
    if (!m[1]) continue;
    const elem = m[1];
    const count = m[2] ? parseInt(m[2]) : 1;
    for (let i = 0; i < count; i++) {
      atoms.push({ element: elem, bonds: 0, formal_charge: 0 });
    }
  }
  if (atoms.length === 0) return null;

  if (charge !== 0) {
    const heaviest = atoms.find((a) => a.element !== "H") || atoms[0];
    heaviest.formal_charge = charge;
  }

  const remaining = atoms.map((a) => MAX_VAL[a.element] || 4);
  for (let i = 0; i < atoms.length; i++) {
    for (let j = i + 1; j < atoms.length; j++) {
      if (remaining[i] <= 0 || remaining[j] <= 0) continue;
      if (atoms[i].element === "H" && atoms[j].element === "H") continue;
      const order = Math.min(remaining[i], remaining[j]);
      atoms[i].bonds += order;
      atoms[j].bonds += order;
      remaining[i] -= order;
      remaining[j] -= order;
    }
  }

  const name = formula.trim() || atoms.map((a) => a.element).join("");
  return { name, atoms };
}

export function parseEquation(equation) {
  const arrow = equation.includes("->")
    ? "->"
    : equation.includes("\u2192")
      ? "\u2192"
      : equation.includes("=>")
        ? "=>"
        : null;
  if (!arrow) return null;

  const [lhs, rhs] = equation.split(arrow).map((s) => s.trim());
  if (!lhs || !rhs) return null;

  const reactants = lhs.split("+").map((s) => parseFormula(s.trim())).filter(Boolean);
  const products = rhs.split("+").map((s) => parseFormula(s.trim())).filter(Boolean);

  if (reactants.length === 0 || products.length === 0) return null;
  return { reactants, products };
}
