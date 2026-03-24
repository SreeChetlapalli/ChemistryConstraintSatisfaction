const API = "/api";

async function fetchJSON(path, opts = {}) {
  const res = await fetch(`${API}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || "Request failed");
  return data;
}

export const api = {
  getInfo: () => fetchJSON("/info"),
  getPresets: () => fetchJSON("/presets"),
  getNoiseSchedule: (T) => fetchJSON(`/noise-schedule?T=${T}`),
  checkReaction: (body) =>
    fetchJSON("/check-reaction", { method: "POST", body: JSON.stringify(body) }),
  checkIntermediate: (body) =>
    fetchJSON("/check-intermediate", { method: "POST", body: JSON.stringify(body) }),
  runSupervisor: (body) =>
    fetchJSON("/run-supervisor", { method: "POST", body: JSON.stringify(body) }),
  runBenchmark: (body) =>
    fetchJSON("/benchmark", { method: "POST", body: JSON.stringify(body) }),
  trainModel: (body) =>
    fetchJSON("/train", { method: "POST", body: JSON.stringify(body) }),
  runMonteCarlo: (body) =>
    fetchJSON("/monte-carlo", { method: "POST", body: JSON.stringify(body) }),
  runPathway: (body) =>
    fetchJSON("/pathway", { method: "POST", body: JSON.stringify(body) }),
};
