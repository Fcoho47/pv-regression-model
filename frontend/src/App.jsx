import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [plants, setPlants] = useState([]);
  const [selectedPlant, setSelectedPlant] = useState("");
  const [series, setSeries] = useState([]);
  const [plantMetrics, setPlantMetrics] = useState(null);
  const [plantTable, setPlantTable] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [sortKey, setSortKey] = useState("RMSE");
  const [sortDir, setSortDir] = useState("desc"); // asc | desc
  const [meta, setMeta] = useState({});
  const [xRange, setXRange] = useState(null); // rango de fechas sincronizado
  const plantMap = useMemo(
    () =>
      plants.reduce((acc, p) => {
        acc[p.id] = { nombre: p.nombre, potencia: p.potencia };
        return acc;
      }, {}),
    [plants]
  );

  useEffect(() => {
    fetch(`${API_URL}/models`)
      .then((res) => res.json())
      .then((data) => {
        setModels(data.models || []);
        if (data.models?.length) {
          setSelectedModel(data.models[0].id);
        }
      })
      .catch((err) => setError(err.message));
  }, []);

  useEffect(() => {
    if (!selectedModel) return;
    setPlants([]);
    setSelectedPlant("");
    setPlantTable([]);
    setMeta({});
    fetch(`${API_URL}/models/${selectedModel}/plants`)
      .then((res) => res.json())
      .then((data) => {
        const opts = (data.plants || []).map((p) => ({
          id: p.id ?? p.planta_id ?? p,
          nombre: p.nombre,
          potencia: p.potencia,
        }));
        setPlants(opts);
        if (opts.length) setSelectedPlant(opts[0].id);
      })
      .catch((err) => setError(err.message));
    fetch(`${API_URL}/models/${selectedModel}/plants/metrics`)
      .then((res) => res.json())
      .then((data) => setPlantTable(data.plants || []))
      .catch((err) => setError(err.message));

    // Fetch metadata for target_transform (if available)
    fetch(`${API_URL}/models`)
      .then((res) => res.json())
      .then((data) => {
        const found = (data.models || []).find((m) => m.id === selectedModel);
        if (found) setMeta(found);
      })
      .catch(() => {});
  }, [selectedModel]);

  const loadSeries = () => {
    if (!selectedModel || !selectedPlant) return;
    setLoading(true);
    setError("");
    Promise.all([
      fetch(`${API_URL}/models/${selectedModel}/plant/${selectedPlant}`),
      fetch(`${API_URL}/models/${selectedModel}/plant/${selectedPlant}/metrics`),
    ])
      .then(async ([sRes, mRes]) => {
        if (!sRes.ok) throw new Error(await sRes.text());
        if (!mRes.ok) throw new Error(await mRes.text());
        const seriesData = await sRes.json();
        const metricsData = await mRes.json();
        setSeries(seriesData.data || []);
        setPlantMetrics(metricsData.metrics || null);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (selectedModel && selectedPlant) loadSeries();
  }, [selectedModel, selectedPlant]);

  const currentMeta = useMemo(() => {
    const found = models.find((m) => m.id === selectedModel);
    return found || {};
  }, [models, selectedModel]);

  const fechas = series.map((d) => d.fecha);
  const reales = series.map((d) => d.real);
  const preds = series.map((d) => d.pred);
  const diffs = series.map((d) => d.pred - d.real);
  const diffsAbs = series.map((d) => Math.abs(d.pred - d.real));
  const minDate = series.length ? series[0].fecha : null;
  const maxDate = series.length ? series[series.length - 1].fecha : null;

  const selectedPlantInfo = useMemo(
    () => plants.find((p) => p.id === selectedPlant) || {},
    [plants, selectedPlant]
  );

  return (
    <div className="app">
      <h1>Model Viewer</h1>
      <div className="layout">
        <div className="left">
          <div className="card">
            <div className="row">
              <label>
                Modelo:
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id}>
                      {m.id}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Planta (holdout):
                <select
                  value={selectedPlant}
                  onChange={(e) => setSelectedPlant(e.target.value)}
                >
                  {plants.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.id} {p.nombre ? `- ${p.nombre}` : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            {selectedPlantInfo?.nombre && (
              <div style={{ marginTop: 8, color: "#9fb3d3" }}>
                {selectedPlantInfo.nombre} | Potencia: {selectedPlantInfo.potencia ?? "-"}
              </div>
            )}
            {meta?.target_transform && (
              <div style={{ marginTop: 4, color: "#6fb1ff", fontSize: 12 }}>
                Transformación de target: {meta.target_transform}
              </div>
            )}
            {error && <div className="empty">Error: {error}</div>}
            {!loading && !error && !series.length && (
              <div className="empty">Sin datos para mostrar.</div>
            )}
          </div>

          <div className="card">
            <h3>Métricas globales (train/test/holdout)</h3>
            <div className="metrics roomy">
              {currentMeta?.metrics ? (
                ["train", "test", "holdout"].map((split) => {
                  const ms = currentMeta.metrics?.[split];
                  return ms ? (
                    <div key={split} className="metric">
                      <div>
                        <strong>{split}</strong>
                      </div>
                      {Object.entries(ms).map(([k, v]) => (
                        <div key={k}>
                          {k}: {typeof v === "number" ? v.toFixed(3) : v}
                        </div>
                      ))}
                    </div>
                  ) : null;
                })
              ) : (
                <div className="empty">Sin métricas cargadas.</div>
              )}
            </div>
          </div>

          <div className="card">
            <h3>Métricas de la planta seleccionada</h3>
            <div className="metrics roomy">
              {plantMetrics ? (
                Object.entries(plantMetrics).map(([k, v]) => (
                  <div key={k} className="metric">
                    <strong>{k}</strong>: {typeof v === "number" ? v.toFixed(3) : v}
                  </div>
                ))
              ) : (
                <div className="empty">Selecciona planta para ver métricas.</div>
              )}
            </div>
          </div>

          <div className="card">
            <h3>Serie real vs pred (holdout)</h3>
            {series.length ? (
              <>
                <Plot
                  data={[
                    {
                      x: fechas,
                      y: reales,
                      type: "scatter",
                      mode: "lines+markers",
                      name: "Real",
                      line: { color: "#2dd4bf" },
                    },
                    {
                      x: fechas,
                      y: preds,
                      type: "scatter",
                      mode: "lines+markers",
                      name: "Pred",
                      line: { color: "#f97316", dash: "dot" },
                    },
                  ]}
                  layout={{
                    title: selectedPlant
                      ? `Planta ${selectedPlant}${
                          selectedPlantInfo?.nombre ? " - " + selectedPlantInfo.nombre : ""
                        }${
                          selectedPlantInfo?.potencia
                            ? ` | Potencia: ${selectedPlantInfo.potencia}`
                            : ""
                        }`
                      : "Serie",
                    autosize: true,
                    paper_bgcolor: "#141a33",
                    plot_bgcolor: "#0e1429",
                    font: { color: "#e9eef7" },
                    margin: { l: 40, r: 20, t: 24, b: 40 },
                    xaxis: {
                      title: "Fecha",
                      range: minDate && maxDate ? [minDate, maxDate] : undefined,
                      rangeslider: {
                        visible: true,
                        thickness: 0.14,
                        bgcolor: "#11182f",
                        bordercolor: "#233057",
                        borderwidth: 1,
                      },
                      rangeselector: {
                        bgcolor: "#1c2545",
                        activecolor: "#2c6dfc",
                        font: { color: "#e9eef7" },
                        buttons: [
                          { count: 30, label: "1m", step: "day", stepmode: "backward" },
                          { count: 90, label: "3m", step: "day", stepmode: "backward" },
                          { count: 180, label: "6m", step: "day", stepmode: "backward" },
                          { step: "all", label: "Todo" },
                        ],
                      },
                    },
                    yaxis: { title: TARGET_LABEL },
                  }}
                  onRelayout={(e) => {
                    if (e["xaxis.range[0]"] && e["xaxis.range[1]"]) {
                      setXRange([e["xaxis.range[0]"], e["xaxis.range[1]"]]);
                    } else if (e["xaxis.autorange"]) {
                      setXRange(null);
                    }
                  }}
                  style={{ width: "100%", height: "500px" }}
                  useResizeHandler
                />
                <div style={{ marginTop: 12 }}>
                  <Plot
                    data={[
                      {
                        x: fechas,
                        y: diffsAbs,
                        type: "bar",
                        name: "Error absoluto (|pred - real|)",
                        marker: { color: "#f97316" },
                      },
                    ]}
                    layout={{
                      autosize: true,
                      paper_bgcolor: "#141a33",
                      plot_bgcolor: "#0e1429",
                      font: { color: "#e9eef7" },
                      margin: { l: 40, r: 20, t: 10, b: 40 },
                      xaxis: {
                        title: "Fecha",
                        range: xRange
                          ? xRange
                          : minDate && maxDate
                            ? [minDate, maxDate]
                            : undefined,
                        rangeslider: {
                          visible: true,
                          thickness: 0.14,
                          bgcolor: "#11182f",
                          bordercolor: "#233057",
                          borderwidth: 1,
                        },
                        rangeselector: {
                          bgcolor: "#1c2545",
                          activecolor: "#2c6dfc",
                          font: { color: "#e9eef7" },
                          buttons: [
                            { count: 30, label: "1m", step: "day", stepmode: "backward" },
                            { count: 90, label: "3m", step: "day", stepmode: "backward" },
                            { count: 180, label: "6m", step: "day", stepmode: "backward" },
                            { step: "all", label: "Todo" },
                          ],
                        },
                      },
                      yaxis: { title: "Error absoluto (|pred - real|)", zeroline: true },
                    }}
                    style={{ width: "100%", height: "500px" }}
                    useResizeHandler
                  />
                </div>
              </>
            ) : (
              <div className="empty">Selecciona modelo y planta para visualizar.</div>
            )}
          </div>
        </div>

        <div className="right">
          <div className="card" style={{ height: "100%", overflow: "hidden" }}>
            <h3>Vista general por planta (holdout)</h3>
            {plantTable.length ? (
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      {["planta_id", "nombre", "potencia", "MAE", "RMSE", "MAPE_%", "R2"].map(
                        (col) => (
                          <th
                            key={col}
                            onClick={() => {
                              if (sortKey === col) {
                                setSortDir(sortDir === "asc" ? "desc" : "asc");
                              } else {
                                setSortKey(col);
                                setSortDir("desc");
                              }
                            }}
                            style={{ cursor: "pointer" }}
                          >
                            {col}
                            {sortKey === col ? (sortDir === "asc" ? " ↑" : " ↓") : ""}
                          </th>
                        )
                      )}
                    </tr>
                  </thead>
                  <tbody>
                    {plantTable
                      .slice()
                      .sort((a, b) => {
                        const dir = sortDir === "asc" ? 1 : -1;
                        const getVal = (row, key) => {
                          if (key === "planta_id") return row.planta_id;
                          if (key === "nombre")
                            return row.nombre || plantMap[row.planta_id]?.nombre || "";
                          if (key === "potencia")
                            return row.potencia ?? plantMap[row.planta_id]?.potencia ?? -Infinity;
                          return row.metrics ? row.metrics[key] : undefined;
                        };
                        const va = getVal(a, sortKey);
                        const vb = getVal(b, sortKey);
                        if (va == null && vb == null) return 0;
                        if (va == null) return 1;
                        if (vb == null) return -1;
                        if (typeof va === "string" || typeof vb === "string") {
                          return va.toString().localeCompare(vb.toString()) * dir;
                        }
                        return (va - vb) * dir;
                      })
                      .map((row) => (
                        <tr key={row.planta_id}>
                          <td>{row.planta_id}</td>
                          <td>{row.nombre || plantMap[row.planta_id]?.nombre || "-"}</td>
                          <td>{row.potencia ?? plantMap[row.planta_id]?.potencia ?? "-"}</td>
                          <td>{row.metrics?.MAE?.toFixed(3)}</td>
                          <td>{row.metrics?.RMSE?.toFixed(3)}</td>
                          <td>
                            {row.metrics && row.metrics["MAPE_%"] != null
                              ? row.metrics["MAPE_%"].toFixed(2)
                              : "N/A"}
                          </td>
                          <td>
                            {row.metrics?.R2 != null ? row.metrics.R2.toFixed(3) : "N/A"}
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <div className="empty">Sin datos de métricas por planta.</div>
            )}
          </div>

          <div className="card">
            <h3>Hiperparámetros del modelo</h3>
            {meta?.hyperparams ? (
              <div className="metrics roomy">
                {Object.entries(meta.hyperparams).map(([k, v]) => (
                  <div key={k} className="metric">
                    <strong>{k}</strong>: {typeof v === "number" ? v : String(v)}
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty">Sin hiperparámetros cargados.</div>
            )}
            {meta?.target_transform && (
              <div style={{ marginTop: 8, color: "#9fb3d3", fontSize: 12 }}>
                Transformación de target: {meta.target_transform}
              </div>
            )}
            {meta?.cv_metrics && (
              <div style={{ marginTop: 8, color: "#9fb3d3", fontSize: 12 }}>
                CV (promedio):{" "}
                {Object.entries(meta.cv_metrics)
                  .map(([k, v]) => `${k}: ${typeof v === "number" ? v.toFixed(3) : v}`)
                  .join(" | ")}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const TARGET_LABEL = "valor_teorico";
