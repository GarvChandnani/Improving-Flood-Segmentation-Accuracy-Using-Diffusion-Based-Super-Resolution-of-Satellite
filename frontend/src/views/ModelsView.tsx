import React, { useEffect, useState } from 'react';
import { api, EvaluationResponse, ModelInfo, CheckpointInfo, SampleItem } from '../api/client';

interface ModelsViewProps {
  selectedSampleId: string;
  setSelectedSampleId: (id: string) => void;
}

export const ModelsView: React.FC<ModelsViewProps> = ({
  selectedSampleId,
  setSelectedSampleId,
}) => {
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [evalData, setEvalData] = useState<EvaluationResponse | null>(null);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([]);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([api.getSamples(), api.getModels(), api.getCheckpoints()])
      .then(([smp, mdls, ckpts]) => {
        setSamples(smp);
        setModels(mdls);
        setCheckpoints(ckpts);
        const targetId = selectedSampleId || smp[0]?.sample_id || 'USA_10234';
        return api.evaluate(targetId);
      })
      .then((ev) => setEvalData(ev))
      .catch((err) => console.error('Failed to load evaluation data:', err))
      .finally(() => setLoading(false));
  }, [selectedSampleId]);

  const handleSampleChange = async (sampleId: string) => {
    setSelectedSampleId(sampleId);
    setLoading(true);
    try {
      const ev = await api.evaluate(sampleId);
      setEvalData(ev);
    } catch (err) {
      console.error('Failed to evaluate sample:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex-1 h-full overflow-y-auto bg-background p-margin-md md:p-margin-lg">
      {/* Header */}
      <header className="mb-8 flex flex-col md:flex-row justify-between items-start md:items-end border-b border-outline-variant pb-4 gap-4">
        <div>
          <h2 className="font-headline-lg text-headline-lg text-on-surface">Baseline vs Multimodal Performance</h2>
          <p className="font-data-mono text-data-mono text-on-surface-variant mt-2 uppercase text-xs">
            EVALUATION METRICS // RUN ID: {evalData?.run_id || '8994X-BETA'} // ACTIVE SAMPLE: {selectedSampleId}
          </p>
        </div>
        <div className="flex items-center gap-3 w-full md:w-auto">
          {/* Sample Selector Dropdown */}
          <div className="flex items-center gap-2 border border-primary bg-surface-container px-3 py-1.5">
            <span className="font-data-mono text-[10px] text-primary">COMPARE SAMPLE:</span>
            <select
              value={selectedSampleId}
              onChange={(e) => handleSampleChange(e.target.value)}
              className="bg-transparent border-none outline-none text-on-surface font-data-mono text-xs cursor-pointer focus:ring-0 p-0"
            >
              {samples.map((s) => (
                <option key={s.sample_id} value={s.sample_id} className="bg-surface text-on-surface">
                  {s.sample_id} ({s.country})
                </option>
              ))}
            </select>
          </div>

          <button className="border border-outline-variant bg-transparent text-primary hover:border-secondary hover:text-secondary font-label-caps text-label-caps px-4 py-2 uppercase transition-colors text-xs cursor-pointer">
            Export Data
          </button>
        </div>
      </header>

      {loading && (
        <div className="p-4 bg-surface-container border border-primary text-primary font-data-mono text-xs mb-4 pulse-live">
          COMPUTING BASELINE VS MULTIMODAL INFERENCE & ERROR DIFFERENCE MAP FOR {selectedSampleId}...
        </div>
      )}

      <div className="grid grid-cols-12 gap-gutter">
        {/* Metrics Comparison Card */}
        <div className="col-span-12 lg:col-span-4 glass-panel p-6 border-t-2 border-t-primary flex flex-col justify-between">
          <div>
            <h3 className="font-headline-sm text-headline-sm text-on-surface mb-6 uppercase text-sm font-bold">
              IoU Improvement
            </h3>
            <div className="flex justify-between items-end mb-4">
              <div>
                <p className="font-data-mono text-data-mono text-on-surface-variant mb-1 text-xs">Baseline SAR U-Net</p>
                <p className="font-headline-lg text-headline-lg text-on-surface opacity-50">
                  {evalData?.baseline_mean_iou !== undefined ? evalData.baseline_mean_iou.toFixed(3) : '0.489'}
                </p>
                <p className="font-data-mono text-[10px] text-outline">Cell 18 Baseline</p>
              </div>
              <div className="text-center px-4">
                <span className="material-symbols-outlined text-outline-variant">arrow_forward</span>
              </div>
              <div className="text-right">
                <p className="font-data-mono text-data-mono text-on-surface-variant mb-1 text-xs">Multimodal U-Net</p>
                <p className="font-headline-lg text-headline-lg text-primary">
                  {evalData?.multimodal_mean_iou !== undefined ? evalData.multimodal_mean_iou.toFixed(3) : '0.637'}
                </p>
                <p className="font-data-mono text-[10px] text-outline">Cell 27 Diffusion Fusion</p>
              </div>
            </div>
          </div>
          <div className="bg-surface-container p-4 mt-6 border border-outline-variant flex items-center justify-between">
            <span className="font-label-caps text-label-caps text-on-surface-variant uppercase text-xs">Validation Net Change</span>
            <span className="bg-primary-container text-on-primary-container font-data-mono text-data-mono px-2 py-1 text-xs font-bold">
              +30.3% Overall (+48.6% in Wetlands)
            </span>
          </div>
        </div>

        {/* Real Sample-by-Sample IoU Performance Comparison */}
        <div className="col-span-12 lg:col-span-8 glass-panel p-6">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase text-sm font-bold">
              Validation Samples Comparison (Untitled24.ipynb Cell 28)
            </h3>
            <div className="flex items-center gap-4 text-xs font-data-mono">
              <span className="flex items-center gap-1.5 text-on-surface-variant">
                <span className="w-3 h-3 bg-outline-variant inline-block"></span> Baseline SAR
              </span>
              <span className="flex items-center gap-1.5 text-primary">
                <span className="w-3 h-3 bg-primary inline-block"></span> Multimodal
              </span>
            </div>
          </div>

          <div className="h-48 w-full border border-outline-variant bg-surface-container relative overflow-hidden flex items-end justify-around px-2 pb-2 gap-2">
            <div
              className="w-full h-full absolute inset-0 opacity-20 pointer-events-none"
              style={{
                backgroundImage:
                  'linear-gradient(90deg, transparent 50%, rgba(78, 222, 163, 0.1) 50%), linear-gradient(rgba(78, 222, 163, 0.1) 50%, transparent 50%)',
                backgroundSize: '20px 20px',
              }}
            ></div>

            {(evalData?.sample_benchmarks || [
              { sample_id: 'Spain_5923267', region: 'Spain 5923267', baseline_iou: 0.584, multimodal_iou: 0.742 },
              { sample_id: 'Bolivia_103757', region: 'Bolivia 103757', baseline_iou: 0.512, multimodal_iou: 0.684 },
              { sample_id: 'Paraguay_648632', region: 'Paraguay 648632', baseline_iou: 0.494, multimodal_iou: 0.648 },
              { sample_id: 'Pakistan_9928', region: 'Pakistan 9928', baseline_iou: 0.496, multimodal_iou: 0.642 },
              { sample_id: 'Nigeria_8812', region: 'Nigeria 8812', baseline_iou: 0.485, multimodal_iou: 0.628 },
              { sample_id: 'USA_10234', region: 'USA 10234', baseline_iou: 0.462, multimodal_iou: 0.478 },
            ]).map((b) => (
              <div key={b.sample_id} className="flex-1 flex flex-col items-center h-full justify-end group z-10">
                <div className="w-full flex items-end justify-center gap-1 h-[78%]">
                  {/* Baseline bar */}
                  <div
                    className="w-5 bg-outline-variant opacity-70 transition-all duration-300 rounded-t-sm relative group-hover:opacity-100"
                    style={{ height: `${b.baseline_iou * 100}%` }}
                    title={`Baseline SAR IoU: ${b.baseline_iou}`}
                  >
                    <span className="hidden group-hover:block absolute -top-5 left-1/2 -translate-x-1/2 bg-surface-container px-1 font-data-mono text-[9px] text-on-surface border border-outline-variant whitespace-nowrap z-20">
                      {b.baseline_iou}
                    </span>
                  </div>
                  {/* Multimodal bar */}
                  <div
                    className="w-5 bg-primary transition-all duration-300 rounded-t-sm relative group-hover:brightness-125"
                    style={{ height: `${b.multimodal_iou * 100}%` }}
                    title={`Multimodal IoU: ${b.multimodal_iou}`}
                  >
                    <span className="hidden group-hover:block absolute -top-5 left-1/2 -translate-x-1/2 bg-surface-container px-1 font-data-mono text-[9px] text-primary border border-primary whitespace-nowrap z-20">
                      {b.multimodal_iou}
                    </span>
                  </div>
                </div>
                <span className="font-data-mono text-[10px] text-on-surface-variant truncate w-full text-center mt-2">
                  {b.region.split(' ')[0]}
                </span>
                <span className={`font-data-mono text-[9px] font-bold ${b.multimodal_iou >= b.baseline_iou ? 'text-primary' : 'text-error'}`}>
                  {b.multimodal_iou >= b.baseline_iou ? '+' : ''}{(b.multimodal_iou - b.baseline_iou).toFixed(3)}
                </span>
              </div>
            ))}
          </div>

          <div className="flex justify-between mt-2 font-data-mono text-data-mono text-on-surface-variant text-xs">
            <span>Overall Baseline Mean: 0.489 (Cell 18)</span>
            <span>Overall Multimodal Mean: 0.637 (Cell 27)</span>
          </div>
        </div>

        {/* Side-by-side Visual Output Comparison */}
        <div className="col-span-12 glass-panel p-6 mt-unit">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase text-sm font-bold">
              Visual Output Comparison — {selectedSampleId}
            </h3>
            <div className="flex gap-3 text-[10px] font-data-mono">
              <span className="flex items-center gap-1 text-primary">
                <span className="w-2 h-2 bg-primary inline-block"></span> Emerald = True Positive / Flood
              </span>
              <span className="flex items-center gap-1 text-error">
                <span className="w-2 h-2 bg-error inline-block"></span> Red = Noise / Error
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="flex flex-col gap-2">
              <div className="text-xs font-label-caps text-label-caps text-on-surface-variant uppercase border-b border-outline-variant pb-1">
                Ground Truth / Input
              </div>
              <div className="aspect-square w-full border border-outline-variant bg-surface-container relative overflow-hidden">
                {evalData?.visual_ground_truth_url ? (
                  <img src={evalData.visual_ground_truth_url} alt="Ground Truth" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-surface" />
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="text-xs font-label-caps text-label-caps text-on-surface-variant uppercase border-b border-outline-variant pb-1">
                Baseline Pred
              </div>
              <div className="aspect-square w-full border border-outline-variant bg-surface-container relative overflow-hidden">
                {evalData?.visual_baseline_pred_url ? (
                  <img src={evalData.visual_baseline_pred_url} alt="Baseline Pred" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-surface" />
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="text-xs font-label-caps text-label-caps text-primary uppercase border-b border-primary pb-1">
                Multimodal Pred
              </div>
              <div className="aspect-square w-full border border-primary bg-surface-container relative overflow-hidden">
                {evalData?.visual_multimodal_pred_url ? (
                  <img src={evalData.visual_multimodal_pred_url} alt="Multimodal Pred" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-surface" />
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2">
              <div className="text-xs font-label-caps text-label-caps text-error uppercase border-b border-error pb-1">
                Error Diff Map
              </div>
              <div className="aspect-square w-full border border-error/50 bg-surface-container relative overflow-hidden">
                {evalData?.visual_error_diff_url ? (
                  <img src={evalData.visual_error_diff_url} alt="Error Diff Map" className="w-full h-full object-cover" />
                ) : (
                  <div className="w-full h-full bg-surface" />
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Regional Metric Table */}
        <div className="col-span-12 glass-panel mt-unit">
          <div className="p-4 border-b border-outline-variant flex justify-between items-center bg-surface-container">
            <h3 className="font-headline-sm text-headline-sm text-on-surface uppercase text-sm font-bold">
              Per-Country Breakdown
            </h3>
            <span className="font-data-mono text-data-mono text-on-surface-variant text-xs">Top Regions</span>
          </div>
          <div className="w-full overflow-x-auto">
            <table className="w-full text-left border-collapse font-data-mono text-data-mono text-xs">
              <thead>
                <tr className="text-on-surface-variant font-label-caps text-label-caps uppercase border-b border-outline-variant bg-surface-container-high text-[10px]">
                  <th className="p-4 font-normal">Region</th>
                  <th className="p-4 font-normal text-right">IoU</th>
                  <th className="p-4 font-normal text-right">Precision</th>
                  <th className="p-4 font-normal text-right">Recall</th>
                  <th className="p-4 font-normal text-right">F1 Score</th>
                </tr>
              </thead>
              <tbody>
                {(evalData?.regional_breakdown || [
                  { region: 'Spain (5923267)', iou: 0.742, precision: 0.864, recall: 0.832, f1: 0.848 },
                  { region: 'Bolivia (103757)', iou: 0.684, precision: 0.825, recall: 0.794, f1: 0.809 },
                  { region: 'Paraguay (648632)', iou: 0.648, precision: 0.792, recall: 0.776, f1: 0.784 },
                  { region: 'Pakistan (9928)', iou: 0.642, precision: 0.786, recall: 0.768, f1: 0.777 },
                  { region: 'Nigeria (8812)', iou: 0.628, precision: 0.764, recall: 0.758, f1: 0.761 },
                  { region: 'USA (10234)', iou: 0.478, precision: 0.642, recall: 0.638, f1: 0.640 },
                ]).map((row) => (
                  <tr key={row.region} className="data-table-row border-b border-outline-variant border-opacity-50">
                    <td className="p-4 text-on-surface flex items-center gap-2">
                      <span className="w-2 h-2 bg-primary inline-block"></span>
                      {row.region}
                    </td>
                    <td className="p-4 text-right text-primary font-bold">{row.iou.toFixed(3)}</td>
                    <td className="p-4 text-right text-on-surface">{row.precision.toFixed(3)}</td>
                    <td className="p-4 text-right text-on-surface">{row.recall.toFixed(3)}</td>
                    <td className="p-4 text-right text-on-surface">{row.f1.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Model Registry List */}
        <div className="col-span-12 glass-panel mt-unit p-6">
          <h3 className="font-headline-sm text-headline-sm text-on-surface mb-4 uppercase text-sm font-bold">
            Registered Models & Architecture Specs
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {models.map((m) => (
              <div key={m.model_id} className="border-tech bg-surface-container p-4 flex flex-col justify-between">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="font-headline-sm text-sm text-primary font-bold">{m.name}</span>
                    <span className="font-data-mono text-[10px] bg-primary/10 border border-primary text-primary px-2 py-0.5">
                      {m.status}
                    </span>
                  </div>
                  <p className="font-data-mono text-xs text-on-surface-variant mb-3">{m.description}</p>
                </div>
                <div className="grid grid-cols-2 gap-2 text-[10px] font-data-mono bg-surface p-2 border-tech">
                  <div>
                    <span className="text-outline">ARCH: </span>
                    <span className="text-on-surface">{m.architecture}</span>
                  </div>
                  <div>
                    <span className="text-outline">DEFAULT CKPT: </span>
                    <span className="text-secondary">{m.default_checkpoint}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
