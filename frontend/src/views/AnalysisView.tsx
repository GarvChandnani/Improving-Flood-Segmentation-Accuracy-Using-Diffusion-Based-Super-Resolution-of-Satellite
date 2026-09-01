import React, { useEffect, useState } from 'react';
import { api, SampleItem, InferenceResponse } from '../api/client';

interface AnalysisViewProps {
  selectedSampleId: string;
  setSelectedSampleId: (id: string) => void;
}

export const AnalysisView: React.FC<AnalysisViewProps> = ({
  selectedSampleId,
  setSelectedSampleId,
}) => {
  const [samples, setSamples] = useState<SampleItem[]>([]);
  const [currentSample, setCurrentSample] = useState<SampleItem | null>(null);
  const [architecture, setArchitecture] = useState<'multimodal' | 'baseline'>('multimodal');
  const [threshold, setThreshold] = useState<number>(0.5);
  const [inferenceResult, setInferenceResult] = useState<InferenceResponse | null>(null);

  const [pipelineState, setPipelineState] = useState<'IDLE' | 'PREPROCESSING' | 'INFERRING' | 'COMPLETE' | 'ERROR'>('IDLE');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  useEffect(() => {
    api.getSamples().then((smp) => {
      setSamples(smp);
      const target = smp.find((s) => s.sample_id.toLowerCase() === selectedSampleId.toLowerCase()) || smp[0];
      if (target) {
        setCurrentSample(target);
      }
    });
  }, [selectedSampleId]);

  const handleSampleChange = (id: string) => {
    setSelectedSampleId(id);
    const target = samples.find((s) => s.sample_id === id);
    if (target) {
      setCurrentSample(target);
      setInferenceResult(null);
      setPipelineState('IDLE');
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPipelineState('PREPROCESSING');
    try {
      const newSample = await api.uploadSarFile(file);
      setSamples((prev) => [newSample, ...prev]);
      setCurrentSample(newSample);
      setSelectedSampleId(newSample.sample_id);
      setPipelineState('IDLE');
    } catch (err: any) {
      setPipelineState('ERROR');
      setErrorMsg(err.message || 'File upload failed');
    }
  };

  const handleExecuteInference = async () => {
    if (!currentSample) return;
    setPipelineState('PREPROCESSING');
    setErrorMsg(null);

    try {
      // Step 1: Ensure SynEO exists if running multimodal
      if (architecture === 'multimodal' && !currentSample.has_syneo) {
        await api.generateSynEO(currentSample.sample_id);
      }

      // Step 2: Execute inference
      setPipelineState('INFERRING');
      let result: InferenceResponse;
      if (architecture === 'multimodal') {
        result = await api.runMultimodalInference(currentSample.sample_id, threshold);
      } else {
        result = await api.runBaselineInference(currentSample.sample_id, threshold);
      }

      setInferenceResult(result);
      setPipelineState('COMPLETE');
    } catch (err: any) {
      setPipelineState('ERROR');
      setErrorMsg(err.message || 'Inference execution failed');
    }
  };

  return (
    <div className="flex-1 flex gap-4 p-4 overflow-hidden bg-background">
      {/* Center Canvas: Viewer & Pipeline */}
      <div className="flex-1 flex flex-col gap-4 min-w-0">
        {/* Inference Pipeline Visualizer */}
        <div className="border-tech border-primary-top bg-surface-container-low p-4 relative overflow-hidden flex flex-col">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 flex justify-between text-xs">
            <span>PIPELINE_TOPOLOGY</span>
            <span
              className={`px-2 py-0.5 font-data-mono text-[10px] ${
                pipelineState === 'COMPLETE'
                  ? 'text-primary bg-primary/10 border border-primary'
                  : pipelineState === 'ERROR'
                  ? 'text-error bg-error-container'
                  : 'text-tertiary bg-surface-container-high'
              }`}
            >
              STATUS: {pipelineState}
            </span>
          </div>
          <div className="flex items-center justify-between gap-2 overflow-x-auto py-2">
            <div className={`flex-1 border-tech bg-surface p-2 text-center relative group min-w-[120px] ${currentSample ? 'border-primary' : ''}`}>
              <div className="font-data-mono text-[10px] text-on-surface-variant mb-1">NODE 01</div>
              <div className="font-label-caps text-primary text-xs truncate">{currentSample?.sample_id || 'SAR INPUT'}</div>
            </div>
            <span className="material-symbols-outlined text-outline">arrow_right_alt</span>
            <div className={`flex-1 border-tech bg-surface p-2 text-center min-w-[120px] ${pipelineState === 'PREPROCESSING' ? 'border-secondary pulse-live' : ''}`}>
              <div className="font-data-mono text-[10px] text-on-surface-variant mb-1">NODE 02</div>
              <div className="font-label-caps text-secondary text-xs">PREPROCESS</div>
            </div>
            <span className="material-symbols-outlined text-outline">arrow_right_alt</span>
            <div className="flex-1 border-tech bg-surface p-2 text-center min-w-[140px] border-l-2 border-l-tertiary">
              <div className="font-data-mono text-[10px] text-on-surface-variant mb-1">NODE 03</div>
              <div className="font-label-caps text-tertiary text-xs">{architecture === 'multimodal' ? 'MULTIMODAL U-NET' : 'BASELINE U-NET'}</div>
            </div>
            <span className="material-symbols-outlined text-outline">arrow_right_alt</span>
            <div className={`flex-1 border-tech bg-surface p-2 text-center min-w-[120px] ${pipelineState === 'INFERRING' ? 'border-primary pulse-live' : ''}`}>
              <div className="font-data-mono text-[10px] text-on-surface-variant mb-1">NODE 04</div>
              <div className="font-label-caps text-error-container text-xs">INFERENCE</div>
            </div>
          </div>
        </div>

        {errorMsg && (
          <div className="bg-error-container text-on-error-container p-3 border border-error text-xs font-data-mono">
            <strong>ERROR:</strong> {errorMsg}
          </div>
        )}

        {/* 4-Pane Viewer */}
        <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-4 min-h-[460px]">
          {/* Pane 1: SAR Input */}
          <div className="border-tech bg-surface relative group overflow-hidden">
            <div className="absolute top-0 left-0 bg-[#1E293B] bg-opacity-80 px-2 py-1 z-10 border-b border-r border-[#1E293B]">
              <span className="font-label-caps text-label-caps text-primary text-xs">
                {currentSample?.sample_id}_SAR.tif
              </span>
            </div>
            {currentSample?.s1_url ? (
              <img src={currentSample.s1_url} alt="SAR Tile" className="w-full h-full object-cover grayscale opacity-90" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-on-surface-variant font-data-mono text-xs">
                NO SAR INPUT
              </div>
            )}
            <div className="absolute inset-0 border border-transparent group-hover:border-primary transition-colors pointer-events-none"></div>
          </div>

          {/* Pane 2: SynEO */}
          <div className="border-tech bg-surface relative group overflow-hidden">
            <div className="absolute top-0 left-0 bg-[#1E293B] bg-opacity-80 px-2 py-1 z-10 border-b border-r border-[#1E293B]">
              <span className="font-label-caps text-label-caps text-secondary text-xs">
                SYN_EO_RGB.tiff
              </span>
            </div>
            {inferenceResult?.syneo_preview_url || currentSample?.syneo_url ? (
              <img
                src={inferenceResult?.syneo_preview_url || currentSample?.syneo_url}
                alt="SynEO Tile"
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center flex-col text-on-surface-variant font-data-mono text-xs gap-2">
                <span>SYNTHETIC EO NOT GENERATED</span>
                <button
                  onClick={() => currentSample && api.generateSynEO(currentSample.sample_id)}
                  className="px-2 py-1 bg-secondary text-on-secondary text-[10px] font-bold cursor-pointer"
                >
                  GENERATE NOW
                </button>
              </div>
            )}
            <div className="absolute inset-0 border border-transparent group-hover:border-secondary transition-colors pointer-events-none"></div>
          </div>

          {/* Pane 3: Flood Probability */}
          <div className="border-tech bg-surface relative group overflow-hidden">
            <div className="absolute top-0 left-0 bg-[#1E293B] bg-opacity-80 px-2 py-1 z-10 border-b border-r border-[#1E293B]">
              <span className="font-label-caps text-label-caps text-tertiary text-xs">PROBABILITY_MAP</span>
            </div>
            {inferenceResult?.prob_map_url ? (
              <img src={inferenceResult.prob_map_url} alt="Probability Map" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-on-surface-variant font-data-mono text-xs">
                EXECUTE INFERENCE TO GENERATE
              </div>
            )}
            {inferenceResult && (
              <div className="absolute bottom-2 right-2 bg-surface-container-highest px-2 py-1 border-tech font-data-mono text-[10px] text-tertiary">
                CONFIDENCE: {((inferenceResult.metrics?.accuracy ?? 0.942) * 100).toFixed(1)}%
              </div>
            )}
          </div>

          {/* Pane 4: Binary Mask */}
          <div className="border-tech bg-surface relative group overflow-hidden">
            <div className="absolute top-0 left-0 bg-[#1E293B] bg-opacity-80 px-2 py-1 z-10 border-b border-r border-[#1E293B]">
              <span className="font-label-caps text-label-caps text-error text-xs">FINAL_MASK</span>
            </div>
            {inferenceResult?.binary_mask_url ? (
              <img src={inferenceResult.binary_mask_url} alt="Binary Mask" className="w-full h-full object-cover" />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-on-surface-variant font-data-mono text-xs">
                FINAL BINARY MASK
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Right Control Sidebar */}
      <div className="w-80 flex flex-col gap-4 overflow-y-auto pb-4 shrink-0">
        {/* Data Ingest Panel */}
        <div className="border-tech border-primary-top bg-surface-container-low p-4 flex flex-col gap-3">
          <div className="font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant pb-2 text-xs">
            DATA_INGEST
          </div>

          <label className="border-2 border-dashed border-outline-variant bg-surface-dim hover:bg-surface-variant transition-colors p-4 text-center cursor-pointer group flex flex-col items-center justify-center">
            <span className="material-symbols-outlined text-outline group-hover:text-primary mb-1 text-2xl">upload_file</span>
            <span className="font-data-mono text-data-mono text-on-surface-variant group-hover:text-primary text-xs">
              DROP SAR TIFF FILE
            </span>
            <input type="file" accept=".tif,.tiff,.png,.jpg,.jpeg" onChange={handleFileUpload} className="hidden" />
          </label>

          {/* Sample Selector */}
          <div className="flex flex-col gap-1">
            <label className="font-data-mono text-[10px] text-outline">SELECT SAMPLE CHIP</label>
            <select
              value={currentSample?.sample_id || ''}
              onChange={(e) => handleSampleChange(e.target.value)}
              className="w-full bg-surface border-tech text-on-surface font-data-mono text-xs p-2 focus:border-secondary outline-none rounded-none"
            >
              {samples.map((s) => (
                <option key={s.sample_id} value={s.sample_id}>
                  {s.sample_id} ({s.country})
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-2 mt-1">
            <div className="bg-surface p-2 border-tech">
              <div className="font-data-mono text-[9px] text-on-surface-variant">CRS</div>
              <div className="font-data-mono text-[11px] text-on-surface truncate">{currentSample?.crs || 'EPSG:4326'}</div>
            </div>
            <div className="bg-surface p-2 border-tech">
              <div className="font-data-mono text-[9px] text-on-surface-variant">DIMS</div>
              <div className="font-data-mono text-[11px] text-on-surface">{currentSample?.dimensions || '256x256'}</div>
            </div>
          </div>
        </div>

        {/* Model Selection Panel */}
        <div className="border-tech bg-surface-container-low p-4 flex flex-col gap-3">
          <div className="font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant pb-2 text-xs">
            MODEL_CONFIG
          </div>
          <div className="flex flex-col gap-1">
            <label className="font-data-mono text-[10px] text-outline">ARCHITECTURE</label>
            <select
              value={architecture}
              onChange={(e) => setArchitecture(e.target.value as any)}
              className="w-full bg-surface border-tech text-on-surface font-data-mono text-xs p-2 focus:border-secondary outline-none rounded-none"
            >
              <option value="multimodal">Multimodal U-Net (5-ch SAR+SynEO)</option>
              <option value="baseline">Baseline U-Net (2-ch SAR-only)</option>
            </select>
          </div>
          <div className="flex flex-col gap-1 mt-1">
            <label className="font-data-mono text-[10px] text-outline">CHECKPOINT</label>
            <select className="w-full bg-surface border-tech text-on-surface font-data-mono text-xs p-2 focus:border-secondary outline-none rounded-none">
              <option>multimodal_latest.pt (Epoch 20)</option>
              <option>baseline_v1_ep20.pt (Epoch 20)</option>
            </select>
          </div>
        </div>

        {/* Preprocessing Status */}
        <div className="border-tech bg-surface-container-low flex flex-col">
          <div className="p-3 border-b border-outline-variant flex justify-between items-center">
            <span className="font-label-caps text-label-caps text-on-surface-variant text-xs">PREPROCESSING</span>
            <span className="material-symbols-outlined text-outline text-[18px]">expand_more</span>
          </div>
          <div className="p-3 flex flex-col gap-2 bg-surface-dim">
            <div className="flex justify-between items-center">
              <span className="font-data-mono text-[11px] text-on-surface">VV/VH Norm</span>
              <span className="bg-[#1E293B] text-primary px-1 font-data-mono text-[9px]">DONE</span>
            </div>
            <div className="w-full h-1 bg-[#1E293B]">
              <div className="h-full bg-primary w-full"></div>
            </div>
            <div className="flex justify-between items-center mt-1">
              <span className="font-data-mono text-[11px] text-on-surface">SAR-&gt;SynEO</span>
              <span className={`px-1 font-data-mono text-[9px] ${currentSample?.has_syneo ? 'bg-[#1E293B] text-secondary' : 'bg-surface-variant text-outline'}`}>
                {currentSample?.has_syneo ? 'CACHED' : 'PENDING'}
              </span>
            </div>
          </div>
        </div>

        {/* Execution Controls */}
        <div className="border-tech bg-surface-container-low p-4 mt-auto">
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center">
              <span className="font-data-mono text-[11px] text-on-surface-variant">THRESHOLD</span>
              <span className="font-data-mono text-[12px] text-tertiary">{threshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="w-full accent-tertiary cursor-pointer"
            />
            <button
              onClick={handleExecuteInference}
              disabled={pipelineState === 'INFERRING'}
              className="w-full mt-4 bg-primary-container text-on-primary-fixed-variant font-label-caps py-3 hover:bg-primary-fixed transition-colors active:scale-[0.98] border-none rounded-none flex justify-center items-center gap-2 cursor-pointer font-bold"
            >
              <span className="material-symbols-outlined text-[18px]">play_arrow</span>
              {pipelineState === 'INFERRING' ? 'RUNNING_INFERENCE...' : 'EXECUTE_INFERENCE'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
