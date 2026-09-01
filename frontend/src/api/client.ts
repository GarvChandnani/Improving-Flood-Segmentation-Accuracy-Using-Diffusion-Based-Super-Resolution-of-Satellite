/**
 * FloodSense API Client
 * Connects the React Stitch frontend directly to the FastAPI ML backend.
 */

export interface SystemStatus {
  backend_status: string;
  gpu_connected: boolean;
  gpu_name: string;
  cuda_available: boolean;
  vram_used_gb: number;
  vram_total_gb: number;
  api_latency_ms: number;
  active_model: string;
  active_checkpoint: string;
  model_loaded: boolean;
  samples_count: number;
  checkpoints_count: number;
}

export interface SampleItem {
  sample_id: string;
  country: string;
  chip_id: string;
  location_name: string;
  center_lat: number;
  center_lon: number;
  bounds: number[];
  crs: string;
  dimensions: string;
  channels: number;
  s1_url: string;
  label_url?: string;
  syneo_url?: string;
  has_ground_truth: boolean;
  has_syneo: boolean;
}

export interface ModelInfo {
  model_id: string;
  name: string;
  architecture: string;
  in_channels: number;
  out_channels: number;
  default_checkpoint: string;
  status: string;
  description: string;
}

export interface CheckpointInfo {
  filename: string;
  model_type: string;
  size_mb: number;
  sha256: string;
  path: string;
  modified_time: string;
  active: boolean;
}

export interface MetricsDict {
  iou: number;
  precision: number;
  recall: number;
  f1: number;
  accuracy: number;
}

export interface InferenceResponse {
  job_id: string;
  sample_id: string;
  model_type: string;
  prob_map_url: string;
  binary_mask_url: string;
  sar_preview_url: string;
  syneo_preview_url?: string;
  ground_truth_url?: string;
  metrics?: MetricsDict;
  inference_time_sec: number;
  checkpoint_used: string;
  timestamp: string;
}

export interface SynEOGenerateResponse {
  sample_id: string;
  syneo_url: string;
  generation_time_sec: number;
  status: string;
}

export interface RegionalMetrics {
  region: string;
  iou: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface EvaluationResponse {
  run_id: string;
  sample_id: string;
  location_name: string;
  baseline_metrics: MetricsDict;
  multimodal_metrics: MetricsDict;
  iou_baseline: number;
  iou_multimodal: number;
  iou_improvement_percent: number;
  visual_ground_truth_url: string;
  visual_baseline_pred_url: string;
  visual_multimodal_pred_url: string;
  visual_error_diff_url: string;
  regional_breakdown: RegionalMetrics[];
}

export interface MapOverlayInfo {
  sample_id: string;
  location_name: string;
  center_lat: number;
  center_lon: number;
  bounds: number[];
  sar_tile_url: string;
  syneo_tile_url?: string;
  prediction_tile_url?: string;
  ground_truth_tile_url?: string;
  metrics?: MetricsDict;
}

export interface TrainingRunInfo {
  run_id: string;
  model_name: string;
  status: string;
  current_epoch: number;
  total_epochs: number;
  loss: number;
  iou: number;
  learning_rate: number;
  checkpoint_saved: string;
  message: string;
}

const API_BASE = '/api';

async function request<T>(endpoint: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${endpoint}`, options);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

export const api = {
  getSystemStatus: () => request<SystemStatus>('/system/status'),
  getSamples: () => request<SampleItem[]>('/samples'),
  getSample: (id: string) => request<SampleItem>(`/samples/${id}`),
  
  uploadSarFile: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await fetch(`${API_BASE}/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: `Upload failed with status ${res.status}` }));
      throw new Error(err.detail || `Upload failed with status ${res.status}`);
    }
    return res.json() as Promise<SampleItem>;
  },

  getModels: () => request<ModelInfo[]>('/models'),
  getCheckpoints: () => request<CheckpointInfo[]>('/checkpoints'),
  
  loadCheckpoint: (model_type: string, checkpoint_path: string) =>
    request<{ status: string; message: string }>('/checkpoints/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model_type, checkpoint_path }),
    }),

  runBaselineInference: (sample_id: string, threshold = 0.5) =>
    request<InferenceResponse>('/inference/baseline', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id, model_type: 'baseline', threshold }),
    }),

  generateSynEO: (sample_id: string) =>
    request<SynEOGenerateResponse>('/syn-eo/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id }),
    }),

  runMultimodalInference: (sample_id: string, threshold = 0.5) =>
    request<InferenceResponse>('/inference/multimodal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id, model_type: 'multimodal', threshold }),
    }),

  evaluate: (sample_id?: string, threshold = 0.5) =>
    request<EvaluationResponse>('/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sample_id, threshold }),
    }),

  getMapOverlays: () => request<MapOverlayInfo[]>('/map/samples'),
  getTrainingRuns: () => request<TrainingRunInfo[]>('/training/runs'),
};
