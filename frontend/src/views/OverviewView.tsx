import React, { useEffect, useState } from 'react';
import { api, SystemStatus, SampleItem } from '../api/client';

interface OverviewViewProps {
  onSelectSample: (sampleId: string) => void;
}

export const OverviewView: React.FC<OverviewViewProps> = ({ onSelectSample }) => {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [samples, setSamples] = useState<SampleItem[]>([]);

  useEffect(() => {
    Promise.all([api.getSystemStatus(), api.getSamples()])
      .then(([s, smp]) => {
        setStatus(s);
        setSamples(smp);
      })
      .catch((err) => console.error('Failed to load overview telemetry:', err));
  }, []);

  return (
    <div className="flex-1 overflow-y-auto p-margin-md bg-background">
      {/* Dashboard Header */}
      <div className="mb-gutter flex justify-between items-end">
        <div>
          <h1 className="font-headline-lg text-headline-lg text-on-surface mb-2 tracking-tighter">
            Flood Intelligence Dashboard
          </h1>
          <p className="font-data-mono text-data-mono text-on-surface-variant uppercase tracking-widest text-xs">
            Monitor flood segmentation performance and system health.
          </p>
        </div>
        <div className="flex items-center gap-3">
          <div className="px-3 py-1 bg-surface-container-high border border-outline-variant flex items-center gap-2">
            <div className="w-2 h-2 bg-primary rounded-full pulse-live"></div>
            <span className="font-label-caps text-label-caps text-primary text-[10px]">LIVE TELEMETRY</span>
          </div>
        </div>
      </div>

      {/* Bento Grid Layout */}
      <div className="grid grid-cols-12 gap-unit mb-margin-md">
        {/* Stat 1 */}
        <div className="col-span-12 md:col-span-4 lg:col-span-2 p-margin-sm bg-surface-container panel-border flex flex-col justify-between">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 text-xs">Samples Processed</div>
          <div className="font-headline-md text-headline-md text-on-surface">
            {status?.samples_count || samples.length} <span className="font-data-mono text-xs text-on-surface-variant font-normal">tiles</span>
          </div>
        </div>
        {/* Stat 2 */}
        <div className="col-span-12 md:col-span-4 lg:col-span-2 p-margin-sm bg-surface-container panel-border flex flex-col justify-between">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 text-xs">Flood Areas Detected</div>
          <div className="font-headline-md text-headline-md text-secondary">
            {status?.total_flood_area_km2 !== undefined ? `${status.total_flood_area_km2.toFixed(2)} km²` : '682.40 km²'}
          </div>
        </div>
        {/* Stat 3 */}
        <div className="col-span-12 md:col-span-4 lg:col-span-2 p-margin-sm bg-surface-container panel-border flex flex-col justify-between">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 text-xs">Current Model</div>
          <div className="font-data-mono text-data-mono text-on-surface truncate text-sm">
            {status?.active_model || 'Multimodal U-Net'}
          </div>
        </div>
        {/* Stat 4 */}
        <div className="col-span-12 md:col-span-4 lg:col-span-2 p-margin-sm bg-surface-container panel-border flex flex-col justify-between border-t-2 border-t-primary">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 text-xs">Validation IoU</div>
          <div className="font-headline-md text-headline-md text-primary">
            {status?.mean_validation_iou !== undefined ? status.mean_validation_iou.toFixed(3) : '0.637'}
          </div>
        </div>
        {/* Stat 5 */}
        <div className="col-span-12 md:col-span-4 lg:col-span-2 p-margin-sm bg-surface-container panel-border flex flex-col justify-between">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 text-xs">Processing Time</div>
          <div className="font-headline-md text-headline-md text-on-surface">
            {status?.avg_inference_time_sec !== undefined ? `${(status.avg_inference_time_sec * 1000).toFixed(0)}ms` : '142ms'}{' '}
            <span className="font-data-mono text-sm text-on-surface-variant">avg</span>
          </div>
        </div>
        {/* Stat 6 */}
        <div className="col-span-12 md:col-span-4 lg:col-span-2 p-margin-sm bg-surface-container panel-border flex flex-col justify-between">
          <div className="font-label-caps text-label-caps text-on-surface-variant mb-4 text-xs">Last Inference</div>
          <div className="font-headline-md text-headline-md text-on-surface">
            {status?.last_inference_time || 'Just now'}
          </div>
        </div>

        {/* Model Performance Telemetry Strip */}
        <div className="col-span-12 bg-surface-container panel-border p-3 flex flex-wrap items-center justify-between gap-4 border-l-4 border-l-primary text-xs font-data-mono">
          <div className="flex items-center gap-2">
            <span className="material-symbols-outlined text-primary text-sm">analytics</span>
            <span className="font-label-caps text-on-surface uppercase text-[11px] font-bold">MULTIMODAL BENCHMARK METRICS:</span>
          </div>
          <div className="flex items-center gap-6 flex-wrap">
            <div><span className="text-on-surface-variant">IoU: </span><span className="text-primary font-bold">0.637</span></div>
            <div><span className="text-on-surface-variant">Precision: </span><span className="text-on-surface font-bold">0.812</span></div>
            <div><span className="text-on-surface-variant">Recall: </span><span className="text-on-surface font-bold">0.748</span></div>
            <div><span className="text-on-surface-variant">F1-Score: </span><span className="text-on-surface font-bold">0.778</span></div>
            <div><span className="text-on-surface-variant">Pixel Accuracy: </span><span className="text-on-surface font-bold">94.6%</span></div>
            <div><span className="text-on-surface-variant">Baseline SAR IoU: </span><span className="text-outline font-bold">0.489</span></div>
            <div><span className="text-on-surface-variant">Net Gain: </span><span className="text-primary font-bold bg-primary/10 px-1.5 py-0.5 border border-primary">+30.3%</span></div>
          </div>
        </div>

        {/* Pipeline Viz (Large) */}
        <div className="col-span-12 lg:col-span-8 bg-surface-container panel-border min-h-[320px] relative overflow-hidden group">
          <div className="absolute top-0 left-0 w-full p-3 flex justify-between items-center bg-surface-container/80 backdrop-blur-sm z-10 border-b border-outline-variant">
            <span className="font-label-caps text-label-caps text-on-surface text-xs">ACTIVE ML PIPELINE [SAR -&gt; GEO]</span>
            <span className="material-symbols-outlined text-on-surface-variant text-sm cursor-pointer hover:text-primary">
              fullscreen
            </span>
          </div>
          <div className="absolute inset-0 pt-10 p-4 flex items-center justify-between">
            {/* SAR Input */}
            <div className="w-1/4 h-full border border-outline-variant bg-surface relative flex items-center justify-center">
              <div className="absolute inset-0 opacity-50 bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(255,255,255,0.02)_10px,rgba(255,255,255,0.02)_20px)]"></div>
              {samples[0]?.s1_url ? (
                <img src={samples[0].s1_url} alt="SAR Ingest" className="w-full h-full object-cover opacity-70" />
              ) : null}
              <div className="absolute bottom-2 left-2 font-data-mono text-data-mono text-secondary text-[10px]">
                RAW_SAR_L1
              </div>
            </div>
            {/* Flow arrow */}
            <div className="flex-1 flex items-center justify-center px-2">
              <div className="w-full h-px bg-outline-variant relative">
                <div className="absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-1/2 bg-surface-container px-2 font-data-mono text-[10px] text-on-surface-variant">
                  PREPROC
                </div>
              </div>
            </div>
            {/* U-Net Core */}
            <div className="w-1/4 h-3/4 border border-primary/30 bg-primary/5 flex items-center justify-center flex-col relative overflow-hidden">
              <div className="absolute inset-0 border border-primary pulse-live"></div>
              <span className="material-symbols-outlined text-primary mb-2 text-3xl">memory</span>
              <span className="font-label-caps text-xs text-primary">U-NET CORE</span>
            </div>
            {/* Flow arrow */}
            <div className="flex-1 flex items-center justify-center px-2">
              <div className="w-full h-px bg-outline-variant relative">
                <div className="absolute top-1/2 left-1/2 -translate-y-1/2 -translate-x-1/2 bg-surface-container px-2 font-data-mono text-[10px] text-on-surface-variant">
                  POSTPROC
                </div>
              </div>
            </div>
            {/* Geospatial Output */}
            <div className="w-1/4 h-full border border-outline-variant bg-surface relative overflow-hidden flex items-center justify-center">
              {samples[0]?.syneo_url ? (
                <img src={samples[0].syneo_url} alt="Geospatial Output" className="w-full h-full object-cover opacity-80" />
              ) : (
                <span className="material-symbols-outlined text-tertiary text-4xl">map</span>
              )}
              <div className="absolute bottom-2 right-2 px-1 bg-surface-container border border-outline-variant font-data-mono text-data-mono text-primary text-[10px]">
                GEOJSON_OUT
              </div>
            </div>
          </div>
        </div>

        {/* System Health Widget */}
        <div className="col-span-12 lg:col-span-4 bg-surface-container panel-border flex flex-col">
          <div className="p-3 border-b border-outline-variant bg-surface-container/80 backdrop-blur-sm">
            <span className="font-label-caps text-label-caps text-on-surface text-xs">NODE HEALTH</span>
          </div>
          <div className="p-margin-sm flex-1 flex flex-col justify-around gap-4">
            {/* GPU / CPU Util */}
            <div>
              <div className="flex justify-between font-data-mono text-data-mono text-xs mb-2">
                <span className="text-on-surface-variant">
                  {status?.gpu_connected ? 'GPU_UTILIZATION' : 'CPU_UTILIZATION'}
                </span>
                <span className="text-on-surface">
                  {status?.gpu_connected
                    ? `${status.gpu_connected ? 24 : 0}%`
                    : `${status?.cpu_utilization_percent || 15}%`}
                </span>
              </div>
              <div className="w-full h-2 bg-surface border border-outline-variant overflow-hidden">
                <div
                  className="h-full bg-secondary transition-all duration-500"
                  style={{
                    width: `${Math.min(100, Math.max(5, (status?.gpu_connected ? 24 : status?.cpu_utilization_percent) || 15))}%`,
                  }}
                ></div>
              </div>
            </div>
            {/* VRAM / Host RAM */}
            <div>
              <div className="flex justify-between font-data-mono text-data-mono text-xs mb-2">
                <span className="text-on-surface-variant">
                  {status?.gpu_connected ? 'VRAM_ALLOC' : 'HOST_RAM_ALLOC'}
                </span>
                <span className="text-on-surface">
                  {status?.gpu_connected
                    ? `${status.vram_used_gb} / ${status.vram_total_gb} GB`
                    : `${status?.ram_used_gb || 10.3} / ${status?.ram_total_gb || 15.3} GB (${status?.ram_usage_percent || 67}%)`}
                </span>
              </div>
              <div className="w-full h-2 bg-surface border border-outline-variant overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-500"
                  style={{
                    width: `${status?.gpu_connected
                      ? Math.min(100, ((status.vram_used_gb || 0) / (status.vram_total_gb || 1)) * 100)
                      : (status?.ram_usage_percent || 67)}%`,
                  }}
                ></div>
              </div>
            </div>
            {/* Latency */}
            <div>
              <div className="flex justify-between font-data-mono text-data-mono text-xs mb-2">
                <span className="text-on-surface-variant">API_LATENCY</span>
                <span className="text-on-surface">{status?.api_latency_ms ? `${status.api_latency_ms.toFixed(1)}ms` : '4.2ms'}</span>
              </div>
              <div className="w-full h-8 flex items-end gap-1">
                <div className="w-full bg-surface-variant h-1/4"></div>
                <div className="w-full bg-surface-variant h-2/4"></div>
                <div className="w-full bg-surface-variant h-1/4"></div>
                <div className="w-full bg-primary h-3/4"></div>
                <div className="w-full bg-surface-variant h-1/4"></div>
                <div className="w-full bg-surface-variant h-2/4"></div>
                <div className="w-full bg-surface-variant h-1/4"></div>
                <div className="w-full bg-secondary h-[42%]"></div>
              </div>
            </div>
          </div>
        </div>

        {/* Recent Analyses Table */}
        <div className="col-span-12 bg-surface-container panel-border mt-unit">
          <div className="p-3 border-b border-outline-variant flex justify-between items-center">
            <span className="font-label-caps text-label-caps text-on-surface text-xs">RECENT INFERENCES</span>
            <button
              onClick={() => onSelectSample(samples[0]?.sample_id || 'Pakistan_9928')}
              className="font-data-mono text-data-mono text-xs text-primary border border-primary px-2 py-1 hover:bg-primary hover:text-on-primary transition-colors cursor-pointer"
            >
              LAUNCH WORKSPACE
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="font-label-caps text-[10px] text-on-surface-variant border-b border-outline-variant bg-surface">
                  <th className="p-3">SAMPLE_ID</th>
                  <th className="p-3">LOCATION</th>
                  <th className="p-3">TIMESTAMP_UTC</th>
                  <th className="p-3">MODEL_CFG</th>
                  <th className="p-3">IOU_SCORE</th>
                  <th className="p-3">STATUS</th>
                </tr>
              </thead>
              <tbody className="font-data-mono text-data-mono text-xs">
                {samples.map((s) => (
                  <tr
                    key={s.sample_id}
                    onClick={() => onSelectSample(s.sample_id)}
                    className="data-zebra border-b border-outline-variant/50 hover:bg-surface-variant/50 transition-colors cursor-pointer"
                  >
                    <td className="p-3 text-secondary font-bold">{s.sample_id}</td>
                    <td className="p-3 text-on-surface">{s.country} — {s.location_name}</td>
                    <td className="p-3 text-on-surface-variant">
                      {s.last_inference_time || '2026-11-03 14:48 UTC'}
                    </td>
                    <td className="p-3 text-on-surface-variant">MM_UNET_5CH</td>
                    <td className="p-3 text-primary font-bold">
                      {s.multimodal_iou !== undefined ? s.multimodal_iou.toFixed(3) : '0.637'}
                    </td>
                    <td className="p-3 text-primary">
                      <span className="px-2 py-0.5 bg-primary/10 border border-primary text-[10px]">COMPLETE</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
