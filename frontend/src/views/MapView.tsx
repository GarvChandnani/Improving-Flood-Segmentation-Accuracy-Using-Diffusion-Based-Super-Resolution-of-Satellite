import React, { useEffect, useState } from 'react';
import { api, MapOverlayInfo } from '../api/client';

interface MapViewProps {
  selectedSampleId?: string;
  setSelectedSampleId?: (sampleId: string) => void;
  onSelectSample: (sampleId: string) => void;
}

export const MapView: React.FC<MapViewProps> = ({
  selectedSampleId,
  setSelectedSampleId,
  onSelectSample,
}) => {
  const [overlays, setOverlays] = useState<MapOverlayInfo[]>([]);
  const [selected, setSelected] = useState<MapOverlayInfo | null>(null);
  const [sarOpacity, setSarOpacity] = useState<number>(100);
  const [maskOpacity, setMaskOpacity] = useState<number>(80);
  const [loading, setLoading] = useState(true);

  const fetchOverlays = () => {
    setLoading(true);
    api.getMapOverlays()
      .then((items) => {
        setOverlays(items);
        if (items.length > 0) {
          // 1. Match selectedSampleId if present
          const match = items.find((o) => o.sample_id.toLowerCase() === selectedSampleId?.toLowerCase());
          // 2. Otherwise pick the newest uploaded sample
          const uploaded = items.find((o) => o.sample_id.startsWith('UPLOAD_'));
          // 3. Fallback to first item
          const target = match || uploaded || items[0];
          setSelected(target);
          if (setSelectedSampleId && target) {
            setSelectedSampleId(target.sample_id);
          }
        }
      })
      .catch((err) => console.error('Map overlay load error:', err))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    fetchOverlays();
  }, []);

  useEffect(() => {
    if (overlays.length > 0 && selectedSampleId) {
      const target = overlays.find((o) => o.sample_id.toLowerCase() === selectedSampleId.toLowerCase());
      if (target) setSelected(target);
    }
  }, [selectedSampleId, overlays]);

  // ── empty / loading states ────────────────────────────────────────────────
  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center bg-background text-on-surface-variant font-data-mono text-sm">
        <span className="material-symbols-outlined animate-spin mr-2">progress_activity</span>
        Loading map overlays…
      </div>
    );
  }

  if (overlays.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center bg-background gap-4">
        <span className="material-symbols-outlined text-5xl text-outline">satellite_alt</span>
        <div className="text-center">
          <div className="font-label-caps text-on-surface font-bold text-sm mb-1">NO INFERRED SAMPLES</div>
          <div className="font-data-mono text-[12px] text-on-surface-variant max-w-xs text-center">
            Upload a SAR image in the <strong>Query/Analysis</strong> tab, run inference, then return here to see the
            colour-coded flood map.
          </div>
        </div>
        <button
          onClick={fetchOverlays}
          className="mt-2 px-4 py-2 bg-primary text-on-primary font-label-caps text-xs cursor-pointer border-none"
        >
          REFRESH
        </button>
      </div>
    );
  }

  // ── main map view ─────────────────────────────────────────────────────────
  return (
    <div className="relative flex-1 h-full overflow-hidden flex bg-[#050a08]">

      {/* ── SAR base layer (full-brightness, no double-opacity wrapper) ── */}
      {selected?.sar_tile_url && (
        <img
          key={selected.sample_id + '_sar'}
          src={selected.sar_tile_url}
          alt="SAR base"
          className="absolute inset-0 w-full h-full object-contain"
          style={{ opacity: sarOpacity / 100 }}
        />
      )}

      {/* ── Colour-coded prediction overlay (RGBA PNG: blue=safe, red=risky) ── */}
      {selected?.prediction_tile_url && (
        <img
          key={selected.sample_id + '_mask'}
          src={selected.prediction_tile_url}
          alt="Flood prediction mask"
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          style={{ opacity: maskOpacity / 100 }}
        />
      )}

      {/* ── Legend (top-left) ── */}
      <div className="absolute top-5 left-5 z-30 glass-panel p-4 flex flex-col gap-3 w-56">
        <div className="font-label-caps text-[10px] text-on-surface-variant border-b border-outline-variant pb-2">
          FLOOD DETECTION LEGEND
        </div>

        {/* Safe area — blue */}
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-sm flex-shrink-0" style={{ background: '#3B82F6' }} />
          <div>
            <div className="font-data-mono text-[11px] text-on-surface">SAFE AREA</div>
            <div className="font-data-mono text-[9px] text-on-surface-variant">High-confidence flood pixel</div>
          </div>
        </div>

        {/* Risky — red */}
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-sm flex-shrink-0" style={{ background: '#EF4444' }} />
          <div>
            <div className="font-data-mono text-[11px] text-on-surface">HIGH RISK</div>
            <div className="font-data-mono text-[9px] text-on-surface-variant">Uncertain / low-confidence pixel</div>
          </div>
        </div>

        {/* No detection */}
        <div className="flex items-center gap-2">
          <div className="w-4 h-4 rounded-sm flex-shrink-0 bg-[#050a08] border border-outline-variant" />
          <div>
            <div className="font-data-mono text-[11px] text-on-surface">NO DETECTION</div>
            <div className="font-data-mono text-[9px] text-on-surface-variant">Dry land / no flood predicted</div>
          </div>
        </div>
      </div>

      {/* ── Opacity controls (bottom-left) ── */}
      <div className="absolute bottom-5 left-5 z-30 glass-panel p-4 flex flex-col gap-4 w-56">
        <div className="font-label-caps text-[10px] text-on-surface-variant border-b border-outline-variant pb-2">
          LAYER OPACITY
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between font-data-mono text-[10px]">
            <span className="text-on-surface">SAR IMAGE</span>
            <span className="text-primary">{sarOpacity}%</span>
          </div>
          <input
            type="range" min="0" max="100" value={sarOpacity}
            onChange={(e) => setSarOpacity(parseInt(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>

        <div className="flex flex-col gap-1">
          <div className="flex justify-between font-data-mono text-[10px]">
            <span className="text-on-surface">PREDICTION MASK</span>
            <span className="text-primary">{maskOpacity}%</span>
          </div>
          <input
            type="range" min="0" max="100" value={maskOpacity}
            onChange={(e) => setMaskOpacity(parseInt(e.target.value))}
            className="w-full accent-red-500"
          />
        </div>
      </div>

      {/* ── Sample selector chips (bottom-centre) ── */}
      <div className="absolute bottom-5 left-1/2 -translate-x-1/2 z-30 flex gap-2 flex-wrap justify-center max-w-lg">
        {overlays.map((item) => {
          const isSelected = selected?.sample_id === item.sample_id;
          return (
            <button
              key={item.sample_id}
              onClick={() => {
                setSelected(item);
                if (setSelectedSampleId) setSelectedSampleId(item.sample_id);
              }}
              className={`font-data-mono text-[11px] px-3 py-1 border uppercase transition-colors cursor-pointer ${
                isSelected
                  ? 'bg-primary text-on-primary border-primary'
                  : 'glass-panel text-on-surface hover:border-primary hover:text-primary'
              }`}
            >
              {item.sample_id}
            </button>
          );
        })}
      </div>

      {/* ── Info badge (top-right) ── */}
      {selected && (
        <div className="absolute top-5 right-5 z-30 glass-panel p-4 w-72 flex flex-col gap-3">
          <div className="font-label-caps text-[10px] text-on-surface-variant border-b border-outline-variant pb-2 flex justify-between">
            <span>ACTIVE SAMPLE</span>
            <button
              onClick={() => onSelectSample(selected.sample_id)}
              className="text-primary hover:text-primary-fixed font-data-mono text-[9px] cursor-pointer underline"
            >
              OPEN IN WORKSPACE →
            </button>
          </div>
          <div className="font-data-mono text-xs text-on-surface font-bold truncate">{selected.sample_id}</div>
          <div className="font-data-mono text-[10px] text-on-surface-variant">{selected.location_name}</div>
          <div className="grid grid-cols-2 gap-2 mt-1">
            <div className="bg-surface p-2 border border-outline-variant">
              <div className="font-data-mono text-[9px] text-on-surface-variant">LATITUDE</div>
              <div className="font-data-mono text-[11px] text-on-surface">{selected.center_lat.toFixed(4)}°</div>
            </div>
            <div className="bg-surface p-2 border border-outline-variant">
              <div className="font-data-mono text-[9px] text-on-surface-variant">LONGITUDE</div>
              <div className="font-data-mono text-[11px] text-on-surface">{selected.center_lon.toFixed(4)}°</div>
            </div>
          </div>
          {/* Thumbnail of prediction mask */}
          {selected.prediction_tile_url && (
            <div className="mt-1">
              <div className="font-data-mono text-[9px] text-on-surface-variant mb-1">PREDICTION MASK PREVIEW</div>
              <div className="border border-outline-variant overflow-hidden" style={{ height: 120 }}>
                <img
                  src={selected.prediction_tile_url}
                  alt="Mask preview"
                  className="w-full h-full object-contain bg-[#050a08]"
                />
              </div>
            </div>
          )}
          <button
            onClick={fetchOverlays}
            className="mt-1 w-full py-1.5 bg-surface-container text-on-surface-variant font-data-mono text-[10px] border border-outline-variant hover:border-primary hover:text-primary transition-colors cursor-pointer"
          >
            ↻ REFRESH OVERLAYS
          </button>
        </div>
      )}
    </div>
  );
};
