import React, { useEffect, useState } from 'react';
import { api, MapOverlayInfo } from '../api/client';

interface MapViewProps {
  onSelectSample: (sampleId: string) => void;
}

export const MapView: React.FC<MapViewProps> = ({ onSelectSample }) => {
  const [overlays, setOverlays] = useState<MapOverlayInfo[]>([]);
  const [selectedOverlay, setSelectedOverlay] = useState<MapOverlayInfo | null>(null);
  const [sarOpacity, setSarOpacity] = useState<number>(100);
  const [predOpacity, setPredOpacity] = useState<number>(70);
  const [gtOpacity, setGtOpacity] = useState<number>(0);
  const [drawerOpen, setDrawerOpen] = useState<boolean>(true);

  useEffect(() => {
    api.getMapOverlays()
      .then((items) => {
        setOverlays(items);
        if (items.length > 0) {
          setSelectedOverlay(items[0]);
        }
      })
      .catch((err) => console.error('Failed to load map overlays:', err));
  }, []);

  return (
    <div className="relative flex-1 h-full map-bg overflow-hidden flex bg-background">
      {/* Map Canvas Background */}
      <div className="absolute inset-0 z-0 opacity-40">
        {selectedOverlay?.sar_tile_url ? (
          <img
            src={selectedOverlay.sar_tile_url}
            alt="Map Base Layer"
            className="w-full h-full object-cover filter grayscale"
            style={{ opacity: sarOpacity / 100 }}
          />
        ) : (
          <div className="w-full h-full bg-[#050a08]" />
        )}
      </div>

      {/* Flood Overlay Layer */}
      {selectedOverlay?.syneo_tile_url && (
        <div
          className="absolute inset-0 z-10 opacity-70 mix-blend-screen pointer-events-none"
          style={{ opacity: predOpacity / 100 }}
        >
          <img src={selectedOverlay.syneo_tile_url} alt="Flood Overlay" className="w-full h-full object-cover" />
        </div>
      )}

      {/* Geospatial Markers */}
      {overlays.map((item, idx) => {
        const topPos = `${30 + (idx % 3) * 20}%`;
        const leftPos = `${25 + (idx % 4) * 18}%`;
        const isSelected = selectedOverlay?.sample_id === item.sample_id;

        return (
          <div
            key={item.sample_id}
            style={{ top: topPos, left: leftPos }}
            onClick={() => {
              setSelectedOverlay(item);
              setDrawerOpen(true);
            }}
            className="absolute z-20 group cursor-pointer"
          >
            <div className="relative flex items-center justify-center">
              <div className={`absolute w-8 h-8 border ${isSelected ? 'border-primary' : 'border-tertiary'} rounded-full animate-ping opacity-50`}></div>
              <div className={`w-3 h-3 ${isSelected ? 'bg-primary' : 'bg-tertiary'} relative z-10 flex items-center justify-center`}>
                <div className={`w-[1px] h-6 ${isSelected ? 'bg-primary' : 'bg-tertiary'} absolute`}></div>
                <div className={`w-6 h-[1px] ${isSelected ? 'bg-primary' : 'bg-tertiary'} absolute`}></div>
              </div>
            </div>
            <div className="absolute top-4 left-4 bg-surface-container border border-tertiary px-2 py-1 hidden group-hover:block whitespace-nowrap z-30">
              <span className="font-data-mono text-xs text-tertiary">
                {item.location_name} (LAT: {item.center_lat.toFixed(4)} / LON: {item.center_lon.toFixed(4)})
              </span>
            </div>
          </div>
        );
      })}

      {/* Floating Controls (Left Side) */}
      <div className="absolute top-6 left-6 z-30 flex flex-col gap-4 w-64">
        {/* Legend Panel */}
        <div className="glass-panel p-4 flex flex-col gap-3">
          <div className="font-label-caps text-[10px] text-on-surface-variant border-b border-outline-variant pb-2 mb-1">
            DETECTION LEGEND
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-tertiary opacity-80 border border-tertiary-fixed"></div>
              <span className="font-data-mono text-[11px] text-on-surface">TRUE WATER</span>
            </div>
            <span className="font-data-mono text-[10px] text-on-surface-variant">84.2%</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-error opacity-80 border border-error"></div>
              <span className="font-data-mono text-[11px] text-on-surface">FALSE POSITIVE</span>
            </div>
            <span className="font-data-mono text-[10px] text-on-surface-variant">12.1%</span>
          </div>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 bg-secondary opacity-80 border border-secondary border-dashed"></div>
              <span className="font-data-mono text-[11px] text-on-surface">MISSED / FN</span>
            </div>
            <span className="font-data-mono text-[10px] text-on-surface-variant">3.7%</span>
          </div>
        </div>

        {/* Layer Controls */}
        <div className="glass-panel p-4 flex flex-col gap-4">
          <div className="font-label-caps text-[10px] text-on-surface-variant border-b border-outline-variant pb-2 flex justify-between">
            <span>OPACITY CONTROLS</span>
            <span className="material-symbols-outlined text-[14px]">tune</span>
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex justify-between font-data-mono text-[10px]">
              <span className="text-on-surface">SAR BACKSCATTER (VV)</span>
              <span className="text-primary">{sarOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={sarOpacity}
              onChange={(e) => setSarOpacity(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex justify-between font-data-mono text-[10px]">
              <span className="text-on-surface">PREDICTION MASK</span>
              <span className="text-primary">{predOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={predOpacity}
              onChange={(e) => setPredOpacity(parseInt(e.target.value))}
              className="w-full"
            />
          </div>
          <div className="flex flex-col gap-1">
            <div className="flex justify-between font-data-mono text-[10px]">
              <span className="text-on-surface">GROUND TRUTH (JRC)</span>
              <span className="text-on-surface-variant">{gtOpacity}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              value={gtOpacity}
              onChange={(e) => setGtOpacity(parseInt(e.target.value))}
              className="w-full opacity-70"
            />
          </div>
        </div>
      </div>

      {/* Filter Chips (Bottom Center) */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 z-30 flex gap-2">
        {overlays.map((item) => {
          const isSelected = selectedOverlay?.sample_id === item.sample_id;
          return (
            <button
              key={item.sample_id}
              onClick={() => {
                setSelectedOverlay(item);
                setDrawerOpen(true);
              }}
              className={`font-data-mono text-[11px] px-3 py-1 border uppercase transition-colors cursor-pointer ${
                isSelected
                  ? 'bg-primary-container text-on-primary-container border-primary-container'
                  : 'glass-panel text-on-surface hover:border-primary hover:text-primary'
              }`}
            >
              {item.sample_id}
            </button>
          );
        })}
      </div>

      {/* Right Information Drawer */}
      {drawerOpen && selectedOverlay && (
        <div className="absolute right-0 top-0 bottom-0 w-96 bg-surface-container-high border-l border-outline-variant z-40 flex flex-col">
          {/* Drawer Header */}
          <div className="p-4 border-b border-primary border-t-2 flex justify-between items-start bg-surface-container-highest">
            <div>
              <div className="font-data-mono text-[10px] text-on-surface-variant mb-1">SELECTED REGION_ID</div>
              <div className="font-headline-sm text-headline-sm text-on-surface font-bold text-sm">
                {selectedOverlay.sample_id}
              </div>
              <div className="mt-2 inline-block px-2 py-0.5 bg-[#1E293B] border border-outline-variant font-data-mono text-[10px] text-tertiary">
                STATUS: ACTIVE SENSING
              </div>
            </div>
            <button onClick={() => setDrawerOpen(false)} className="text-on-surface-variant hover:text-on-surface cursor-pointer">
              <span className="material-symbols-outlined">close</span>
            </button>
          </div>

          {/* Drawer Content */}
          <div className="flex-1 overflow-y-auto p-4 flex flex-col gap-6">
            {/* Metadata Section */}
            <div className="border border-outline-variant p-3 bg-surface-container">
              <div className="font-label-caps text-[11px] text-on-surface-variant border-b border-outline-variant pb-2 mb-3">
                METADATA
              </div>
              <div className="grid grid-cols-2 gap-y-3 gap-x-4">
                <div>
                  <div className="font-data-mono text-[9px] text-on-surface-variant">LOCATION</div>
                  <div className="font-data-mono text-xs text-on-surface mt-0.5">{selectedOverlay.location_name}</div>
                </div>
                <div>
                  <div className="font-data-mono text-[9px] text-on-surface-variant">SENSOR / MODE</div>
                  <div className="font-data-mono text-xs text-on-surface mt-0.5">Sentinel-1 / IW</div>
                </div>
                <div>
                  <div className="font-data-mono text-[9px] text-on-surface-variant">POLARIZATION</div>
                  <div className="font-data-mono text-xs text-on-surface mt-0.5">VV + VH</div>
                </div>
                <div>
                  <div className="font-data-mono text-[9px] text-on-surface-variant">COORDINATES</div>
                  <div className="font-data-mono text-xs text-on-surface mt-0.5">
                    {selectedOverlay.center_lat.toFixed(2)}N, {selectedOverlay.center_lon.toFixed(2)}E
                  </div>
                </div>
              </div>
            </div>

            {/* Preview Image */}
            <div className="border border-outline-variant bg-surface-container flex flex-col p-3">
              <div className="font-label-caps text-[11px] text-on-surface-variant border-b border-outline-variant pb-2 mb-3 flex justify-between items-center">
                <span>SAR TILE PREVIEW</span>
                <span className="material-symbols-outlined text-[14px]">policy</span>
              </div>
              <div className="aspect-video bg-surface overflow-hidden border border-tech">
                <img src={selectedOverlay.sar_tile_url} alt="Tile Preview" className="w-full h-full object-cover" />
              </div>
              <button
                onClick={() => onSelectSample(selectedOverlay.sample_id)}
                className="w-full mt-3 bg-primary text-on-primary font-label-caps py-2 hover:bg-primary-fixed transition-colors border-none rounded-none text-xs font-bold cursor-pointer"
              >
                OPEN IN WORKSPACE
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
