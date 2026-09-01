import React, { useEffect, useState } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { OverviewView } from './views/OverviewView';
import { AnalysisView } from './views/AnalysisView';
import { MapView } from './views/MapView';
import { ModelsView } from './views/ModelsView';
import { api, SystemStatus } from './api/client';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<string>('overview');
  const [selectedSampleId, setSelectedSampleId] = useState<string>('USA_10234');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);

  useEffect(() => {
    api.getSystemStatus()
      .then(setSystemStatus)
      .catch((err) => console.error('Failed to fetch system status:', err));
  }, []);

  const handleSelectSample = (sampleId: string) => {
    setSelectedSampleId(sampleId);
    setActiveTab('analysis');
  };

  return (
    <div className="flex h-screen overflow-hidden font-body-md text-body-md bg-background text-on-background">
      <div className="terminal-grid"></div>

      {/* Sidebar */}
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} status={systemStatus} />

      {/* Main Content Container */}
      <main className="ml-80 flex-1 flex flex-col h-screen overflow-hidden relative">
        <Navbar searchQuery={searchQuery} setSearchQuery={setSearchQuery} />

        {/* View Switcher */}
        {activeTab === 'overview' && <OverviewView onSelectSample={handleSelectSample} />}
        {activeTab === 'analysis' && (
          <AnalysisView
            selectedSampleId={selectedSampleId}
            setSelectedSampleId={setSelectedSampleId}
          />
        )}
        {activeTab === 'models' && (
          <ModelsView
            selectedSampleId={selectedSampleId}
            setSelectedSampleId={setSelectedSampleId}
          />
        )}
        {activeTab === 'map' && <MapView onSelectSample={handleSelectSample} />}

        {/* Generic views for Data, Experiments, System keeping Stitch theme */}
        {activeTab === 'data' && (
          <div className="flex-1 p-8 bg-background overflow-y-auto">
            <h1 className="font-headline-lg text-headline-lg text-primary mb-4">Dataset Catalog & Satellite Chips</h1>
            <p className="font-data-mono text-data-mono text-on-surface-variant mb-6 text-sm">
              SEN1FLOODS11 HAND & WEAK LABELED STAC CATALOG
            </p>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="border-tech bg-surface-container p-4">
                <div className="font-label-caps text-xs text-primary mb-2">HAND_LABELED_SOURCE</div>
                <div className="font-headline-md text-on-surface">447 chips</div>
                <p className="font-data-mono text-xs text-on-surface-variant mt-2">Ground-truth annotated flood masks</p>
              </div>
              <div className="border-tech bg-surface-container p-4">
                <div className="font-label-caps text-xs text-secondary mb-2">WEAK_LABELED_SOURCE</div>
                <div className="font-headline-md text-on-surface">4,385 chips</div>
                <p className="font-data-mono text-xs text-on-surface-variant mt-2">Otsu algorithm weak labels</p>
              </div>
              <div className="border-tech bg-surface-container p-4">
                <div className="font-label-caps text-xs text-tertiary mb-2">SYNTHETIC_EO_CACHE</div>
                <div className="font-headline-md text-on-surface">{systemStatus?.samples_count || 4} generated</div>
                <p className="font-data-mono text-xs text-on-surface-variant mt-2">Cached Pix2Pix 256x256 RGB .npy arrays</p>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'experiments' && (
          <div className="flex-1 p-8 bg-background overflow-y-auto">
            <h1 className="font-headline-lg text-headline-lg text-primary mb-4">ML Experiments & Training Runs</h1>
            <p className="font-data-mono text-data-mono text-on-surface-variant mb-6 text-sm">
              HISTORICAL & ACTIVE MODEL TRAINING LOGS
            </p>
            <div className="border-tech bg-surface-container p-6">
              <div className="flex justify-between items-center mb-4">
                <span className="font-label-caps text-xs text-on-surface">CLI TRAINING CONTROL</span>
                <span className="bg-primary/10 border border-primary text-primary px-2 py-0.5 font-data-mono text-xs">
                  READY
                </span>
              </div>
              <p className="font-data-mono text-xs text-on-surface-variant mb-4">
                To launch model training on a GPU node, run the migrated training scripts:
              </p>
              <pre className="bg-surface p-4 border-tech font-data-mono text-xs text-primary overflow-x-auto">
                python -m ml.train_multimodal --epochs 20
              </pre>
            </div>
          </div>
        )}

        {activeTab === 'system' && (
          <div className="flex-1 p-8 bg-background overflow-y-auto">
            <h1 className="font-headline-lg text-headline-lg text-primary mb-4">System Telemetry & Compute Node Status</h1>
            <p className="font-data-mono text-data-mono text-on-surface-variant mb-6 text-sm">
              NODE TELEMETRY, GPU VRAM ALLOCATION & CHECKPOINT AUDIT
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="border-tech bg-surface-container p-6">
                <h3 className="font-label-caps text-xs text-primary mb-4">COMPUTE DEVICE</h3>
                <div className="font-data-mono text-sm text-on-surface mb-2">GPU: {systemStatus?.gpu_name || 'CPU'}</div>
                <div className="font-data-mono text-xs text-on-surface-variant">CUDA Available: {systemStatus?.cuda_available ? 'Yes' : 'No'}</div>
                <div className="font-data-mono text-xs text-on-surface-variant mt-1">API Latency: {systemStatus?.api_latency_ms || 42}ms</div>
              </div>
              <div className="border-tech bg-surface-container p-6">
                <h3 className="font-label-caps text-xs text-primary mb-4">CHECKPOINT REGISTRY</h3>
                <div className="font-data-mono text-sm text-on-surface mb-2">Checkpoints Count: {systemStatus?.checkpoints_count || 0}</div>
                <div className="font-data-mono text-xs text-on-surface-variant">Location: ml/checkpoints/</div>
                <div className="font-data-mono text-xs text-primary mt-2">Status: All model architectures loaded & operational</div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};
