import React from 'react';

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  status: any;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab, status }) => {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: 'dashboard' },
    { id: 'analysis', label: 'Analysis', icon: 'query_stats' },
    { id: 'models', label: 'Models', icon: 'memory' },
    { id: 'map', label: 'Map', icon: 'map' },
    { id: 'data', label: 'Data', icon: 'database' },
    { id: 'experiments', label: 'Experiments', icon: 'science' },
    { id: 'system', label: 'System', icon: 'analytics' },
  ];

  return (
    <nav className="h-screen w-80 fixed left-0 top-0 bg-surface-container flex flex-col border-r border-outline-variant py-4 z-50">
      <div className="px-margin-md mb-8">
        <div className="flex items-center gap-3 mb-2">
          <div className="w-8 h-8 rounded-none border-tech flex items-center justify-center bg-primary/10 border-primary">
            <span className="material-symbols-outlined text-primary text-xl">radar</span>
          </div>
          <div>
            <div className="font-headline-sm text-headline-sm text-primary uppercase tracking-tighter">FLOODSENSE</div>
            <div className="font-data-mono text-data-mono text-on-surface-variant text-[10px]">V2.4 SATELLITE OPS</div>
          </div>
        </div>
      </div>
      <div className="flex-1 flex flex-col gap-1 px-2 overflow-y-auto">
        {navItems.map((item) => {
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-3 px-4 py-3 w-full text-left transition-all duration-75 active:scale-[0.99] cursor-pointer ${
                isActive
                  ? 'bg-primary text-on-primary font-bold border-l-4 border-primary'
                  : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-variant border-l-4 border-transparent'
              }`}
            >
              <span className="material-symbols-outlined font-data-mono text-data-mono">{item.icon}</span>
              <span className="font-label-caps text-label-caps">{item.label}</span>
            </button>
          );
        })}
      </div>
      <div className="mt-auto px-4 py-3 border-t border-outline-variant flex flex-col gap-2">
        <div className="flex items-center justify-between font-data-mono text-data-mono text-[10px]">
          <div className="flex items-center gap-2 text-primary">
            <span className="material-symbols-outlined text-[14px]">lan</span>
            <span>{status?.gpu_name || 'GPU Connected'}</span>
          </div>
          <span className={`w-2 h-2 ${status?.gpu_connected ? 'bg-primary pulse-live' : 'bg-secondary'}`}></span>
        </div>
        <div className="flex items-center justify-between font-data-mono text-data-mono text-[10px]">
          <div className="flex items-center gap-2 text-primary">
            <span className="material-symbols-outlined text-[14px]">cloud_done</span>
            <span>API {status?.backend_status === 'online' ? 'Online' : 'Offline'}</span>
          </div>
          <span className="w-2 h-2 bg-primary"></span>
        </div>
        <div className="flex items-center justify-between font-data-mono text-data-mono text-[10px]">
          <div className="flex items-center gap-2 text-primary">
            <span className="material-symbols-outlined text-[14px]">check_circle</span>
            <span>{status?.active_model || 'Model Loaded'}</span>
          </div>
          <span className="w-2 h-2 bg-primary"></span>
        </div>
      </div>
    </nav>
  );
};
