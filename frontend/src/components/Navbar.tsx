import React from 'react';

interface NavbarProps {
  searchQuery: string;
  setSearchQuery: (q: string) => void;
}

export const Navbar: React.FC<NavbarProps> = ({ searchQuery, setSearchQuery }) => {
  return (
    <header className="w-full h-16 bg-surface flex justify-between items-center px-margin-md border-b border-outline-variant z-40 relative shrink-0">
      <div className="flex items-center gap-6 font-headline-sm text-headline-sm font-data-mono text-data-mono">
        <span className="font-headline-md text-headline-md text-primary tracking-tighter">FLOODSENSE</span>
        <nav className="hidden md:flex gap-4">
          <a className="text-on-surface-variant hover:bg-surface-variant transition-colors cursor-pointer active:opacity-80 px-2 py-1" href="#docs">Docs</a>
          <a className="text-on-surface-variant hover:bg-surface-variant transition-colors cursor-pointer active:opacity-80 px-2 py-1" href="#api">API</a>
          <a className="text-on-surface-variant hover:bg-surface-variant transition-colors cursor-pointer active:opacity-80 px-2 py-1" href="#support">Support</a>
        </nav>
      </div>
      <div className="flex items-center gap-4">
        <div className="relative flex items-center border border-outline-variant bg-surface-container px-3 py-1 w-64 h-10">
          <span className="material-symbols-outlined text-on-surface-variant text-sm mr-2">search</span>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="ID / Region"
            className="bg-transparent border-none outline-none text-on-surface font-data-mono text-data-mono w-full focus:ring-0 placeholder:text-outline p-0 text-xs"
          />
          <div className="absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-1">
            <span className="px-1 border border-outline-variant text-[10px] text-outline font-data-mono">/</span>
          </div>
        </div>
        <button className="w-10 h-10 flex items-center justify-center hover:bg-surface-variant transition-colors cursor-pointer active:opacity-80 text-on-surface-variant">
          <span className="material-symbols-outlined">notifications</span>
        </button>
        <button className="w-10 h-10 flex items-center justify-center hover:bg-surface-variant transition-colors cursor-pointer active:opacity-80 text-on-surface-variant">
          <span className="material-symbols-outlined">settings</span>
        </button>
        <div className="w-8 h-8 rounded-none border-tech overflow-hidden bg-surface-variant">
          <div className="w-full h-full flex items-center justify-center bg-surface-container-high text-primary font-data-mono text-xs font-bold">
            OPS
          </div>
        </div>
      </div>
    </header>
  );
};
