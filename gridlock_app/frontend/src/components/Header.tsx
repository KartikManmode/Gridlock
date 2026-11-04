import React from 'react';
import { Radio, Activity, AlertTriangle, Users, PlusCircle, RefreshCw } from 'lucide-react';
import { UserRole, CityKPIs } from '../types';

interface HeaderProps {
  currentRole: UserRole;
  onRoleChange: (role: UserRole) => void;
  kpis: CityKPIs;
  onOpenReportModal: () => void;
  onRefresh: () => void;
  isLoading: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentRole,
  onRoleChange,
  kpis,
  onOpenReportModal,
  onRefresh,
  isLoading
}) => {
  return (
    <header className="h-16 border-b border-slate-800 bg-slate-900/90 backdrop-blur px-5 flex items-center justify-between z-20 shrink-0">
      {/* Brand & Mission */}
      <div className="flex items-center space-x-3">
        <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-600 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
          <Activity className="w-5 h-5 text-white animate-pulse" />
        </div>
        <div>
          <div className="flex items-center space-x-2">
            <h1 className="text-lg font-bold tracking-tight text-white">Gridlock Heatmap</h1>
            <span className="text-[10px] px-1.5 py-0.5 rounded font-mono font-semibold bg-sky-500/10 text-sky-400 border border-sky-500/30">
              v0.2 · SOTA 93.087%
            </span>
          </div>
          <p className="text-xs text-slate-400 hidden sm:block">Bangalore Real-Time Traffic Operations Platform</p>
        </div>
      </div>

      {/* KPI Ticker */}
      <div className="hidden lg:flex items-center space-x-6 px-4 py-1.5 rounded-lg bg-slate-950/60 border border-slate-800/80 text-xs">
        <div className="flex items-center space-x-2">
          <span className="text-slate-400">Avg Speed:</span>
          <span className="font-mono font-semibold text-emerald-400">{kpis.average_city_speed_kmh} km/h</span>
        </div>
        <div className="w-px h-3 bg-slate-800" />
        <div className="flex items-center space-x-2">
          <span className="text-slate-400">Congestion:</span>
          <span className="font-mono font-semibold text-amber-400">{kpis.city_wide_congestion_index}%</span>
        </div>
        <div className="w-px h-3 bg-slate-800" />
        <div className="flex items-center space-x-2">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-400" />
          <span className="text-slate-400">Incidents:</span>
          <span className="font-mono font-semibold text-rose-400">{kpis.active_incidents_count} active</span>
        </div>
        <div className="w-px h-3 bg-slate-800" />
        <div className="flex items-center space-x-2">
          <Radio className="w-3.5 h-3.5 text-sky-400" />
          <span className="text-slate-400">Dispatched:</span>
          <span className="font-mono font-semibold text-sky-400">{kpis.dispatched_units_count} units</span>
        </div>
      </div>

      {/* Right Controls: Role Switcher & Action */}
      <div className="flex items-center space-x-3">
        {/* Role Toggle */}
        <div className="flex items-center bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs">
          <Users className="w-3.5 h-3.5 text-slate-400 ml-2 mr-1.5" />
          {(['citizen', 'operator', 'admin'] as UserRole[]).map(role => (
            <button
              key={role}
              onClick={() => onRoleChange(role)}
              className={`px-2.5 py-1 rounded-md font-medium capitalize transition-all ${
                currentRole === role
                  ? 'bg-sky-600 text-white shadow-sm'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {role}
            </button>
          ))}
        </div>

        {/* Refresh */}
        <button
          onClick={onRefresh}
          disabled={isLoading}
          title="Refresh Data"
          className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors border border-slate-700"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-sky-400' : ''}`} />
        </button>

        {/* Report Violation Button */}
        <button
          onClick={onOpenReportModal}
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-rose-600 hover:bg-rose-500 text-white text-xs font-semibold shadow-md shadow-rose-600/20 transition-all cursor-pointer"
        >
          <PlusCircle className="w-4 h-4" />
          <span>Report Chokepoint</span>
        </button>
      </div>
    </header>
  );
};
