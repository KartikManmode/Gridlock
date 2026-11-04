import React, { useState } from 'react';
import { ShieldAlert, Radio, CheckCircle, ChevronLeft, ChevronRight, UserCheck } from 'lucide-react';
import { Incident } from '../types';

interface OperatorTriagePanelProps {
  incidents: Incident[];
  onDispatchOfficer: (incidentId: number, badgeNumber: string) => Promise<void>;
  onSelectIncidentZone: (zoneId: number) => void;
}

export const OperatorTriagePanel: React.FC<OperatorTriagePanelProps> = ({
  incidents,
  onDispatchOfficer,
  onSelectIncidentZone,
}) => {
  const [isOpen, setIsOpen] = useState(true);
  const [filter, setFilter] = useState<'ALL' | 'ACTIVE' | 'DISPATCHED'>('ALL');
  const [activeDispatchId, setActiveDispatchId] = useState<number | null>(null);
  const [badgeInput, setBadgeInput] = useState('KA-TP-1042');

  const filteredIncidents = incidents.filter((inc) => {
    if (filter === 'ACTIVE') return inc.status === 'REPORTED';
    if (filter === 'DISPATCHED') return inc.status === 'DISPATCHED';
    return true;
  });

  const handleDispatch = async (incidentId: number) => {
    if (!badgeInput.trim()) return;
    await onDispatchOfficer(incidentId, badgeInput.trim());
    setActiveDispatchId(null);
  };

  return (
    <div
      className={`absolute top-20 left-6 z-20 transition-all duration-300 flex ${
        isOpen ? 'w-84' : 'w-10'
      }`}
    >
      {/* Triage Container */}
      {isOpen && (
        <div className="w-80 rounded-2xl bg-slate-900/95 border border-slate-800 shadow-2xl backdrop-blur-xl flex flex-col max-h-[calc(100vh-14rem)] overflow-hidden">
          {/* Panel Header */}
          <div className="p-3.5 border-b border-slate-800 flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <ShieldAlert className="w-4 h-4 text-rose-400" />
              <h3 className="font-bold text-xs text-white uppercase tracking-wider">Operator Triage Board</h3>
            </div>
            <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-rose-500/20 text-rose-300 border border-rose-500/30">
              {incidents.filter(i => i.status !== 'RESOLVED').length} Active
            </span>
          </div>

          {/* Filter Bar */}
          <div className="flex p-1.5 bg-slate-950/60 border-b border-slate-800 text-[10px] font-medium">
            {(['ALL', 'ACTIVE', 'DISPATCHED'] as const).map((tab) => (
              <button
                key={tab}
                onClick={() => setFilter(tab)}
                className={`flex-1 py-1 rounded-md text-center transition-all ${
                  filter === tab
                    ? 'bg-slate-800 text-sky-400 font-semibold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Incidents Scroll List */}
          <div className="p-3 space-y-2.5 overflow-y-auto flex-1">
            {filteredIncidents.length === 0 ? (
              <div className="text-center py-8 text-xs text-slate-500">
                No active violations in this queue.
              </div>
            ) : (
              filteredIncidents.map((inc) => (
                <div
                  key={inc.id}
                  className="p-3 rounded-xl bg-slate-950/70 border border-slate-800/80 hover:border-slate-700 transition-all text-xs"
                >
                  <div className="flex items-start justify-between">
                    <button
                      onClick={() => onSelectIncidentZone(inc.zone_id)}
                      className="font-bold text-white hover:text-sky-400 text-left cursor-pointer transition-colors"
                    >
                      {inc.junction_name}
                    </button>
                    <span
                      className={`text-[9px] font-bold px-1.5 py-0.5 rounded border uppercase ${
                        inc.severity === 'HIGH'
                          ? 'bg-rose-500/20 text-rose-300 border-rose-500/40'
                          : inc.severity === 'MEDIUM'
                          ? 'bg-amber-500/20 text-amber-300 border-amber-500/40'
                          : 'bg-emerald-500/20 text-emerald-300 border-emerald-500/40'
                      }`}
                    >
                      {inc.severity}
                    </span>
                  </div>

                  <div className="flex items-center space-x-2 my-1.5 text-[11px] text-slate-400">
                    <span className="font-semibold text-slate-300">{inc.violation_type}</span>
                    <span>•</span>
                    <span>{inc.occurred_at}</span>
                  </div>

                  {inc.description && (
                    <p className="text-[11px] text-slate-400 italic line-clamp-2 mb-2">
                      "{inc.description}"
                    </p>
                  )}

                  {/* Status & Actions */}
                  {inc.status === 'REPORTED' && (
                    <div>
                      {activeDispatchId === inc.id ? (
                        <div className="flex items-center space-x-1.5 mt-2">
                          <input
                            type="text"
                            value={badgeInput}
                            onChange={(e) => setBadgeInput(e.target.value)}
                            placeholder="Officer Badge #"
                            className="bg-slate-900 border border-slate-700 rounded-lg px-2 py-1 text-xs text-white w-28 focus:outline-none focus:border-sky-500"
                          />
                          <button
                            onClick={() => handleDispatch(inc.id)}
                            className="px-2.5 py-1 bg-sky-600 hover:bg-sky-500 text-white rounded-lg text-xs font-semibold"
                          >
                            Send
                          </button>
                          <button
                            onClick={() => setActiveDispatchId(null)}
                            className="px-2 py-1 bg-slate-800 text-slate-400 rounded-lg text-xs"
                          >
                            Cancel
                          </button>
                        </div>
                      ) : (
                        <button
                          onClick={() => setActiveDispatchId(inc.id)}
                          className="w-full mt-1.5 py-1.5 rounded-lg bg-sky-600/20 hover:bg-sky-600/30 text-sky-400 border border-sky-500/40 font-semibold text-[11px] flex items-center justify-center space-x-1.5 transition-all cursor-pointer"
                        >
                          <Radio className="w-3.5 h-3.5" />
                          <span>Dispatch Officer</span>
                        </button>
                      )}
                    </div>
                  )}

                  {inc.status === 'DISPATCHED' && (
                    <div className="mt-2 flex items-center justify-between text-[11px] bg-slate-900 px-2 py-1.5 rounded-lg border border-sky-500/30">
                      <div className="flex items-center space-x-1.5 text-sky-400 font-mono">
                        <UserCheck className="w-3.5 h-3.5" />
                        <span>{inc.officer_badge_number || 'KA-TP-1042'}</span>
                      </div>
                      <span className="text-[10px] text-amber-400 font-medium">En Route</span>
                    </div>
                  )}

                  {inc.status === 'RESOLVED' && (
                    <div className="mt-2 flex items-center space-x-1 text-emerald-400 text-[11px] font-medium">
                      <CheckCircle className="w-3.5 h-3.5" />
                      <span>Bottleneck Cleared</span>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {/* Collapse/Expand Handle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="h-10 w-6 self-start mt-3 rounded-r-lg bg-slate-900 border-y border-r border-slate-800 text-slate-400 hover:text-white flex items-center justify-center shadow-lg transition-colors cursor-pointer"
      >
        {isOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
      </button>
    </div>
  );
};
