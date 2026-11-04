import React, { useState } from 'react';
import { X, AlertTriangle, Send } from 'lucide-react';
import { Zone, ViolationType, SeverityLevel } from '../types';

interface IncidentModalProps {
  isOpen: boolean;
  onClose: () => void;
  zones: Zone[];
  preselectedZone: Zone | null;
  onSubmit: (data: { zone_id: number; violation_type: ViolationType; severity: SeverityLevel; description: string }) => Promise<void>;
}

export const IncidentModal: React.FC<IncidentModalProps> = ({
  isOpen,
  onClose,
  zones,
  preselectedZone,
  onSubmit,
}) => {
  const [selectedZoneId, setSelectedZoneId] = useState<number>(preselectedZone?.id || zones[0]?.id || 1);
  const [violationType, setViolationType] = useState<ViolationType>('PARKING');
  const [severity, setSeverity] = useState<SeverityLevel>('HIGH');
  const [description, setDescription] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    try {
      await onSubmit({
        zone_id: selectedZoneId,
        violation_type: violationType,
        severity,
        description: description || 'Citizen reported traffic obstruction causing chokepoint',
      });
      setDescription('');
      onClose();
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-2xl bg-slate-900 border border-slate-700 shadow-2xl p-6 text-slate-100">
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <div className="p-2 rounded-lg bg-rose-500/10 border border-rose-500/30 text-rose-400">
              <AlertTriangle className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-base text-white">Report Traffic Violation</h3>
              <p className="text-xs text-slate-400">Flag an active bottleneck to control room</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="mt-4 space-y-4 text-xs">
          {/* Target Zone */}
          <div>
            <label className="block font-semibold text-slate-300 mb-1.5">Target Junction / Zone</label>
            <select
              value={selectedZoneId}
              onChange={(e) => setSelectedZoneId(Number(e.target.value))}
              className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2.5 text-slate-200 focus:outline-none focus:border-sky-500"
            >
              {zones.map((z) => (
                <option key={z.id} value={z.id}>
                  {z.junction_name} ({z.geohash})
                </option>
              ))}
            </select>
          </div>

          {/* Violation Type */}
          <div>
            <label className="block font-semibold text-slate-300 mb-1.5">Violation Category</label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { type: 'PARKING', label: 'Illegal Parking' },
                { type: 'SIGNAL', label: 'Signal Jump' },
                { type: 'SPEEDING', label: 'Speeding' },
              ].map(({ type, label }) => (
                <button
                  type="button"
                  key={type}
                  onClick={() => setViolationType(type as ViolationType)}
                  className={`py-2 px-2.5 rounded-xl text-center border font-medium transition-all ${
                    violationType === type
                      ? 'bg-rose-500/20 border-rose-500 text-rose-300 shadow-sm'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Severity */}
          <div>
            <label className="block font-semibold text-slate-300 mb-1.5">Severity Level</label>
            <div className="grid grid-cols-3 gap-2">
              {(['LOW', 'MEDIUM', 'HIGH'] as SeverityLevel[]).map((lvl) => (
                <button
                  type="button"
                  key={lvl}
                  onClick={() => setSeverity(lvl)}
                  className={`py-2 rounded-xl text-center border font-medium transition-all ${
                    severity === lvl
                      ? lvl === 'HIGH'
                        ? 'bg-red-500/20 border-red-500 text-red-300'
                        : lvl === 'MEDIUM'
                        ? 'bg-amber-500/20 border-amber-500 text-amber-300'
                        : 'bg-emerald-500/20 border-emerald-500 text-emerald-300'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {lvl}
                </button>
              ))}
            </div>
          </div>

          {/* Description */}
          <div>
            <label className="block font-semibold text-slate-300 mb-1.5">Observation Details</label>
            <textarea
              rows={3}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g. Commercial delivery truck double-parked on main artery near hospital entrance"
              className="w-full bg-slate-950 border border-slate-700 rounded-xl p-3 text-slate-200 placeholder-slate-500 focus:outline-none focus:border-sky-500 resize-none"
            />
          </div>

          {/* Submit */}
          <div className="pt-2">
            <button
              type="submit"
              disabled={isSubmitting}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-rose-600 to-red-600 hover:from-rose-500 hover:to-red-500 text-white font-semibold flex items-center justify-center space-x-2 shadow-lg shadow-rose-600/30 transition-all cursor-pointer disabled:opacity-50"
            >
              <Send className="w-4 h-4" />
              <span>{isSubmitting ? 'Registering with Control Room...' : 'Dispatch Alert to Police'}</span>
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
