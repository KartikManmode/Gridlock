import { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { MapLibreHeatmap } from './components/MapLibreHeatmap';
import { TimeSlider } from './components/TimeSlider';
import { ZoneDrawer } from './components/ZoneDrawer';
import { IncidentModal } from './components/IncidentModal';
import { OperatorTriagePanel } from './components/OperatorTriagePanel';
import { Zone, Incident, CityKPIs, UserRole, ViolationType, SeverityLevel } from './types';
import {
  fetchDashboardData,
  fetchZones,
  fetchIncidents,
  reportIncident,
  dispatchOfficer,
  FALLBACK_KPIS,
  FALLBACK_ZONES,
  FALLBACK_INCIDENTS,
} from './services/api';

export function App() {
  const [currentRole, setCurrentRole] = useState<UserRole>('operator');
  const [slotIndex, setSlotIndex] = useState<number>(36); // Default 09:00 AM Morning Rush
  const [zones, setZones] = useState<Zone[]>(FALLBACK_ZONES);
  const [incidents, setIncidents] = useState<Incident[]>(FALLBACK_INCIDENTS);
  const [kpis, setKpis] = useState<CityKPIs>(FALLBACK_KPIS);
  const [selectedZone, setSelectedZone] = useState<Zone | null>(null);
  const [isReportModalOpen, setIsReportModalOpen] = useState<boolean>(false);
  const [preselectedZoneForReport, setPreselectedZoneForReport] = useState<Zone | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Load initial data from backend (with graceful fallback)
  const loadData = async () => {
    setIsLoading(true);
    try {
      const [dash, zList, incList] = await Promise.all([
        fetchDashboardData(),
        fetchZones(),
        fetchIncidents(),
      ]);

      if (dash && dash.kpis) setKpis(dash.kpis);
      if (zList && zList.length > 0) setZones(zList);
      if (incList && incList.length > 0) setIncidents(incList);
    } catch (err) {
      console.warn('Backend connecting or offline; using local high-fidelity Bangalore data.');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  // Handle reporting violation
  const handleReportIncident = async (payload: {
    zone_id: number;
    violation_type: ViolationType;
    severity: SeverityLevel;
    description: string;
  }) => {
    const newInc = await reportIncident(payload);
    setIncidents(prev => [newInc, ...prev]);
    setKpis(prev => ({
      ...prev,
      active_incidents_count: prev.active_incidents_count + 1,
    }));
  };

  // Handle officer dispatch
  const handleDispatchOfficer = async (incidentId: number, badgeNumber: string) => {
    await dispatchOfficer(incidentId, badgeNumber);
    setIncidents(prev =>
      prev.map(inc =>
        inc.id === incidentId
          ? { ...inc, status: 'DISPATCHED', officer_badge_number: badgeNumber }
          : inc
      )
    );
    setKpis(prev => ({
      ...prev,
      dispatched_units_count: prev.dispatched_units_count + 1,
    }));
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-slate-950 overflow-hidden font-sans">
      {/* Top Application Bar */}
      <Header
        currentRole={currentRole}
        onRoleChange={setCurrentRole}
        kpis={kpis}
        onOpenReportModal={() => {
          setPreselectedZoneForReport(selectedZone || zones[0]);
          setIsReportModalOpen(true);
        }}
        onRefresh={loadData}
        isLoading={isLoading}
      />

      {/* Main Map Workspace */}
      <main className="relative flex-1 w-full h-[calc(100vh-4rem)] overflow-hidden">
        {/* Full-bleed MapLibre GL Heatmap Canvas */}
        <MapLibreHeatmap
          zones={zones}
          incidents={incidents}
          selectedZone={selectedZone}
          onSelectZone={setSelectedZone}
          slotIndex={slotIndex}
        />

        {/* Operator Triage Board (Visible for Operator & Admin) */}
        {(currentRole === 'operator' || currentRole === 'admin') && (
          <OperatorTriagePanel
            incidents={incidents}
            onDispatchOfficer={handleDispatchOfficer}
            onSelectIncidentZone={(zoneId) => {
              const matched = zones.find(z => z.id === zoneId);
              if (matched) setSelectedZone(matched);
            }}
          />
        )}

        {/* Zone Details & Predicted vs Observed Chart Drawer */}
        <ZoneDrawer
          zone={selectedZone}
          onClose={() => setSelectedZone(null)}
          onReportIncidentForZone={(z) => {
            setPreselectedZoneForReport(z);
            setIsReportModalOpen(true);
          }}
        />

        {/* 15-Minute Playback Slider (Docked at Bottom Center) */}
        <TimeSlider slotIndex={slotIndex} onSlotChange={setSlotIndex} />

        {/* Violation Reporting Modal */}
        <IncidentModal
          isOpen={isReportModalOpen}
          onClose={() => setIsReportModalOpen(false)}
          zones={zones}
          preselectedZone={preselectedZoneForReport}
          onSubmit={handleReportIncident}
        />
      </main>
    </div>
  );
}

export default App;
