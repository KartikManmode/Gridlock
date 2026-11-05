import { KPICalculator } from './kpiCalculator.js';
import { HeatmapGenerator } from './heatmapGenerator.js';
import { HotspotAnalyzer } from './hotspotAnalyzer.js';

export class AnalyticsService {
  static async getDashboardSummary() {
    const [kpis, heatmap, active_hotspots] = await Promise.all([
      KPICalculator.calculateSummaryKPIs(),
      HeatmapGenerator.generateHeatmapPoints(),
      HotspotAnalyzer.getActiveHotspots(),
    ]);

    return {
      kpis,
      heatmap,
      active_hotspots,
    };
  }
}
