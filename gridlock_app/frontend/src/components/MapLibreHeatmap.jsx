import { useEffect, useRef, useState, useMemo } from 'react';
import maplibregl from 'maplibre-gl';
import { geohashToPolygonCoordinates, getCongestionColor } from '../utils/geohash';
import { checkWebGLSupport } from '../utils/webgl';
import { Tactical2DMap } from './Tactical2DMap';
import { Maximize2, Layers, MapPin } from 'lucide-react';

// Resilient inline style using CARTO Dark Matter raster tiles (zero external json/font fetch dependency)
const INLINE_MAP_STYLE = {
  version: 8,
  sources: {
    'carto-dark': {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
        'https://d.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
      ],
      tileSize: 256,
      attribution: '&copy; <a href="https://carto.com/">CARTO</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
    },
  },
  layers: [
    {
      id: 'background',
      type: 'background',
      paint: {
        'background-color': '#090d16',
      },
    },
    {
      id: 'carto-tiles',
      type: 'raster',
      source: 'carto-dark',
      minzoom: 0,
      maxzoom: 20,
    },
  ],
};

export const MapLibreHeatmap = ({
  zones,
  incidents,
  selectedZone,
  onSelectZone,
  slotIndex,
}) => {
  // Check WebGL availability safely
  const [webglStatus] = useState(() => checkWebGLSupport());
  const [webglFailed, setWebglFailed] = useState(!webglStatus.supported);

  const mapContainerRef = useRef(null);
  const mapRef = useRef(null);
  const markersRef = useRef([]);
  const [hoveredZone, setHoveredZone] = useState(null);
  const [tooltipPos, setTooltipPos] = useState(null);
  const [isMapReady, setIsMapReady] = useState(false);

  // Keep live references for map event listeners without stale closures
  const zonesRef = useRef(zones);
  zonesRef.current = zones;
  const slotIndexRef = useRef(slotIndex);
  slotIndexRef.current = slotIndex;

  // Calculate diurnal multiplier based on 15-minute slotIndex (0 = 00:00, 36 = 09:00, 72 = 18:00)
  const getSlotCongestion = (baseLevel, zoneId, slot) => {
    const hour = (slot * 15) / 60;
    const mPeak = Math.exp(-Math.pow(hour - 9.0, 2) / (2 * 1.8 * 1.8));
    const ePeak = Math.exp(-Math.pow(hour - 18.5, 2) / (2 * 2.2 * 2.2));
    const timeFactor = 0.2 + 0.8 * Math.max(mPeak, ePeak * 1.15);

    const baseBias = baseLevel === 'SEVERE' ? 0.2 : baseLevel === 'HIGH' ? 0.1 : 0.0;
    const zoneSeed = ((zoneId * 17) % 10) / 50;
    const finalVolume = Math.min(1.0, Math.max(0.08, timeFactor + zoneSeed + baseBias));

    let level = 'LOW';
    if (finalVolume >= 0.75) level = 'SEVERE';
    else if (finalVolume >= 0.55) level = 'HIGH';
    else if (finalVolume >= 0.35) level = 'MEDIUM';

    return { level, volume: parseFloat(finalVolume.toFixed(3)) };
  };

  // Build GeoJSON features for zones
  const buildZonesGeoJSON = () => {
    const currentZones = zonesRef.current;
    const currentSlot = slotIndexRef.current;

    return {
      type: 'FeatureCollection',
      features: currentZones.map(zone => {
        const { level, volume } = getSlotCongestion(zone.current_congestion_level, zone.id, currentSlot);
        const colors = getCongestionColor(level);
        return {
          type: 'Feature',
          id: zone.id,
          properties: {
            id: zone.id,
            geohash: zone.geohash,
            junction_name: zone.junction_name,
            congestion_level: level,
            volume,
            capacity: zone.capacity,
            fill_color: colors.fillHex,
            border_color: colors.hex,
            zone_tier: zone.zone_tier || 'Core City',
          },
          geometry: {
            type: 'Polygon',
            coordinates: geohashToPolygonCoordinates(zone.geohash),
          },
        };
      }),
    };
  };

  // Compute bounding box of current zones for auto-fit
  const zoneBounds = useMemo(() => {
    if (!zones || zones.length === 0) return null;
    let minLon = 180, maxLon = -180, minLat = 90, maxLat = -90;
    zones.forEach(z => {
      if (z.longitude < minLon) minLon = z.longitude;
      if (z.longitude > maxLon) maxLon = z.longitude;
      if (z.latitude < minLat) minLat = z.latitude;
      if (z.latitude > maxLat) maxLat = z.latitude;
    });
    return [
      [minLon - 0.015, minLat - 0.015],
      [maxLon + 0.015, maxLat + 0.015],
    ];
  }, [zones]);

  // Fit bounds helper
  const fitToAllZones = () => {
    const map = mapRef.current;
    if (!map || !zoneBounds) return;
    map.fitBounds(zoneBounds, {
      padding: { top: 70, bottom: 90, left: 320, right: 380 },
      maxZoom: 14,
      duration: 1000,
    });
  };

  // Initialize MapLibre (guarded with WebGL fallback)
  useEffect(() => {
    if (webglFailed || !webglStatus.supported) return;
    if (!mapContainerRef.current || mapRef.current) return;

    // Center around the dataset's geohash coordinates by default
    const initialCenter = [90.7745, -5.4547];

    try {
      const map = new maplibregl.Map({
        container: mapContainerRef.current,
        style: INLINE_MAP_STYLE,
        center: initialCenter,
        zoom: 11.5,
        minZoom: 3,
        maxZoom: 18,
        attributionControl: false,
      });

      map.addControl(new maplibregl.NavigationControl({ showCompass: true, showZoom: true }), 'top-right');
      mapRef.current = map;

      map.on('error', (e) => {
        if (e?.error?.message?.includes('WebGL') || e?.statusMessage?.includes('WebGL')) {
          console.warn('MapLibre WebGL runtime error caught, falling back to Tactical 2D Canvas:', e);
          setWebglFailed(true);
        }
      });

      map.on('load', () => {
        setIsMapReady(true);

        // Add Zones Polygon Source
        map.addSource('zones-source', {
          type: 'geojson',
          data: buildZonesGeoJSON(),
        });

        // Layer 1: Polygon Fill with dynamic opacity
        map.addLayer({
          id: 'zones-fill',
          type: 'fill',
          source: 'zones-source',
          paint: {
            'fill-color': ['get', 'fill_color'],
            'fill-opacity': [
              'case',
              ['boolean', ['feature-state', 'hover'], false],
              0.88,
              0.62,
            ],
          },
        });

        // Layer 2: Glowing Polygon Outline
        map.addLayer({
          id: 'zones-outline',
          type: 'line',
          source: 'zones-source',
          paint: {
            'line-color': ['get', 'border_color'],
            'line-width': [
              'case',
              ['boolean', ['feature-state', 'selected'], false],
              4.0,
              2.2,
            ],
            'line-opacity': 0.95,
          },
        });

        // Mouse Hover & Click Handlers
        map.on('mousemove', 'zones-fill', (e) => {
          if (e.features && e.features.length > 0) {
            const feature = e.features[0];
            const zoneId = feature.properties?.id;
            const matched = zonesRef.current.find(z => z.id === zoneId);
            if (matched) {
              setHoveredZone({
                ...matched,
                current_congestion_level: feature.properties.congestion_level,
              });
              setTooltipPos({ x: e.point.x, y: e.point.y });
            }
            map.getCanvas().style.cursor = 'pointer';
          }
        });

        map.on('mouseleave', 'zones-fill', () => {
          setHoveredZone(null);
          setTooltipPos(null);
          map.getCanvas().style.cursor = '';
        });

        map.on('click', 'zones-fill', (e) => {
          if (e.features && e.features.length > 0) {
            const zoneId = e.features[0].properties?.id;
            const matched = zonesRef.current.find(z => z.id === zoneId);
            if (matched) {
              onSelectZone(matched);
            }
          }
        });

        // Fit map to zones on first load
        if (zoneBounds) {
          map.fitBounds(zoneBounds, {
            padding: { top: 70, bottom: 90, left: 320, right: 380 },
            maxZoom: 14,
            duration: 800,
          });
        }
      });
    } catch (err) {
      console.warn('MapLibre initialization threw WebGL error. Switching to Tactical 2D Canvas fallback:', err);
      setWebglFailed(true);
    }

    return () => {
      markersRef.current.forEach(m => m.remove());
      markersRef.current = [];
      if (mapRef.current) {
        try {
          mapRef.current.remove();
        } catch (_) {}
        mapRef.current = null;
      }
    };
  }, [webglFailed, webglStatus.supported]);

  // If WebGL is not supported or failed to initialize, render the robust 2D Canvas/SVG Map
  if (webglFailed || !webglStatus.supported) {
    return (
      <Tactical2DMap
        zones={zones}
        incidents={incidents}
        selectedZone={selectedZone}
        onSelectZone={onSelectZone}
        slotIndex={slotIndex}
        webglDisabledReason={webglStatus.reason}
      />
    );
  }

  // Update GeoJSON source whenever zones or slotIndex changes
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !map.isStyleLoaded()) return;

    const source = map.getSource('zones-source');
    if (source) {
      source.setData(buildZonesGeoJSON());
    }
  }, [slotIndex, zones]);

  // Sync HTML markers on zone centroids
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isMapReady) return;

    markersRef.current.forEach(m => m.remove());
    markersRef.current = [];

    zones.forEach(zone => {
      const { level, volume } = getSlotCongestion(zone.current_congestion_level, zone.id, slotIndex);
      const colors = getCongestionColor(level);

      const el = document.createElement('div');
      el.className = 'group cursor-pointer select-none';
      const isSelected = selectedZone?.id === zone.id;

      el.innerHTML = `
        <div class="px-2 py-0.5 rounded-md backdrop-blur-md shadow-lg transition-transform transform hover:scale-110 flex items-center space-x-1.5 border ${
          isSelected
            ? 'bg-sky-500/90 border-white text-white font-bold ring-2 ring-sky-400'
            : 'bg-slate-900/85 border-slate-700/80 text-slate-200 hover:border-slate-400'
        }">
          <span class="w-2 h-2 rounded-full inline-block" style="background-color: ${colors.hex}; box-shadow: 0 0 6px ${colors.hex}"></span>
          <span class="text-[11px] font-medium tracking-tight whitespace-nowrap">${zone.junction_name.split(' - ')[0]}</span>
          <span class="text-[10px] font-mono text-slate-300 font-semibold">${Math.round(volume * 100)}%</span>
        </div>
      `;

      el.addEventListener('click', (e) => {
        e.stopPropagation();
        onSelectZone(zone);
      });

      const marker = new maplibregl.Marker({ element: el, anchor: 'center' })
        .setLngLat([zone.longitude, zone.latitude])
        .addTo(map);

      markersRef.current.push(marker);
    });
  }, [zones, slotIndex, selectedZone, isMapReady]);

  // Fit bounds when zones change if bounds are defined
  useEffect(() => {
    if (isMapReady && zoneBounds && mapRef.current) {
      fitToAllZones();
    }
  }, [zoneBounds, isMapReady]);

  // Center on selected zone if changed
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !selectedZone) return;

    map.flyTo({
      center: [selectedZone.longitude, selectedZone.latitude],
      zoom: 13.5,
      essential: true,
      duration: 900,
    });
  }, [selectedZone]);

  return (
    <div className="relative w-full h-full overflow-hidden bg-slate-950">
      {/* MapLibre Canvas Container */}
      <div ref={mapContainerRef} className="w-full h-full" />

      {/* Loading state indicator */}
      {!isMapReady && (
        <div className="absolute inset-0 flex items-center justify-center bg-slate-950/80 backdrop-blur-sm z-30">
          <div className="flex flex-col items-center space-y-3">
            <div className="w-10 h-10 border-3 border-sky-500 border-t-transparent rounded-full animate-spin" />
            <span className="text-sm font-semibold text-slate-300">Initializing Gridlock Map Engine...</span>
          </div>
        </div>
      )}

      {/* Recenter / Fit All Zones Button */}
      <div className="absolute top-4 left-4 z-10 flex items-center space-x-2">
        <button
          onClick={fitToAllZones}
          title="Fit All Zones"
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 text-slate-200 text-xs font-medium border border-slate-700/80 shadow-lg backdrop-blur cursor-pointer transition-all"
        >
          <Maximize2 className="w-3.5 h-3.5 text-sky-400" />
          <span>Fit City View ({zones.length} Zones)</span>
        </button>

        <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900/90 text-slate-400 text-xs border border-slate-800 backdrop-blur">
          <MapPin className="w-3.5 h-3.5 text-emerald-400" />
          <span className="font-mono text-[11px]">SOTA v47 Geo-Grid</span>
        </div>
      </div>

      {/* Hover Tooltip */}
      {hoveredZone && tooltipPos && (
        <div
          className="absolute z-30 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-3 px-3 py-2 rounded-lg bg-slate-900/95 border border-slate-700 shadow-2xl backdrop-blur text-xs"
          style={{ left: `${tooltipPos.x}px`, top: `${tooltipPos.y}px` }}
        >
          <div className="flex items-center space-x-2">
            <span className="font-bold text-white">{hoveredZone.junction_name}</span>
            <span className="text-[10px] font-mono px-1 py-0.2 rounded bg-slate-800 text-sky-400">
              {hoveredZone.geohash}
            </span>
          </div>
          <div className="mt-1 flex items-center justify-between space-x-3 text-slate-300">
            <span>Status:</span>
            <span className={`font-semibold ${
              hoveredZone.current_congestion_level === 'SEVERE' ? 'text-red-400' :
              hoveredZone.current_congestion_level === 'HIGH' ? 'text-orange-400' :
              hoveredZone.current_congestion_level === 'MEDIUM' ? 'text-yellow-400' : 'text-emerald-400'
            }`}>
              {hoveredZone.current_congestion_level}
            </span>
          </div>
          <div className="flex items-center justify-between space-x-3 text-slate-400 text-[11px]">
            <span>Capacity:</span>
            <span className="font-mono text-slate-200">{hoveredZone.capacity} veh/hr</span>
          </div>
        </div>
      )}

      {/* Floating Legend */}
      <div className="absolute bottom-28 left-6 z-10 px-3.5 py-3 rounded-xl bg-slate-900/90 border border-slate-800 backdrop-blur shadow-xl text-xs space-y-1.5">
        <div className="flex items-center space-x-1.5 font-semibold text-slate-300 text-[11px] mb-1">
          <Layers className="w-3.5 h-3.5 text-sky-400" />
          <span>Congestion Index</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-red-600/80 border border-red-500 shadow-sm shadow-red-500/40" />
          <span className="text-slate-300">Severe (&gt; 75%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-orange-500/80 border border-orange-400 shadow-sm shadow-orange-500/40" />
          <span className="text-slate-300">High (55% - 75%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-yellow-500/80 border border-yellow-400 shadow-sm shadow-yellow-500/40" />
          <span className="text-slate-300">Medium (35% - 55%)</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-3 h-3 rounded-sm bg-emerald-500/80 border border-emerald-400 shadow-sm shadow-emerald-500/40" />
          <span className="text-slate-300">Low (&lt; 35%)</span>
        </div>
      </div>
    </div>
  );
};
