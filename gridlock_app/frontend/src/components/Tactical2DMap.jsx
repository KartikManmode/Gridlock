import { useState, useRef, useMemo, useEffect } from 'react';
import { geohashToPolygonCoordinates, getCongestionColor } from '../utils/geohash';
import { ZoomIn, ZoomOut, Maximize2, Layers } from 'lucide-react';

export const Tactical2DMap = ({
  zones,
  incidents,
  selectedZone,
  onSelectZone,
  slotIndex,
  webglDisabledReason: _webglDisabledReason,
}) => {
  const containerRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 1000, height: 700 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [hoveredZone, setHoveredZone] = useState(null);
  const [tooltipPos, setTooltipPos] = useState(null);
  const [showWebglNotice, setShowWebglNotice] = useState(true);

  // Resize listener to adapt SVG viewport
  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth || 1000,
          height: containerRef.current.clientHeight || 700,
        });
      }
    };
    updateSize();
    window.addEventListener('resize', updateSize);
    return () => window.removeEventListener('resize', updateSize);
  }, []);

  // Compute bounding box of all zones
  const bbox = useMemo(() => {
    if (!zones || zones.length === 0) {
      return { minLon: 90.64, maxLon: 90.89, minLat: -5.49, maxLat: -5.42 };
    }
    let minLon = 180, maxLon = -180, minLat = 90, maxLat = -90;
    zones.forEach((z) => {
      if (z.longitude < minLon) minLon = z.longitude;
      if (z.longitude > maxLon) maxLon = z.longitude;
      if (z.latitude < minLat) minLat = z.latitude;
      if (z.latitude > maxLat) maxLat = z.latitude;
    });
    // Add 10% breathing margin
    const lonPadding = (maxLon - minLon) * 0.1 || 0.02;
    const latPadding = (maxLat - minLat) * 0.1 || 0.02;
    return {
      minLon: minLon - lonPadding,
      maxLon: maxLon + lonPadding,
      minLat: minLat - latPadding,
      maxLat: maxLat + latPadding,
    };
  }, [zones]);

  // Project longitude & latitude into SVG pixel coordinates
  const project = (lon, lat) => {
    const pad = 60;
    const availW = dimensions.width - 2 * pad;
    const availH = dimensions.height - 2 * pad;

    const normX = (lon - bbox.minLon) / (bbox.maxLon - bbox.minLon);
    // Invert Y axis: latitude increases going north, SVG Y increases going south
    const normY = (bbox.maxLat - lat) / (bbox.maxLat - bbox.minLat);

    return {
      x: pad + normX * availW,
      y: pad + normY * availH,
    };
  };

  // Diurnal multiplier based on 15-minute slotIndex
  const getSlotCongestion = (baseLevel, zoneId) => {
    const hour = (slotIndex * 15) / 60;
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

  // Pan and drag handlers
  const handleMouseDown = (e) => {
    if (e.button !== 0) return;
    setIsDragging(true);
    setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.88;
    setZoom((prev) => Math.max(0.6, Math.min(4.5, prev * factor)));
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Zone incident count map
  const zoneIncidentsMap = useMemo(() => {
    const map = new Map();
    incidents.forEach((inc) => {
      const existing = map.get(inc.zone_id) || [];
      existing.push(inc);
      map.set(inc.zone_id, existing);
    });
    return map;
  }, [incidents]);

  return (
    <div
      ref={containerRef}
      className="relative w-full h-full overflow-hidden bg-slate-950 select-none cursor-grab active:cursor-grabbing"
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onWheel={handleWheel}
    >
      {/* WebGL Disabled Informational Banner */}
      {showWebglNotice && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-30 max-w-xl w-[92%] px-3.5 py-2 rounded-xl bg-slate-900/95 border border-sky-500/40 shadow-2xl backdrop-blur-md flex items-center justify-between text-xs text-slate-200">
          <div className="flex items-center space-x-2.5">
            <div className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
            <div>
              <span className="font-semibold text-sky-400">Tactical 2D Engine Active</span>
              <span className="text-slate-400 ml-1.5 hidden sm:inline">
                (Browser WebGL disabled — Running full-fidelity vector canvas)
              </span>
            </div>
          </div>
          <button
            onClick={() => setShowWebglNotice(false)}
            className="text-[11px] text-slate-400 hover:text-white px-2 py-0.5 rounded hover:bg-slate-800 transition-colors"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* SVG Canvas Map */}
      <svg
        width="100%"
        height="100%"
        className="w-full h-full"
        viewBox={`0 0 ${dimensions.width} ${dimensions.height}`}
      >
        <defs>
          {/* Background Grid Pattern */}
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#1e293b" strokeWidth="0.8" strokeOpacity="0.4" />
          </pattern>
          <pattern id="grid-sub" width="160" height="160" patternUnits="userSpaceOnUse">
            <rect width="160" height="160" fill="url(#grid)" />
            <path d="M 160 0 L 0 0 0 160" fill="none" stroke="#334155" strokeWidth="1.2" strokeOpacity="0.6" />
          </pattern>
          {/* Glow Filters */}
          <filter id="glow-selected" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* Global Dark Background */}
        <rect width="100%" height="100%" fill="#070a13" />

        {/* Scalable & Pannable Group */}
        <g
          transform={`translate(${pan.x + (dimensions.width / 2) * (1 - zoom)}, ${
            pan.y + (dimensions.height / 2) * (1 - zoom)
          }) scale(${zoom})`}
        >
          {/* Tactical Coordinate Grid */}
          <rect
            x={-dimensions.width * 2}
            y={-dimensions.height * 2}
            width={dimensions.width * 5}
            height={dimensions.height * 5}
            fill="url(#grid-sub)"
          />

          {/* Radar Circles */}
          <circle
            cx={dimensions.width / 2}
            cy={dimensions.height / 2}
            r={240}
            fill="none"
            stroke="#1e293b"
            strokeWidth="1"
            strokeDasharray="4 6"
            strokeOpacity="0.4"
          />
          <circle
            cx={dimensions.width / 2}
            cy={dimensions.height / 2}
            r={380}
            fill="none"
            stroke="#1e293b"
            strokeWidth="1"
            strokeDasharray="4 6"
            strokeOpacity="0.25"
          />

          {/* Zones Polygons Layer */}
          {zones.map((zone) => {
            const rawCoords = geohashToPolygonCoordinates(zone.geohash)[0];
            const projectedPoints = rawCoords.map(([lon, lat]) => {
              const pt = project(lon, lat);
              return `${pt.x},${pt.y}`;
            }).join(' ');

            const { level, volume } = getSlotCongestion(zone.current_congestion_level, zone.id);
            const colors = getCongestionColor(level);
            const isSelected = selectedZone?.id === zone.id;
            const isHovered = hoveredZone?.id === zone.id;
            const zoneIncidents = zoneIncidentsMap.get(zone.id) || [];
            const centerPt = project(zone.longitude, zone.latitude);

            return (
              <g
                key={zone.id}
                className="cursor-pointer transition-all duration-150"
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectZone(zone);
                }}
                onMouseEnter={(e) => {
                  setHoveredZone({ ...zone, current_congestion_level: level });
                  setTooltipPos({ x: e.clientX, y: e.clientY });
                }}
                onMouseMove={(e) => {
                  setTooltipPos({ x: e.clientX, y: e.clientY });
                }}
                onMouseLeave={() => {
                  setHoveredZone(null);
                  setTooltipPos(null);
                }}
              >
                {/* Polygon Bounding Cell */}
                <polygon
                  points={projectedPoints}
                  fill={colors.fillHex}
                  stroke={isSelected ? '#38bdf8' : isHovered ? '#ffffff' : colors.hex}
                  strokeWidth={isSelected ? 3.5 : isHovered ? 2.5 : 1.6}
                  strokeOpacity={isSelected ? 1.0 : isHovered ? 0.95 : 0.8}
                  filter={isSelected ? 'url(#glow-selected)' : undefined}
                  className="transition-colors duration-150 hover:brightness-125"
                />

                {/* Centroid Badge & Junction Label */}
                <g transform={`translate(${centerPt.x}, ${centerPt.y})`}>
                  {/* Badge Pill Background */}
                  <rect
                    x={-45}
                    y={-11}
                    width={90}
                    height={22}
                    rx={6}
                    fill={isSelected ? '#0284c7' : '#0f172a'}
                    stroke={isSelected ? '#ffffff' : '#334155'}
                    strokeWidth={isSelected ? 1.5 : 0.8}
                    fillOpacity={0.92}
                  />

                  {/* Congestion Status Dot */}
                  <circle
                    cx={-34}
                    cy={0}
                    r={3.5}
                    fill={colors.hex}
                  />

                  {/* Percentage */}
                  <text
                    x={-24}
                    y={3.5}
                    fill="#f8fafc"
                    fontSize={10}
                    fontWeight="bold"
                    fontFamily="monospace"
                  >
                    {Math.round(volume * 100)}%
                  </text>

                  {/* Junction Short Code */}
                  <text
                    x={12}
                    y={3.5}
                    fill={isSelected ? '#ffffff' : '#94a3b8'}
                    fontSize={9}
                    fontWeight="500"
                    textAnchor="middle"
                  >
                    {zone.geohash}
                  </text>

                  {/* Incident Alert Icon */}
                  {zoneIncidents.length > 0 && (
                    <g transform="translate(32, -8)">
                      <circle cx={4} cy={4} r={6} fill="#ef4444" />
                      <text x={4} y={7} fill="#ffffff" fontSize={8} fontWeight="bold" textAnchor="middle">
                        !
                      </text>
                    </g>
                  )}
                </g>
              </g>
            );
          })}
        </g>
      </svg>

      {/* Floating View Controls (Top Left) */}
      <div className="absolute top-4 left-4 z-20 flex items-center space-x-2">
        <button
          onClick={resetView}
          title="Reset City View"
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-slate-900/90 hover:bg-slate-800 text-slate-200 text-xs font-medium border border-slate-700/80 shadow-lg backdrop-blur cursor-pointer transition-all"
        >
          <Maximize2 className="w-3.5 h-3.5 text-sky-400" />
          <span>Fit City View ({zones.length} Zones)</span>
        </button>

        <div className="flex items-center space-x-1 bg-slate-900/90 border border-slate-700/80 rounded-lg p-0.5 backdrop-blur shadow-lg">
          <button
            onClick={() => setZoom((z) => Math.min(4.5, z * 1.25))}
            title="Zoom In"
            className="p-1.5 rounded-md hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <div className="w-px h-3 bg-slate-800" />
          <button
            onClick={() => setZoom((z) => Math.max(0.6, z / 1.25))}
            title="Zoom Out"
            className="p-1.5 rounded-md hover:bg-slate-800 text-slate-300 hover:text-white transition-colors"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
        </div>

        <div className="hidden md:flex items-center space-x-1.5 px-2.5 py-1.5 rounded-lg bg-slate-900/90 text-slate-400 text-xs border border-slate-800 backdrop-blur">
          <span className="font-mono text-[11px] text-sky-400">{Math.round(zoom * 100)}%</span>
        </div>
      </div>

      {/* Hover Tooltip */}
      {hoveredZone && tooltipPos && (
        <div
          className="fixed z-50 pointer-events-none transform -translate-x-1/2 -translate-y-full mb-3 px-3 py-2 rounded-lg bg-slate-900/95 border border-slate-700 shadow-2xl backdrop-blur text-xs"
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
            <span
              className={`font-semibold ${
                hoveredZone.current_congestion_level === 'SEVERE'
                  ? 'text-red-400'
                  : hoveredZone.current_congestion_level === 'HIGH'
                  ? 'text-orange-400'
                  : hoveredZone.current_congestion_level === 'MEDIUM'
                  ? 'text-yellow-400'
                  : 'text-emerald-400'
              }`}
            >
              {hoveredZone.current_congestion_level}
            </span>
          </div>
          <div className="flex items-center justify-between space-x-3 text-slate-400 text-[11px]">
            <span>Capacity:</span>
            <span className="font-mono text-slate-200">{hoveredZone.capacity} veh/hr</span>
          </div>
        </div>
      )}

      {/* Floating Legend (Bottom Left) */}
      <div className="absolute bottom-28 left-6 z-20 px-3.5 py-3 rounded-xl bg-slate-900/90 border border-slate-800 backdrop-blur shadow-xl text-xs space-y-1.5">
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
