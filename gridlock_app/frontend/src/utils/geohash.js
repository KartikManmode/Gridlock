const BASE32 = '0123456789bcdefghjkmnpqrstuvwxyz';

/**
 * Decodes geohash string into latitude and longitude intervals.
 * @param {string} geohash
 * @returns {{ minLat: number, maxLat: number, minLon: number, maxLon: number }}
 */
export function decodeGeohashBBox(geohash) {
  let isEven = true;
  let latInterval = [-90.0, 90.0];
  let lonInterval = [-180.0, 180.0];

  for (let i = 0; i < geohash.length; i++) {
    const c = geohash[i].toLowerCase();
    const cd = BASE32.indexOf(c);
    if (cd === -1) break;

    for (const mask of [16, 8, 4, 2, 1]) {
      if (isEven) {
        const mid = (lonInterval[0] + lonInterval[1]) / 2;
        if (cd & mask) {
          lonInterval[0] = mid;
        } else {
          lonInterval[1] = mid;
        }
      } else {
        const mid = (latInterval[0] + latInterval[1]) / 2;
        if (cd & mask) {
          latInterval[0] = mid;
        } else {
          latInterval[1] = mid;
        }
      }
      isEven = !isEven;
    }
  }

  return {
    minLat: latInterval[0],
    maxLat: latInterval[1],
    minLon: lonInterval[0],
    maxLon: lonInterval[1],
  };
}

/**
 * Decodes geohash to center { lat, lon }.
 * @param {string} geohash
 * @returns {{ lat: number, lon: number }}
 */
export function decodeGeohash(geohash) {
  const bbox = decodeGeohashBBox(geohash);
  return {
    lat: (bbox.minLat + bbox.maxLat) / 2,
    lon: (bbox.minLon + bbox.maxLon) / 2,
  };
}

/**
 * Returns GeoJSON polygon coordinates for a geohash bounding box.
 * @param {string} geohash
 * @returns {number[][][]}
 */
export function geohashToPolygonCoordinates(geohash) {
  const bbox = decodeGeohashBBox(geohash);
  return [
    [
      [bbox.minLon, bbox.minLat],
      [bbox.maxLon, bbox.minLat],
      [bbox.maxLon, bbox.maxLat],
      [bbox.minLon, bbox.maxLat],
      [bbox.minLon, bbox.minLat],
    ],
  ];
}

/**
 * Maps congestion level to Tailwind & HEX colors.
 * @param {'LOW' | 'MEDIUM' | 'HIGH' | 'SEVERE'} level
 * @returns {{ hex: string, fillHex: string, badgeClass: string }}
 */
export function getCongestionColor(level) {
  switch (level) {
    case 'SEVERE':
      return {
        hex: '#dc2626',
        fillHex: 'rgba(220, 38, 38, 0.65)',
        badgeClass: 'bg-red-500/20 text-red-400 border-red-500/40',
      };
    case 'HIGH':
      return {
        hex: '#f97316',
        fillHex: 'rgba(249, 115, 22, 0.55)',
        badgeClass: 'bg-orange-500/20 text-orange-400 border-orange-500/40',
      };
    case 'MEDIUM':
      return {
        hex: '#eab308',
        fillHex: 'rgba(234, 179, 8, 0.45)',
        badgeClass: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/40',
      };
    case 'LOW':
    default:
      return {
        hex: '#22c55e',
        fillHex: 'rgba(34, 197, 94, 0.35)',
        badgeClass: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/40',
      };
  }
}
