const CITY_CENTER_LAT = 12.9716;
const CITY_CENTER_LON = 77.5946;

export function decodeGeohash(gh) {
  const base32 = '0123456789bcdefghjkmnpqrstuvwxyz';
  const latInterval = [-90.0, 90.0];
  const lonInterval = [-180.0, 180.0];
  let isEven = true;

  for (const char of gh.toLowerCase()) {
    const cd = base32.indexOf(char);
    if (cd === -1) break;
    for (const mask of [16, 8, 4, 2, 1]) {
      if (isEven) {
        const mid = (lonInterval[0] + lonInterval[1]) / 2.0;
        if (cd & mask) {
          lonInterval[0] = mid;
        } else {
          lonInterval[1] = mid;
        }
      } else {
        const mid = (latInterval[0] + latInterval[1]) / 2.0;
        if (cd & mask) {
          latInterval[0] = mid;
        } else {
          latInterval[1] = mid;
        }
      }
      isEven = !isEven;
    }
  }

  const lat = (latInterval[0] + latInterval[1]) / 2.0;
  const lon = (lonInterval[0] + lonInterval[1]) / 2.0;
  return { lat, lon };
}

export function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371.0;
  const dLat = ((lat2 - lat1) * Math.PI) / 180.0;
  const dLon = ((lon2 - lon1) * Math.PI) / 180.0;
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180.0) *
      Math.cos((lat2 * Math.PI) / 180.0) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

export function classifyZoneTier(lat, lon) {
  const dist = haversineKm(lat, lon, CITY_CENTER_LAT, CITY_CENTER_LON);
  if (dist <= 7.0) return 'Core City';
  if (dist <= 15.0) return 'Mid-Ring';
  return 'Periphery';
}
