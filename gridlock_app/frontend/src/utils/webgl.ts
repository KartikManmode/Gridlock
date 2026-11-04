/**
 * Utility to safely detect WebGL support in the user's browser.
 */
export function checkWebGLSupport(): { supported: boolean; reason?: string } {
  if (typeof window === 'undefined') {
    return { supported: false, reason: 'SSR / Window undefined' };
  }

  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
    if (!gl) {
      return { supported: false, reason: 'WebGL context creation failed (hardware acceleration disabled)' };
    }
    return { supported: true };
  } catch (err: any) {
    return { supported: false, reason: err?.message || 'WebGL check exception' };
  }
}
