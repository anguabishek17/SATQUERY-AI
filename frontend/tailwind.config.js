/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Named to the diagram's ramps so the dashboard visually matches the
        // architecture doc: gray = structural, teal = single-image tools,
        // coral = multi-image tools, amber = confidence/output.
        base: {
          950: '#0B0F0E',
          900: '#12172F' /* fallback */,
        },
        surface: '#0F1413',
        surface2: '#161C1B',
        border: '#263230',
        teal: { DEFAULT: '#1D9E75', dim: '#0F6E56' },
        coral: { DEFAULT: '#D85A30', dim: '#993C1D' },
        amber: { DEFAULT: '#BA7517', dim: '#854F0B' },
        ink: { DEFAULT: '#E7EFEC', dim: '#8FA39C' },
      },
      fontFamily: {
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'SFMono-Regular', 'monospace'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
