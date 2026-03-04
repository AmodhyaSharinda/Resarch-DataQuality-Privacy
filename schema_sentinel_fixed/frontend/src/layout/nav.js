export const NAV = [
  {
    label: 'CORE',
    items: [
      { id: 'dashboard', path: '/', icon: '📊', text: 'Dashboard' },
      { id: 'stream', path: '/stream', icon: '📡', text: 'Live Stream' },
      { id: 'drift', path: '/drift', icon: '⚠️', text: 'Drift Events' },
      { id: 'staging', path: '/staging', icon: '🧪', text: 'Staging Queue' },
       { id: 'risk', path: '/risk', icon: '🎯', text: 'Risk Scoring' },
    ],
  },
  // {
  //   label: 'INTELLIGENCE',
  //   items: [
  //     // { id: 'xai', path: '/xai', icon: '🧠', text: 'XAI Explainer' },
  //     { id: 'risk', path: '/risk', icon: '🎯', text: 'Risk Scoring' },
  //     // { id: 'rename', path: '/rename', icon: '🔁', text: 'Rename Engine' },
  //   ],
  // },
  {
    label: 'GOVERNANCE',
    items: [
      { id: 'registry', path: '/registry', icon: '🧬', text: 'Schema Registry' },
      { id: 'audit', path: '/audit', icon: '🧾', text: 'Audit Log' },
      { id: 'datasets', path: '/datasets', icon: '🗂️', text: 'Datasets' },
    ],
  },
]

export function getRouteLabel(pathname) {
  const all = NAV.flatMap((g) => g.items)
  const hit = all.find((it) => it.path === pathname)
  return hit?.text || 'Dashboard'
}
