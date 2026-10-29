type Name = 'arrow' | 'external' | 'signal' | 'layers' | 'chart' | 'book' | 'check' | 'info'
const paths: Record<Name, string> = {
  arrow: 'M5 12h14m-6-6 6 6-6 6',
  external: 'M14 4h6v6m0-6L10 14M10 4H5a1 1 0 0 0-1 1v14a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-5',
  signal: 'M3 12h4l3-7 4 14 3-7h4',
  layers: 'm12 3 10 5-10 5L2 8l10-5Zm-10 9 10 5 10-5M2 16l10 5 10-5',
  chart: 'M4 4v16h16M8 15v-4m5 4V7m5 8v-6',
  book: 'M12 6c-3-3-7-2-9-1v14c3-1 6-1 9 1m0-14c3-3 7-2 9-1v14c-3-1-6-1-9 1V6Z',
  check: 'm5 12 4 4L19 6',
  info: 'M12 11v6m0-10v.1M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0Z',
}
export function Icon({ name, size = 20 }: { name: Name; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>
}
