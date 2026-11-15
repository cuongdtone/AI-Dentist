/* Icon SVG (path theo bộ Lucide, ISC) + con trỏ chuột tạo từ chính các icon đó. */
const PATHS = {
  hand: ['M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2', 'M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2', 'M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8',
    'M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15'],
  pencil: ['M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z', 'm15 5 4 4'],
  eraser: ['m7 21-4.3-4.3c-1-1-1-2.5 0-3.4l9.6-9.6c1-1 2.5-1 3.4 0l5.6 5.6c1 1 1 2.5 0 3.4L13 21', 'M22 21H7', 'm5 11 9 9'],
  undo: ['M9 14 4 9l5-5', 'M4 9h10.5a5.5 5.5 0 0 1 5.5 5.5a5.5 5.5 0 0 1-5.5 5.5H11'],
  redo: ['m15 14 5-5-5-5', 'M20 9H9.5A5.5 5.5 0 0 0 4 14.5A5.5 5.5 0 0 0 9.5 20H13'],
  eye: ['M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0', 'M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6'],
  chevron: ['m6 9 6 6 6-6'],
  image: ['M5 3h14a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z', 'M9 7a2 2 0 1 0 0 4a2 2 0 1 0 0-4', 'm21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21'],
  cpu: ['M6 6h12v12H6z', 'M12 20v2', 'M12 2v2', 'M17 20v2', 'M17 2v2', 'M2 12h2', 'M2 17h2', 'M2 7h2', 'M20 12h2', 'M20 17h2', 'M20 7h2', 'M7 20v2', 'M7 2v2'],
  check: ['M20 6 9 17l-5-5'],
}

export default function Icon({ name, size = 16 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      {PATHS[name].map((d) => <path key={d} d={d} />)}
    </svg>
  )
}

const cursorOf = (name, hx, hy) => {
  const p = PATHS[name].map((d) => `<path d='${d}'/>`).join('')
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='24' height='24' viewBox='-1 -1 26 26' fill='none' stroke-linecap='round' stroke-linejoin='round'>`
    + `<g stroke='black' stroke-width='4'>${p}</g><g stroke='white' stroke-width='2'>${p}</g></svg>`
  return `url("data:image/svg+xml;utf8,${encodeURIComponent(svg)}") ${hx} ${hy}, crosshair`
}

export const CURSORS = { pen: cursorOf('pencil', 2, 22), eraser: cursorOf('eraser', 5, 20) }
