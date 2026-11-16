/* Dấu "?" giải thích: rê chuột để xem, nhấp để ghim / bỏ ghim. */
import { useEffect, useRef, useState } from 'react'

export default function Hint({ text }) {
  const [hover, setHover] = useState(false)
  const [pinned, setPinned] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    if (!pinned) return
    const close = (e) => !ref.current?.contains(e.target) && setPinned(false)
    document.addEventListener('pointerdown', close)
    return () => document.removeEventListener('pointerdown', close)
  }, [pinned])

  return (
    <span className="hint" ref={ref} onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}>
      <button type="button" className={`hint-btn ${pinned ? 'on' : ''}`} onClick={() => setPinned(!pinned)} aria-label="Giải thích">?</button>
      {(hover || pinned) && <span className="hint-pop" role="tooltip">{text}</span>}
    </span>
  )
}
