/* Ô chọn xổ xuống: nút hiển thị lựa chọn hiện tại; options = [{value, label, desc?}]. */
import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'

export default function Dropdown({ value, options, onChange, title, className = '', align = 'left' }) {
  const [open, setOpen] = useState(false)
  const ref = useRef(null)
  useEffect(() => {
    if (!open) return
    const close = (e) => !ref.current?.contains(e.target) && setOpen(false)
    document.addEventListener('pointerdown', close)
    return () => document.removeEventListener('pointerdown', close)
  }, [open])
  const cur = options.find((o) => o.value === value) ?? options[0]

  return (
    <div className={`dropdown ${className}`} ref={ref}>
      <button className={`dd-btn ${open ? 'open' : ''}`} title={title} onClick={() => setOpen(!open)}>
        {cur.label}<Icon name="chevron" size={14} />
      </button>
      {open && (
        <div className={`dd-menu ${align}`}>
          {options.map((o) => (
            <button key={o.value} className={`dd-item ${o.value === value ? 'active' : ''}`} onClick={() => { onChange(o.value); setOpen(false) }}>
              <span className="dd-check">{o.value === value && <Icon name="check" size={14} />}</span>
              <span>
                <span className="dd-label">{o.label}</span>
                {o.desc && <small>{o.desc}</small>}
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}
