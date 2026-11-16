/* Hộp thoại cấu hình 1 bước pipeline, form sinh từ schema backend (/api/health -> config). */
import { useEffect, useState } from 'react'
import Hint from './Hint'

export const defaultsOf = (fields = []) => Object.fromEntries(fields.map((f) => [f.key, f.default]))

function Field({ f, value, onChange }) {
  if (f.type === 'select') {
    return (
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {f.options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
      </select>
    )
  }
  return (
    <div className="num-field">
      <input type="range" min={f.min} max={f.max} step={f.step} value={value} onChange={(e) => onChange(+e.target.value)} />
      <input type="number" min={f.min} max={f.max} step={f.step} value={value} onChange={(e) => onChange(e.target.value === '' ? '' : +e.target.value)} />
    </div>
  )
}

export default function ConfigDialog({ step, fields, values, canRun, onApply, onClose }) {
  const [v, setV] = useState(values)
  useEffect(() => {
    const onKey = (e) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const clamp = (f, x) => (f.type === 'number' ? Math.min(f.max, Math.max(f.min, x === '' ? f.default : x)) : x)
  const apply = (run) => onApply(Object.fromEntries(fields.map((f) => [f.key, clamp(f, v[f.key])])), run)

  return (
    <div className="modal-backdrop" onPointerDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" role="dialog" aria-label={`Cấu hình ${step.name}`}>
        <div className="modal-head">
          <h3>Cấu hình — {step.name}</h3>
          <button className="btn ghost" onClick={onClose} aria-label="Đóng">✕</button>
        </div>
        <div className="modal-body">
          {fields.map((f) => (
            <div key={f.key} className="cfg-row">
              <label className="cfg-label">
                {f.label}
                <Hint text={f.hint} />
                {v[f.key] !== f.default && <span className="cfg-changed" title={`Mặc định: ${f.default}`}>đã đổi</span>}
              </label>
              <Field f={f} value={v[f.key]} onChange={(x) => setV({ ...v, [f.key]: x })} />
            </div>
          ))}
          <p className="muted small">Thay đổi áp dụng cho lần chạy tiếp theo.</p>
        </div>
        <div className="modal-foot">
          <button className="btn" onClick={() => setV(defaultsOf(fields))}>Mặc định</button>
          <span className="spacer" />
          <button className="btn" onClick={onClose}>Huỷ</button>
          <button className="btn" onClick={() => apply(false)}>Áp dụng</button>
          {canRun && <button className="btn primary" onClick={() => apply(true)}>Áp dụng & chạy lại</button>}
        </div>
      </div>
    </div>
  )
}
