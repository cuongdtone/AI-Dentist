import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import Dropdown from '../components/Dropdown'
import Icon from '../components/Icon'
import ImageViewer from '../components/ImageViewer'
import PipelineGraph from '../components/PipelineGraph'
import ResultPanel from '../components/ResultPanel'

const EMPTY = { teeth: [], strokes: [] }
const MODES = [
  { value: 'edit', label: <><Icon name="pencil" size={14} /> Chỉnh sửa</>, desc: 'Sửa polygon, vẽ chú thích' },
  { value: 'view', label: <><Icon name="eye" size={14} /> Chỉ xem</>, desc: 'Xem kết quả, không thay đổi' },
]
// Độ rộng panel trái / phải (% màn hình), kéo thanh chia để đổi; nhớ theo trình duyệt
const DEFAULT_SIZES = { left: 16, right: 30 }
const loadSizes = () => { try { return JSON.parse(localStorage.getItem('diagSizes')) ?? DEFAULT_SIZES } catch { return DEFAULT_SIZES } }
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

/* Luôn được mount sau lần mở đầu tiên (xem main.jsx) -> chuyển tab không mất ảnh/kết quả. */
export default function Diagnosis({ active }) {
  const [stepDefs, setStepDefs] = useState([])
  const [file, setFile] = useState(null)
  const [src, setSrc] = useState(null)
  const [job, setJob] = useState(null)
  const [busy, setBusy] = useState(false) // đang upload
  const [doc, setDoc] = useState(EMPTY) // dữ liệu chỉnh sửa được: polygon răng + nét vẽ
  const [dirty, setDirty] = useState(false)
  const [mode, setMode] = useState('view') // view | edit
  const [hover, setHover] = useState(null)
  const [opts, setOpts] = useState({ opacity: 0.8, mask: false, polygon: false, fdi: true, wisdom: true })
  const [msg, setMsg] = useState('')
  const [sizes, setSizes] = useState(loadSizes)
  const diagRef = useRef(null)
  const inputRef = useRef(null)
  const timer = useRef(null)
  const docRef = useRef(doc)
  const past = useRef([])
  const future = useRef([])
  docRef.current = doc

  useEffect(() => {
    api('/health').then((h) => setStepDefs(h.steps)).catch((e) => setMsg(e.message))
    return () => clearInterval(timer.current)
  }, [])
  useEffect(() => () => src && URL.revokeObjectURL(src), [src])

  // --- Lịch sử chỉnh sửa (undo / redo) ---
  const change = (fn) => { setDoc((d) => ({ ...d, ...fn(d) })); setDirty(true) }
  const snapshot = () => { past.current = [...past.current.slice(-99), docRef.current]; future.current = [] }
  const undo = () => {
    const prev = past.current.pop()
    if (!prev) return
    future.current.push(docRef.current)
    setDoc(prev)
    setDirty(true)
  }
  const redo = () => {
    const next = future.current.pop()
    if (!next) return
    past.current.push(docRef.current)
    setDoc(next)
    setDirty(true)
  }
  useEffect(() => {
    if (!active || mode !== 'edit') return
    const onKey = (e) => {
      if (!(e.ctrlKey || e.metaKey)) return
      const k = e.key.toLowerCase()
      if (k === 'z' && !e.shiftKey) { e.preventDefault(); undo() }
      else if (k === 'y' || (k === 'z' && e.shiftKey)) { e.preventDefault(); redo() }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [active, mode])

  // --- Kéo thanh chia panel ---
  useEffect(() => { try { localStorage.setItem('diagSizes', JSON.stringify(sizes)) } catch { /* bỏ qua */ } }, [sizes])
  const startResize = (side) => (e) => {
    e.preventDefault()
    const r = diagRef.current.getBoundingClientRect()
    const move = (ev) => {
      const x = ((ev.clientX - r.left) / r.width) * 100
      setSizes((s) => (side === 'left'
        ? { ...s, left: clamp(x, 10, 100 - s.right - 30) }
        : { ...s, right: clamp(100 - x, 15, 100 - s.left - 30) }))
    }
    const up = () => { window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up); document.body.classList.remove('resizing') }
    document.body.classList.add('resizing')
    window.addEventListener('pointermove', move)
    window.addEventListener('pointerup', up)
  }

  // --- Chạy pipeline ---
  const running = busy || job?.status === 'running'
  const reset = () => {
    clearInterval(timer.current)
    setJob(null); setDoc(EMPTY); setDirty(false); setHover(null); setMsg('')
    past.current = []; future.current = []
  }

  const poll = (id) => {
    timer.current = setInterval(async () => {
      try {
        const j = await api(`/jobs/${id}`)
        setJob(j)
        setDoc((d) => (d.teeth.length ? d : { ...d, teeth: j.result.teeth })) // vẽ lại ngay khi phân đoạn xong
        if (j.status !== 'running') clearInterval(timer.current)
      } catch (e) {
        setMsg(e.message)
        clearInterval(timer.current)
      }
    }, 300)
  }

  const start = async (f = file) => {
    reset()
    setBusy(true)
    try {
      const form = new FormData()
      form.append('file', f)
      const j = await api('/jobs', { method: 'POST', form })
      setJob(j)
      poll(j.id)
    } catch (e) {
      setMsg(e.message)
    } finally {
      setBusy(false)
    }
  }

  const pick = (e) => { // tải ảnh -> chạy luôn
    const f = e.target.files[0]
    e.target.value = ''
    if (!f) return
    setFile(f)
    setSrc(URL.createObjectURL(f))
    start(f)
  }

  const cancel = async () => job && setJob(await api(`/jobs/${job.id}/cancel`, { method: 'POST' }))

  const save = async () => {
    try {
      const r = await api(`/inferences/${job.inference_id}`, { method: 'PUT', json: doc })
      if (r.findings) setJob((j) => ({ ...j, result: { ...j.result, findings: r.findings } })) // răng khôn tính lại theo polygon đã sửa
      setDirty(false)
      setMsg('Đã lưu chỉnh sửa')
    } catch (e) {
      setMsg(e.message)
    }
  }

  const steps = [
    { key: 'input', name: 'Ảnh đầu vào', status: file ? 'done' : 'pending', note: file?.name },
    ...stepDefs.map(([key, name]) => job?.steps.find((s) => s.key === key) ?? { key, name, status: 'pending' }),
  ]
  const opt = (k) => (e) => setOpts({ ...opts, [k]: e.target.type === 'checkbox' ? e.target.checked : +e.target.value })
  const editable = mode === 'edit' && doc.teeth.length > 0

  return (
    <div className="diag" ref={diagRef} style={{ gridTemplateColumns: `${sizes.left}% 4px 1fr 4px ${sizes.right}%` }}>
      <section className="panel left">
        <div className="toolbar">
          <input ref={inputRef} type="file" accept="image/*" hidden onChange={pick} />
          <button className="btn small" onClick={() => inputRef.current.click()} disabled={running}>⬆ Tải ảnh</button>
          <button className="btn small primary" onClick={() => start()} disabled={!file || running}>▶ Chạy</button>
          <button className="btn small danger" onClick={cancel} disabled={!job || job.status !== 'running'}>■ Huỷ</button>
        </div>
        <div className="graph"><PipelineGraph steps={steps} /></div>
      </section>
      <div className="splitter" onPointerDown={startResize('left')} onDoubleClick={() => setSizes(DEFAULT_SIZES)} title="Kéo để đổi kích thước · nhấp đúp: mặc định" />

      <section className="panel center">
        <div className="toolbar">
          <label><input type="checkbox" checked={opts.mask} onChange={opt('mask')} /> Mask</label>
          <label><input type="checkbox" checked={opts.polygon} onChange={opt('polygon')} /> Polygon</label>
          <label><input type="checkbox" checked={opts.fdi} onChange={opt('fdi')} /> FDI</label>
          <label title="Trục răng khôn, trục răng 7, mặt nhai, CEJ"><input type="checkbox" checked={opts.wisdom} onChange={opt('wisdom')} /> Răng khôn</label>
          <label>Opacity <input type="range" min="0" max="1" step="0.05" value={opts.opacity} onChange={opt('opacity')} /></label>
          <span className="spacer" />
          <Dropdown className={`mode-dd ${mode}`} title="Chế độ" value={mode} options={MODES} onChange={setMode} align="right" />
          <button className="btn" onClick={save} disabled={mode !== 'edit' || !dirty || !job?.inference_id}>💾 Lưu</button>
        </div>
        <ImageViewer src={src} teeth={doc.teeth} strokes={doc.strokes} opts={opts} editable={editable}
          hover={hover} onHover={setHover} onChange={change} onEditStart={snapshot} onUndo={undo} onRedo={redo}
          wisdom={job?.result.findings?.wisdom ?? []} />
      </section>
      <div className="splitter" onPointerDown={startResize('right')} onDoubleClick={() => setSizes(DEFAULT_SIZES)} title="Kéo để đổi kích thước · nhấp đúp: mặc định" />

      <section className="panel right">
        <ResultPanel job={job} busy={busy} teeth={doc.teeth} hover={hover} onHover={setHover} msg={msg} />
      </section>
    </div>
  )
}
