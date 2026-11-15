/* Image viewer: lăn chuột = zoom quanh con trỏ, kéo nền (hoặc chuột giữa) = pan.
   Chế độ sửa:
   - Bàn tay: kéo nền = pan; kéo đỉnh = di chuyển, kéo cạnh = chèn đỉnh rồi kéo, chuột phải vào đỉnh = xoá đỉnh.
   - Bút chì / Tẩy: vẽ chú thích tự do lên lớp canvas (toạ độ ảnh), tẩy xoá theo pixel.
   Mọi thay đổi đi qua onChange(doc => patch); onEditStart() được gọi trước mỗi thao tác để lưu lịch sử undo. */
import { useEffect, useRef, useState } from 'react'
import Dropdown from './Dropdown'
import Icon, { CURSORS } from './Icon'

const QUAD_HUE = { 1: 213, 2: 18, 3: 158, 4: 330 } // màu theo cung hàm FDI
export const toothColor = (fdi) => `hsl(${QUAD_HUE[fdi[0]] ?? 0} 85% 60%)`
const centroid = (p) => p.reduce((a, [x, y]) => [a[0] + x / p.length, a[1] + y / p.length], [0, 0])
const round = ([x, y]) => [Math.round(x * 10) / 10, Math.round(y * 10) / 10]

const TOOLS = [['hand', 'hand', 'Bàn tay: di chuyển ảnh, kéo đỉnh/cạnh polygon'], ['pen', 'pencil', 'Bút chì'], ['eraser', 'eraser', 'Tẩy']]
const WIDTHS = [1, 2, 4, 8, 12, 16, 24, 32].map((w) => ({
  value: w,
  label: <span className="weight"><i style={{ height: Math.min(w, 14) }} />{w}px</span>,
}))
const COLORS = ['#ef4444', '#facc15', '#22c55e', '#3b82f6', '#ffffff']

function drawStrokes(canvas, strokes, view) {
  const dpr = window.devicePixelRatio || 1
  canvas.width = canvas.clientWidth * dpr
  canvas.height = canvas.clientHeight * dpr
  const ctx = canvas.getContext('2d')
  ctx.setTransform(dpr * view.s, 0, 0, dpr * view.s, dpr * view.tx, dpr * view.ty)
  ctx.lineCap = ctx.lineJoin = 'round'
  for (const st of strokes) { // vẽ lại theo thứ tự -> tẩy chỉ xoá nét vẽ trước nó
    ctx.globalCompositeOperation = st.mode === 'erase' ? 'destination-out' : 'source-over'
    ctx.strokeStyle = st.color
    ctx.lineWidth = st.width
    ctx.beginPath()
    st.points.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)))
    if (st.points.length === 1) ctx.lineTo(...st.points[0])
    ctx.stroke()
  }
}

/* Lớp vẽ phân tích răng khôn: trục răng khôn (đỏ = ngầm/lệch, xanh lá = bình thường), trục răng 7 (xanh dương),
   mức mặt nhai (vàng) và CEJ ước lượng (tím) của răng 7. Toạ độ ảnh gốc. */
function WisdomOverlay({ items, px, active }) {
  return (
    <g className="wisdom-axis">
      {items.filter((w) => w.m3_axis).map((w) => {
        const on = w.tooth_id === active
        const c = w.impacted ? '#ef4444' : '#22c55e'
        const [cx, cy] = w.m3_center, [ax, ay] = w.m3_axis
        const L3 = w.m2_len ? w.m2_len / 2 : 120 // nửa chiều dài mũi tên ~ nửa răng 7
        const seg = (x, y, dx, dy, l) => [x - dx * l, y - dy * l, x + dx * l, y + dy * l]
        const [x1, y1, x2, y2] = seg(cx, cy, ax, ay, L3)
        const parts = [<line key="m3" x1={x1} y1={y1} x2={x2} y2={y2} stroke={c} strokeWidth={on ? 3.5 : 2.5} vectorEffect="non-scaling-stroke" markerEnd={`url(#arrow-${w.impacted ? 'bad' : 'ok'})`} />]
        if (w.m2_axis) {
          const [bx, by] = w.m2_center, [ux, uy] = w.m2_axis
          const [p1, q1, p2, q2] = seg(bx, by, ux, uy, w.m2_len / 2)
          parts.push(<line key="m2" x1={p1} y1={q1} x2={p2} y2={q2} stroke="#60a5fa" strokeWidth={1.5} strokeDasharray={`${6 * px} ${4 * px}`} vectorEffect="non-scaling-stroke" />)
          const n = [-uy, ux], half = w.m2_len * 0.9
          for (const [lvl, col, k] of [[w.occ_level, '#facc15', 'occ'], [w.cej_level, '#d946ef', 'cej']]) {
            const t = lvl - (bx * ux + by * uy) // điểm trên trục răng 7 tại mức lvl
            const ox = bx + ux * t, oy = by + uy * t
            parts.push(<line key={k} x1={ox - n[0] * half} y1={oy - n[1] * half} x2={ox + n[0] * half} y2={oy + n[1] * half} stroke={col} strokeWidth={1.5} strokeOpacity={0.9} vectorEffect="non-scaling-stroke" />)
          }
        }
        const below = w.fdi[0] === '3' || w.fdi[0] === '4'
        parts.push(<text key="t" x={cx} y={cy + (below ? 1 : -1) * (L3 + 22 * px)} className="fdi" fontSize={13 * px} strokeWidth={3 * px} fill={c}>
          {w.fdi} {w.winter_angle > 0 ? '+' : ''}{Math.round(w.winter_angle)}°{w.depth ? ` ${w.depth}` : ''}
        </text>)
        return <g key={w.fdi}>{parts}</g>
      })}
    </g>
  )
}

export default function ImageViewer({ src, teeth, strokes, opts, editable, hover, onHover, onChange, onEditStart, onUndo, onRedo, wisdom = [] }) {
  const boxRef = useRef(null)
  const svgRef = useRef(null)
  const canvasRef = useRef(null)
  const drag = useRef(null) // {pan:[cx,cy,tx,ty]} | {id, idx} | {draw:true}
  const [size, setSize] = useState(null)
  const [view, setView] = useState({ s: 1, tx: 0, ty: 0 })
  const [boxSize, setBoxSize] = useState(null)
  const [cursor, setCursor] = useState(null)
  const [tool, setTool] = useState('hand')
  const [brush, setBrush] = useState({ color: COLORS[0], width: 4 })
  const editPoly = editable && tool === 'hand'
  const drawing = editable && tool !== 'hand'

  const fit = (w = size?.w, h = size?.h) => {
    const r = boxRef.current.getBoundingClientRect()
    const s = Math.min(r.width / w, r.height / h) * 0.95
    setView({ s, tx: (r.width - w * s) / 2, ty: (r.height - h * s) / 2 })
  }

  useEffect(() => {
    setSize(null)
    if (!src) return
    const img = new Image()
    img.onload = () => { setSize({ w: img.naturalWidth, h: img.naturalHeight }); fit(img.naturalWidth, img.naturalHeight) }
    img.src = src
  }, [src])

  useEffect(() => {
    const el = boxRef.current
    const onWheel = (e) => { // non-passive để chặn cuộn trang
      e.preventDefault()
      const r = el.getBoundingClientRect()
      const mx = e.clientX - r.left, my = e.clientY - r.top
      setView((v) => {
        const s = Math.min(50, Math.max(0.02, v.s * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
        return { s, tx: mx - (mx - v.tx) * (s / v.s), ty: my - (my - v.ty) * (s / v.s) }
      })
    }
    const ro = new ResizeObserver(([e]) => setBoxSize([e.contentRect.width, e.contentRect.height]))
    el.addEventListener('wheel', onWheel, { passive: false })
    ro.observe(el)
    return () => { el.removeEventListener('wheel', onWheel); ro.disconnect() }
  }, [])

  useEffect(() => { canvasRef.current && drawStrokes(canvasRef.current, strokes, view) }, [strokes, view, boxSize, size])

  const toImage = (e) => {
    const r = svgRef.current.getBoundingClientRect()
    return [(e.clientX - r.left - view.tx) / view.s, (e.clientY - r.top - view.ty) / view.s]
  }
  const setPoly = (id, fn) => onChange((d) => ({ teeth: d.teeth.map((t) => (t.id === id ? { ...t, polygon: fn(t.polygon) } : t)) }))
  const begin = (e, d) => { e.stopPropagation(); svgRef.current.setPointerCapture(e.pointerId); drag.current = d }

  const onDown = (e) => {
    if (drawing && e.button === 0) {
      onEditStart()
      const stroke = { mode: tool === 'eraser' ? 'erase' : 'pen', color: brush.color, width: brush.width / view.s, points: [round(toImage(e))] }
      onChange((d) => ({ strokes: [...d.strokes, stroke] }))
      return begin(e, { draw: true })
    }
    begin(e, { pan: [e.clientX, e.clientY, view.tx, view.ty] })
  }
  const onVertexDown = (e, id, idx) => {
    if (e.button === 1) return
    onEditStart()
    if (e.button === 2) { e.stopPropagation(); return setPoly(id, (p) => (p.length > 3 ? p.filter((_, i) => i !== idx) : p)) }
    begin(e, { id, idx })
  }
  const onEdgeDown = (e, id, idx) => {
    if (e.button !== 0) return
    onEditStart()
    const p = round(toImage(e))
    setPoly(id, (poly) => [...poly.slice(0, idx + 1), p, ...poly.slice(idx + 1)])
    begin(e, { id, idx: idx + 1 })
  }
  const onMove = (e) => {
    const p = round(toImage(e))
    setCursor(p)
    const d = drag.current
    if (!d) return
    if (d.pan) setView((v) => ({ ...v, tx: d.pan[2] + e.clientX - d.pan[0], ty: d.pan[3] + e.clientY - d.pan[1] }))
    else if (d.draw) onChange(({ strokes: s }) => ({ strokes: [...s.slice(0, -1), { ...s.at(-1), points: [...s.at(-1).points, p] }] }))
    else setPoly(d.id, (poly) => poly.map((q, i) => (i === d.idx ? p : q)))
  }

  const active = drag.current?.id ?? hover
  const px = 1 / view.s // 1 pixel màn hình theo đơn vị ảnh
  const hovered = teeth.find((t) => t.id === active)

  return (
    <div className="viewer" ref={boxRef}>
      {!src && <div className="viewer-empty">Chưa có ảnh — nhấn “Tải ảnh”</div>}
      {size && (
        <svg ref={svgRef} className="viewer-svg" style={drawing ? { cursor: CURSORS[tool] } : undefined} onPointerDown={onDown} onPointerMove={onMove}
          onPointerUp={() => (drag.current = null)} onPointerLeave={() => setCursor(null)} onContextMenu={(e) => e.preventDefault()}>
          <defs>
            {[['ok', '#22c55e'], ['bad', '#ef4444']].map(([k, c]) => (
              <marker key={k} id={`arrow-${k}`} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
                <path d="M0,0 L10,5 L0,10 z" fill={c} />
              </marker>
            ))}
          </defs>
          <g transform={`translate(${view.tx} ${view.ty}) scale(${view.s})`}>
            <image href={src} width={size.w} height={size.h} style={{ imageRendering: view.s > 2 ? 'pixelated' : 'auto' }} />
            <g opacity={opts.opacity}>
              {teeth.map((t) => {
                const on = t.id === active
                const c = toothColor(t.fdi)
                const poly = t.polygon
                const [cx, cy] = centroid(poly)
                return (
                  <g key={t.id} onPointerEnter={() => onHover(t.id)} onPointerLeave={() => onHover(null)}>
                    <polygon points={poly.join(' ')} pointerEvents="all" vectorEffect="non-scaling-stroke"
                      fill={c} fillOpacity={opts.mask ? (on ? 0.6 : 0.35) : on ? 0.15 : 0}
                      stroke={on ? '#fff' : c} strokeOpacity={on || opts.polygon ? 1 : 0.4}
                      strokeWidth={on ? 2.5 : opts.polygon ? 1.5 : 1} />
                    {on && editPoly && poly.map((p, i) => {
                      const q = poly[(i + 1) % poly.length]
                      return <line key={`e${i}`} className="edge-hit" x1={p[0]} y1={p[1]} x2={q[0]} y2={q[1]}
                        stroke="transparent" strokeWidth={10} vectorEffect="non-scaling-stroke" onPointerDown={(e) => onEdgeDown(e, t.id, i)} />
                    })}
                    {(opts.polygon || (on && editPoly)) && poly.map((p, i) => (
                      <circle key={i} cx={p[0]} cy={p[1]} r={(on && editPoly ? 5 : 2) * px} fill={on ? '#fff' : c}
                        stroke={on ? c : 'none'} strokeWidth={1.5} vectorEffect="non-scaling-stroke"
                        className={on && editPoly ? 'vertex' : undefined}
                        onPointerDown={on && editPoly ? (e) => onVertexDown(e, t.id, i) : undefined} />
                    ))}
                    {opts.fdi && <text x={cx} y={cy} className="fdi" fontSize={(on ? 16 : 13) * px} strokeWidth={3 * px}>{t.fdi}</text>}
                  </g>
                )
              })}
            </g>
            {opts.wisdom && wisdom.length > 0 && <WisdomOverlay items={wisdom} px={px} active={active} />}
          </g>
          {drawing && cursor && ( // vòng tròn xem trước cỡ bút
            <circle cx={view.tx + cursor[0] * view.s} cy={view.ty + cursor[1] * view.s} r={brush.width / 2}
              fill="none" stroke={tool === 'eraser' ? '#fff' : brush.color} strokeDasharray={tool === 'eraser' ? '3 2' : undefined} pointerEvents="none" />
          )}
        </svg>
      )}
      {size && <canvas ref={canvasRef} className="viewer-canvas" style={{ opacity: opts.opacity }} />}

      {size && editable && (
        <div className="draw-tools">
          {TOOLS.map(([k, icon, title]) => (
            <button key={k} className={`tool ${tool === k ? 'active' : ''}`} title={title} onClick={() => setTool(k)}><Icon name={icon} size={17} /></button>
          ))}
          <span className="sep" />
          <Dropdown className="weight-dd" title="Độ dày nét" value={brush.width} options={WIDTHS} onChange={(w) => setBrush({ ...brush, width: w })} />
          <span className="sep" />
          {COLORS.map((c) => (
            <button key={c} className={`swatch ${brush.color === c ? 'active' : ''}`} style={{ background: c }} title={c}
              onClick={() => setBrush({ ...brush, color: c })} />
          ))}
          <label className={`swatch custom ${COLORS.includes(brush.color) ? '' : 'active'}`} title="Màu khác"
            style={COLORS.includes(brush.color) ? undefined : { background: brush.color }}>
            <input type="color" value={brush.color} onChange={(e) => setBrush({ ...brush, color: e.target.value })} />
          </label>
          <span className="sep" />
          <button className="tool" title="Hoàn tác (Ctrl+Z)" onClick={onUndo}><Icon name="undo" size={17} /></button>
          <button className="tool" title="Làm lại (Ctrl+Y)" onClick={onRedo}><Icon name="redo" size={17} /></button>
        </div>
      )}
      {size && <div className="viewer-tools"><button className="btn small" onClick={() => fit()}>Fit</button></div>}
      {size && (
        <div className="viewer-status">
          {editPoly && <span>Kéo đỉnh/cạnh để sửa · Chuột phải: xoá đỉnh · Ctrl+Z: hoàn tác</span>}
          {hovered && <span>Răng {hovered.fdi} · {(hovered.conf * 100).toFixed(0)}% · {hovered.polygon.length} đỉnh</span>}
          {cursor && <span>x {cursor[0].toFixed(0)}, y {cursor[1].toFixed(0)}</span>}
          <span>{(view.s * 100).toFixed(0)}%</span>
          <span>{size.w} × {size.h} px</span>
        </div>
      )}
    </div>
  )
}
