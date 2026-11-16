/* Graph pipeline kiểu tool-block (Cognex VisionPro / QuickBuild): header màu, cổng in/out, footer trạng thái + thời gian.
   Nút ⚙ trên header = Configure; chuột phải vào node = menu Configure / Disable·Enable / Hide·Show. */
import { Background, Handle, Position, ReactFlow } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { useEffect, useRef, useState } from 'react'
import Icon from './Icon'

export const STATUS = {
  done: ['#22c55e', 'OK'],
  running: ['#eab308', 'Đang chạy'],
  error: ['#ef4444', 'Lỗi'],
  pending: ['#6b7280', 'Chờ'],
  skipped: ['#6b7280', 'Chưa có model'],
  disabled: ['#6b7280', 'Đã tắt'],
}

// Cổng vào / ra của từng bước (chỉ để hiển thị)
const PORTS = {
  input: [[], ['Image']],
  seg: [['Image In'], ['Teeth']],
  wisdom: [['Teeth', 'CEJ'], ['Wisdom']],
  caries: [['Image In', 'Teeth'], ['Caries']],
  cej: [['Image In', 'Teeth'], ['CEJ', 'Regions']],
  perio: [['Teeth', 'Regions'], ['Bone loss']],
}

function StepNode({ data }) {
  const [color, label] = STATUS[data.status]
  const [ins, outs] = PORTS[data.key] ?? [[], []]
  return (
    <div className={`tb-node ${data.status} ${data.key === 'input' ? 'is-input' : ''}`}>
      <Handle type="target" position={Position.Top} />
      <div className="tb-head">
        <Icon name={data.key === 'input' ? 'image' : 'cpu'} size={12} />
        <span className="tb-name">{data.name}</span>
        {data.hidden && <span className="tb-flag" title="Đang ẩn trên ảnh"><Icon name="eyeOff" size={12} /></span>}
        {data.hasConfig && (
          <button className="tb-gear nodrag" title="Configure" onClick={() => data.onAction(data.key, 'config')}>
            <Icon name="gear" size={13} />
          </button>
        )}
      </div>
      <div className="tb-body">
        <div>{ins.map((p) => <div key={p} className="port in">{p}</div>)}</div>
        <div>{outs.map((p) => <div key={p} className="port out">{p}</div>)}</div>
      </div>
      <div className="tb-foot" title={data.error || data.note || ''}>
        <span style={{ color }}>● {data.error ? 'Lỗi' : data.note || label}</span>
        <span>{data.ms != null ? `${data.ms} ms` : ''}</span>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

const nodeTypes = { step: StepNode }

function ContextMenu({ menu, step, onAction, onClose }) {
  const ref = useRef(null)
  useEffect(() => {
    const close = (e) => !ref.current?.contains(e.target) && onClose()
    const esc = (e) => e.key === 'Escape' && onClose()
    document.addEventListener('pointerdown', close)
    window.addEventListener('keydown', esc)
    return () => { document.removeEventListener('pointerdown', close); window.removeEventListener('keydown', esc) }
  }, [onClose])
  const items = [
    ['config', 'gear', 'Configure…', step.hasConfig],
    ['toggle', step.enabled ? 'power' : 'play', step.enabled ? 'Disable' : 'Enable', step.canToggle],
    ['visible', step.hidden ? 'eye' : 'eyeOff', step.hidden ? 'Show (hiện trên ảnh)' : 'Hide (ẩn trên ảnh)', step.canHide],
  ]
  return (
    <div className="ctx-menu" ref={ref} style={{ left: menu.x, top: menu.y }}>
      <div className="ctx-title">{step.name}</div>
      {items.map(([a, icon, text, ok]) => (
        <button key={a} disabled={!ok} onClick={() => { onAction(step.key, a); onClose() }}>
          <Icon name={icon} size={14} />{text}
        </button>
      ))}
    </div>
  )
}

export default function PipelineGraph({ steps, onAction }) {
  const wrap = useRef(null)
  const [menu, setMenu] = useState(null) // {key, x, y}
  const nodes = steps.map((s, i) => ({ id: s.key, type: 'step', position: { x: 0, y: i * 96 }, data: { ...s, onAction }, draggable: false }))
  const edges = steps.slice(1).map((s, i) => ({
    id: `e${i}`, source: steps[i].key, target: s.key, type: 'smoothstep',
    animated: s.status === 'running',
    style: { stroke: steps[i].status === 'done' ? '#2dd4bf' : '#4b5563', strokeWidth: 1.5, strokeDasharray: s.status === 'disabled' ? '4 3' : undefined },
  }))
  const onNodeContextMenu = (e, node) => {
    e.preventDefault()
    const r = wrap.current.getBoundingClientRect()
    setMenu({ key: node.id, x: Math.min(e.clientX - r.left, r.width - 190), y: e.clientY - r.top })
  }
  const step = menu && steps.find((s) => s.key === menu.key)

  return (
    <div ref={wrap} className="graph-wrap" onContextMenu={(e) => e.preventDefault()}>
      <ReactFlow key={steps.length} nodes={nodes} edges={edges} nodeTypes={nodeTypes} colorMode="dark"
        fitView fitViewOptions={{ maxZoom: 1, padding: 0.1 }} onNodeContextMenu={onNodeContextMenu} onPaneClick={() => setMenu(null)}
        nodesConnectable={false} elementsSelectable={false} proOptions={{ hideAttribution: true }}>
        <Background gap={16} color="#2a2a2a" />
      </ReactFlow>
      {step && <ContextMenu menu={menu} step={step} onAction={onAction} onClose={() => setMenu(null)} />}
    </div>
  )
}
