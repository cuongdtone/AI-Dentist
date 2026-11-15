/* Graph pipeline kiểu tool-block (Cognex VisionPro / QuickBuild): header màu, cổng in/out, footer trạng thái + thời gian. */
import { Background, Handle, Position, ReactFlow } from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import Icon from './Icon'

export const STATUS = {
  done: ['#22c55e', 'OK'],
  running: ['#eab308', 'Đang chạy'],
  error: ['#ef4444', 'Lỗi'],
  pending: ['#6b7280', 'Chờ'],
  skipped: ['#6b7280', 'Chưa có model'],
}

// Cổng vào / ra của từng bước (chỉ để hiển thị)
const PORTS = {
  input: [[], ['Image']],
  seg: [['Image In'], ['Teeth']],
  wisdom: [['Teeth'], ['Wisdom']],
  caries: [['Image In', 'Teeth'], ['Caries']],
  perio: [['Image In', 'Teeth'], ['Bone line']],
  cej: [['Image In', 'Teeth'], ['CEJ']],
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

export default function PipelineGraph({ steps }) {
  const nodes = steps.map((s, i) => ({ id: s.key, type: 'step', position: { x: 0, y: i * 96 }, data: s, draggable: false }))
  const edges = steps.slice(1).map((s, i) => ({
    id: `e${i}`, source: steps[i].key, target: s.key, type: 'smoothstep',
    animated: s.status === 'running',
    style: { stroke: steps[i].status === 'done' ? '#2dd4bf' : '#4b5563', strokeWidth: 1.5 },
  }))
  return (
    <ReactFlow key={steps.length} nodes={nodes} edges={edges} nodeTypes={nodeTypes} colorMode="dark"
      fitView fitViewOptions={{ maxZoom: 1, padding: 0.1 }}
      nodesConnectable={false} elementsSelectable={false} proOptions={{ hideAttribution: true }}>
      <Background gap={16} color="#2a2a2a" />
    </ReactFlow>
  )
}
