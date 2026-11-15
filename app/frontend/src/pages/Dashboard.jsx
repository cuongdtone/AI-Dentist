import { useEffect, useState } from 'react'
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api } from '../api'

const PERIODS = [['day', 'Ngày', '30 ngày qua'], ['week', 'Tuần', '12 tuần qua'], ['month', 'Tháng', '12 tháng qua'], ['year', 'Năm', '5 năm qua']]
// Loại phát hiện: thứ tự & màu cố định (palette tối đã kiểm định, slot 1-4)
const FINDINGS = [
  ['wisdom', 'Răng khôn', '#3987e5'],
  ['caries', 'Sâu răng', '#d95926'],
  ['perio', 'Bệnh nha chu', '#199e70'],
  ['cej', 'Bất thường CEJ', '#c98500'],
]
const CARDS = [['images', 'Ảnh đã phân tích'], ['teeth', 'Răng phát hiện'], ...FINDINGS]

const axis = { stroke: 'var(--muted)', fontSize: 12, tickLine: false }
const tooltip = { contentStyle: { background: 'var(--panel-2)', border: '1px solid var(--border)', color: 'var(--text)' }, cursor: { fill: 'rgba(255,255,255,.05)' } }

export default function Dashboard() {
  const [period, setPeriod] = useState('day')
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    api(`/stats?period=${period}`).then(setData).catch((e) => setError(e.message))
  }, [period])

  const label = PERIODS.find((p) => p[0] === period)[2]

  return (
    <div className="page">
      <div className="row">
        <h2>Thống kê</h2>
        <span className="spacer" />
        <div className="segmented">
          {PERIODS.map(([k, name]) => (
            <button key={k} className={period === k ? 'active' : ''} onClick={() => setPeriod(k)}>{name}</button>
          ))}
        </div>
      </div>
      {error && <div className="error">{error}</div>}
      {data && (
        <>
          <div className="cards">
            {CARDS.map(([k, name, color]) => (
              <div key={k} className="card stat">
                <div className="muted">{color && <i className="dot" style={{ background: color }} />}{name}</div>
                <div className="value">{data.range[k].toLocaleString()}</div>
                <div className="muted small">{label} · tổng {data.all_time[k].toLocaleString()}</div>
              </div>
            ))}
          </div>
          <div className="charts">
            <div className="card">
              <h3>Số ảnh đã phân tích</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data.series}>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="bucket" {...axis} />
                  <YAxis allowDecimals={false} {...axis} axisLine={false} />
                  <Tooltip {...tooltip} />
                  <Bar dataKey="images" name="Ảnh" fill="#3987e5" radius={[4, 4, 0, 0]} maxBarSize={28} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card">
              <h3>Phát hiện theo loại</h3>
              <ResponsiveContainer width="100%" height={280}>
                <BarChart data={data.series}>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="bucket" {...axis} />
                  <YAxis allowDecimals={false} {...axis} axisLine={false} />
                  <Tooltip {...tooltip} />
                  <Legend wrapperStyle={{ fontSize: 12 }} formatter={(v) => <span style={{ color: 'var(--text)' }}>{v}</span>} />
                  {FINDINGS.map(([k, name, color], i) => (
                    <Bar key={k} dataKey={k} name={name} stackId="f" fill={color} stroke="var(--panel)" strokeWidth={2}
                      radius={i === FINDINGS.length - 1 ? [4, 4, 0, 0] : 0} maxBarSize={28} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
