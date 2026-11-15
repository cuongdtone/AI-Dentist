/* Panel kết quả: tóm tắt bằng lời + sơ đồ răng FDI (hover đồng bộ với viewer). */
import { toothColor } from './ImageViewer'

const UPPER = ['18', '17', '16', '15', '14', '13', '12', '11', '21', '22', '23', '24', '25', '26', '27', '28']
const LOWER = ['48', '47', '46', '45', '44', '43', '42', '41', '31', '32', '33', '34', '35', '36', '37', '38']
const QUAD = { 1: 'hàm trên phải', 2: 'hàm trên trái', 3: 'hàm dưới trái', 4: 'hàm dưới phải' }
const WISDOM = ['18', '28', '38', '48']
const JOB_TEXT = { running: 'Đang chạy…', done: 'Hoàn thành', error: 'Lỗi', cancelled: 'Đã huỷ' }
const OTHER = [['caries', 'Sâu răng'], ['perio', 'Đường nha chu'], ['cej', 'Mức CEJ']]

const where = (fdi) => `${fdi} (${QUAD[fdi[0]]})`

function wisdomText(found) {
  if (!found.length) return 'Không thấy răng khôn trên phim.'
  if (found.length === 4) return 'Đủ 4 răng khôn: 18, 28, 38, 48.'
  return `Răng khôn ${found.join(', ')}: ${found.map(where).join(', ')}.`
}

function Chart({ row, byFdi, hover, onHover }) {
  return (
    <div className="chart-row">
      {row.map((f) => {
        const t = byFdi[f]
        const cls = ['tooth', t ? 'present' : 'absent', WISDOM.includes(f) && t ? 'wisdom' : '', t && t.id === hover ? 'active' : ''].join(' ')
        return (
          <div key={f} className={cls} style={t ? { '--c': toothColor(f) } : undefined}
            onMouseEnter={() => t && onHover(t.id)} onMouseLeave={() => onHover(null)} title={t ? `Răng ${where(f)} · ${(t.conf * 100).toFixed(0)}%` : `${f}: không thấy`}>
            {f}
          </div>
        )
      })}
    </div>
  )
}

export default function ResultPanel({ job, busy, teeth, hover, onHover, msg }) {
  const step = (k) => job?.steps.find((s) => s.key === k)
  const byFdi = Object.fromEntries(teeth.map((t) => [t.fdi, t]))
  const wisdom = WISDOM.filter((f) => byFdi[f])
  const missing = [...UPPER, ...LOWER].filter((f) => !byFdi[f]).sort()
  const hasSeg = step('seg')?.status === 'done'

  return (
    <div className="results">
      <div className="res-head">
        <h3>Kết quả</h3>
        <span className="muted small">{busy ? 'Đang tải lên…' : JOB_TEXT[job?.status] ?? 'Chưa chạy'}</span>
      </div>
      {!hasSeg ? (
        <p className="muted small">Tải ảnh lên để bắt đầu phân tích.</p>
      ) : (
        <>
          <div className="res-item">
            <div className="res-title">Số răng <b>{teeth.length} / 32</b></div>
            <p>{missing.length ? `Không thấy trên phim: ${missing.join(', ')} (mất răng hoặc model bỏ sót).` : 'Đủ 32 răng.'}</p>
          </div>
          <div className="res-item">
            <div className="res-title">
              Răng khôn
              <span className={`badge ${wisdom.length ? 'yes' : ''}`}>{wisdom.length ? `Có · ${wisdom.length}` : 'Không'}</span>
            </div>
            <p>{wisdomText(wisdom)}</p>
          </div>
          <div className="dental-chart">
            <div className="muted small">Hàm trên</div>
            <Chart row={UPPER} byFdi={byFdi} hover={hover} onHover={onHover} />
            <Chart row={LOWER} byFdi={byFdi} hover={hover} onHover={onHover} />
            <div className="muted small">Hàm dưới</div>
          </div>
          {OTHER.map(([k, name]) => {
            const s = step(k)?.status
            const n = job?.result.findings[k]?.length
            return (
              <div key={k} className="res-item">
                <div className="res-title">{name}<span className="badge">{s === 'done' ? (n ? `Có · ${n}` : 'Không') : s === 'skipped' ? 'Chưa có model' : 'Chờ'}</span></div>
              </div>
            )
          })}
        </>
      )}
      {msg && <div className="msg">{msg}</div>}
    </div>
  )
}
