/* Panel kết quả: tóm tắt bằng lời + sơ đồ răng FDI (hover đồng bộ với viewer). */
import { STAGE_COLOR, toothColor } from './ImageViewer'

const UPPER = ['18', '17', '16', '15', '14', '13', '12', '11', '21', '22', '23', '24', '25', '26', '27', '28']
const LOWER = ['48', '47', '46', '45', '44', '43', '42', '41', '31', '32', '33', '34', '35', '36', '37', '38']
const QUAD = { 1: 'hàm trên phải', 2: 'hàm trên trái', 3: 'hàm dưới trái', 4: 'hàm dưới phải' }
const WISDOM = ['18', '28', '38', '48']
const JOB_TEXT = { running: 'Đang chạy…', done: 'Hoàn thành', error: 'Lỗi', cancelled: 'Đã huỷ' }
const STAGE_VI = { 1: 'I', 2: 'II', 3: 'III/IV' }

const where = (fdi) => `${fdi} (${QUAD[fdi[0]]})`

function wisdomText(found) {
  if (!found.length) return 'Không thấy răng khôn trên phim.'
  if (found.length === 4) return 'Đủ 4 răng khôn: 18, 28, 38, 48.'
  return `Răng khôn ${found.join(', ')}: ${found.map(where).join(', ')}.`
}

const WINTER_VI = { vertical: 'Thẳng', mesioangular: 'Nghiêng gần', horizontal: 'Nằm ngang', distoangular: 'Nghiêng xa', other: 'Khác' }
const DEPTH_VI = { A: 'A · ngang/trên mặt nhai R7', B: 'B · giữa mặt nhai và CEJ R7', C: 'C · dưới CEJ R7' }
const FLAG_VI = { thieu_m2: 'thiếu răng 7, góc so với trục đứng', truc_m3_khong_chac: 'trục răng khôn không chắc (mầm răng?)', truc_m2_khong_chac: 'trục răng 7 không chắc', mask_m3_qua_nho: 'mask quá nhỏ' }

/* Bảng phân tích từng răng khôn (Winter, độ sâu, ngầm/lệch); hover đồng bộ với viewer. */
function WisdomList({ items, hover, onHover }) {
  return (
    <div className="wisdom-list">
      {items.map((w) => (
        <div key={w.fdi} className={`wisdom-row ${w.impacted ? 'bad' : 'ok'} ${w.tooth_id === hover ? 'active' : ''}`}
          onMouseEnter={() => onHover(w.tooth_id)} onMouseLeave={() => onHover(null)}>
          <div className="wisdom-head">
            <b>{w.fdi}</b>
            <span className={`badge ${w.impacted ? 'yes' : ''}`}>{w.winter ? (w.impacted ? 'Ngầm/lệch' : 'Bình thường') : '—'}</span>
          </div>
          {w.winter ? (
            <div className="small">
              {WINTER_VI[w.winter]} {w.winter_angle > 0 ? '+' : ''}{w.winter_angle}°
              {w.depth && <> · độ sâu {DEPTH_VI[w.depth]}</>}
            </div>
          ) : null}
          {w.flags?.length > 0 && <div className="muted small">⚠ {w.flags.map((f) => FLAG_VI[f] ?? f).join('; ')}</div>}
        </div>
      ))}
    </div>
  )
}

/* Danh sách răng có vùng sâu (gộp theo răng); hover đồng bộ với viewer. */
function CariesList({ items, hover, onHover }) {
  const byTooth = {}
  for (const c of items) (byTooth[c.tooth_id] ??= { fdi: c.fdi, tooth_id: c.tooth_id, list: [] }).list.push(c)
  return (
    <div className="caries-list">
      {Object.values(byTooth).map((t) => {
        const frac = t.list.reduce((a, c) => a + c.tooth_frac, 0)
        return (
          <div key={t.tooth_id} className={`caries-row ${t.tooth_id === hover ? 'active' : ''}`}
            onMouseEnter={() => onHover(t.tooth_id)} onMouseLeave={() => onHover(null)}>
            <b>{t.fdi}</b>
            <span className="small">{t.list.length > 1 ? `${t.list.length} vùng · ` : ''}{(frac * 100).toFixed(0)}% diện tích răng</span>
            <span className="muted small" title="Độ tin cậy từng vùng">{t.list.map((c) => `${(c.conf * 100).toFixed(0)}%`).join(' · ')}</span>
          </div>
        )
      })}
    </div>
  )
}

/* Răng tiêu xương giai đoạn ≥ II (sắp theo % giảm dần); hover đồng bộ với viewer. */
function PerioList({ items, hover, onHover }) {
  const pct = (s) => (s ? `${Math.round(s.rbl * 100)}%` : '—')
  return (
    <div className="caries-list">
      {items.map((m) => (
        <div key={m.tooth_id ?? m.fdi} className={`caries-row perio-row ${m.tooth_id === hover ? 'active' : ''}`} style={{ borderLeftColor: STAGE_COLOR[m.stage] }}
          onMouseEnter={() => onHover(m.tooth_id)} onMouseLeave={() => onHover(null)}>
          <b>{m.fdi}</b>
          <span className="small">gần {pct(m.mesial)} · xa {pct(m.distal)}{m.angular ? ' · dạng góc' : ''}</span>
          <span className="small" style={{ color: STAGE_COLOR[m.stage] }}>GĐ {STAGE_VI[m.stage]}</span>
        </div>
      ))}
    </div>
  )
}

const stepBadge = (st, done, text) => (done ? text : st?.status === 'skipped' ? (st.note ?? 'Bỏ qua')
  : st?.status === 'disabled' ? 'Tắt' : st?.status === 'error' ? 'Lỗi' : 'Chờ')

function Chart({ row, byFdi, hover, onHover, cariesIds, impactedIds, stages }) {
  return (
    <div className="chart-row">
      {row.map((f) => {
        const t = byFdi[f]
        const car = t && cariesIds.has(t.id), imp = t && impactedIds.has(t.id)
        const stg = t && stages[t.id]
        const cls = ['tooth', t ? 'present' : 'absent', WISDOM.includes(f) && t ? 'wisdom' : '', car ? 'caries' : '', imp ? 'impacted' : '',
          stg >= 2 ? `perio${stg}` : '', t && t.id === hover ? 'active' : ''].join(' ')
        const tip = t ? `Răng ${where(f)} · ${(t.conf * 100).toFixed(0)}%${imp ? ' · ngầm/lệch' : ''}${car ? ' · sâu răng' : ''}${stg >= 2 ? ` · tiêu xương GĐ ${STAGE_VI[stg]}` : ''}` : `${f}: không thấy`
        return (
          <div key={f} className={cls} style={t ? { '--c': toothColor(f) } : undefined}
            onMouseEnter={() => t && onHover(t.id)} onMouseLeave={() => onHover(null)} title={tip}>
            {f}
          </div>
        )
      })}
    </div>
  )
}

export default function ResultPanel({ job, busy, teeth, caries, hover, onHover, msg }) {
  const step = (k) => job?.steps.find((s) => s.key === k)
  const byFdi = Object.fromEntries(teeth.map((t) => [t.fdi, t]))
  const wisdom = WISDOM.filter((f) => byFdi[f])
  const wAnalysis = job?.result.findings.wisdom
  const wDone = step('wisdom')?.status === 'done' && Array.isArray(wAnalysis)
  const nImpacted = wDone ? wAnalysis.filter((w) => w.impacted).length : 0
  const missing = [...UPPER, ...LOWER].filter((f) => !byFdi[f]).sort()
  const hasSeg = step('seg')?.status === 'done'
  const cSt = step('caries')
  const cAnalysis = caries // bản đang sửa (doc), không phải kết quả gốc của job
  const cDone = cSt?.status === 'done' && Array.isArray(cAnalysis)
  const cariesIds = new Set(cDone ? cAnalysis.map((c) => c.tooth_id) : [])
  const impactedIds = new Set(wDone ? wAnalysis.filter((w) => w.impacted).map((w) => w.tooth_id) : [])
  const jSt = step('cej'), pSt = step('perio')
  const cejList = job?.result.findings.cej, perio = job?.result.findings.perio
  const jDone = jSt?.status === 'done' && Array.isArray(cejList)
  const pDone = pSt?.status === 'done' && Array.isArray(perio)
  const stages = Object.fromEntries(pDone ? perio.map((m) => [m.tooth_id, m.stage]) : [])
  const perioBad = pDone ? perio.filter((m) => m.stage >= 2).sort((a, b) => b.rbl - a.rbl) : []
  const maxStage = pDone && perio.length ? Math.max(...perio.map((m) => m.stage)) : 0
  const nMeasured = jDone ? cejList.filter((c) => c.mesial || c.distal).length : 0
  const cejWisdom = wDone && wAnalysis.some((w) => w.cej_source === 'measured')

  return (
    <div className="results">
      <div className="res-head">
        <h3>Kết quả</h3>
        <span className="muted small">{busy ? 'Đang tải lên…' : JOB_TEXT[job?.status] ?? 'Chưa chạy'}</span>
      </div>
      {!hasSeg ? (
        <p className="muted small">
          {!job ? 'Tải ảnh lên để bắt đầu phân tích.'
            : step('seg')?.status === 'disabled' ? 'Bước Phân đoạn răng đang tắt nên không có kết quả. Chuột phải vào node → Enable.'
              : 'Đang chờ kết quả phân đoạn răng…'}
        </p>
      ) : (
        <>
          <div className="res-item">
            <div className="res-title">Số răng <b>{teeth.length} / 32</b></div>
            <p>{missing.length ? `Không thấy trên phim: ${missing.join(', ')} (mất răng hoặc model bỏ sót).` : 'Đủ 32 răng.'}</p>
          </div>
          <div className="dental-chart">
            <div className="muted small">Hàm trên</div>
            <Chart row={UPPER} byFdi={byFdi} hover={hover} onHover={onHover} cariesIds={cariesIds} impactedIds={impactedIds} stages={stages} />
            <Chart row={LOWER} byFdi={byFdi} hover={hover} onHover={onHover} cariesIds={cariesIds} impactedIds={impactedIds} stages={stages} />
            <div className="muted small">Hàm dưới</div>
            {(impactedIds.size > 0 || cariesIds.size > 0 || perioBad.length > 0) && (
              <div className="chart-legend muted small">
                {impactedIds.size > 0 && <span><i className="imp" />Răng khôn ngầm/lệch</span>}
                {cariesIds.size > 0 && <span><i className="car" />Sâu răng</span>}
                {perioBad.length > 0 && <span><i className="per" />Tiêu xương II · <i className="per3" />III/IV</span>}
              </div>
            )}
          </div>
          <div className="res-item">
            <div className="res-title">
              Răng khôn
              <span className={`badge ${nImpacted ? 'yes' : ''}`}>
                {!wisdom.length ? 'Không' : wDone ? (nImpacted ? `Ngầm/lệch · ${nImpacted}/${wisdom.length}` : `Có · ${wisdom.length}`) : `Có · ${wisdom.length}`}
              </span>
            </div>
            <p>{wisdomText(wisdom)}</p>
            {wDone && wAnalysis.length > 0 && <WisdomList items={wAnalysis} hover={hover} onHover={onHover} />}
            {wDone && wAnalysis.length > 0 && (
              <p className="muted small">Góc Winter: trục răng khôn so với trục răng 7 (dương = nghiêng về phía gần). Kết quả tự động, cần bác sĩ xác nhận.</p>
            )}
          </div>
          <div className="res-item">
            <div className="res-title">
              Sâu răng
              <span className={`badge ${cariesIds.size ? 'yes' : ''}`}>
                {cDone ? (cariesIds.size ? `${cariesIds.size} răng` : 'Không') : cSt?.status === 'skipped' ? (cSt.note ?? 'Bỏ qua')
                  : cSt?.status === 'disabled' ? 'Tắt' : cSt?.status === 'error' ? 'Lỗi' : 'Chờ'}
              </span>
            </div>
            {cDone && (cAnalysis.length > 0
              ? <><CariesList items={cAnalysis} hover={hover} onHover={onHover} />
                <p className="muted small">Viền đỏ trên ảnh (bật Mask để tô), % là độ tin cậy. Phát hiện tự động trên từng răng, cần bác sĩ xác nhận.</p></>
              : <p className="muted small">Không phát hiện vùng sâu.</p>)}
          </div>
          <div className="res-item">
            <div className="res-title">
              Mức CEJ
              <span className="badge">{stepBadge(jSt, jDone, nMeasured ? `${nMeasured} răng` : 'Không')}</span>
            </div>
            {jDone && (
              <p className="muted small">
                Xác định được CEJ mặt gần/xa trên {nMeasured}/{teeth.length} răng (chấm tím; viền tím đứt nét = vùng thân răng).
                {cejWisdom && ' Độ sâu răng khôn (B/C) dùng CEJ thật của răng 7.'}
              </p>
            )}
          </div>
          <div className="res-item">
            <div className="res-title">
              Tiêu xương nha chu
              <span className={`badge ${maxStage >= 2 ? 'yes' : ''}`}>
                {stepBadge(pSt, pDone, !perio?.length ? 'Không đo được' : maxStage >= 2 ? `GĐ ${STAGE_VI[maxStage]} · ${perioBad.length} răng` : 'GĐ I')}
              </span>
            </div>
            {pDone && (perioBad.length > 0
              ? <PerioList items={perioBad} hover={hover} onHover={onHover} />
              : perio.length > 0 && <p className="muted small">Tất cả {perio.length} răng đo được tiêu xương &lt; 15% (giai đoạn I).</p>)}
            {pDone && perio.length > 0 && (
              <p className="muted small">% tiêu xương = (CEJ → đỉnh xương − khoảng sinh lý) / (CEJ → chóp), đo ở mặt gần và xa; giai đoạn theo AAP/EFP 2017.
                Ước lượng trên phim toàn cảnh để sàng lọc, cần bác sĩ xác nhận.</p>
            )}
          </div>
        </>
      )}
      {msg && <div className="msg">{msg}</div>}
    </div>
  )
}
