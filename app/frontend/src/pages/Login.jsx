import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { api, session } from '../api'

export default function Login() {
  const nav = useNavigate()
  const [form, setForm] = useState({ username: '', password: '' })
  const [error, setError] = useState('')
  const set = (k) => (e) => setForm({ ...form, [k]: e.target.value })

  const submit = async (e) => {
    e.preventDefault()
    setError('')
    try {
      const r = await api('/auth/login', { method: 'POST', json: form })
      session.save(r.token, r.username)
      nav('/')
    } catch (err) {
      setError(err.message)
    }
  }

  return (
    <div className="login-wrap">
      <form className="login card" onSubmit={submit}>
        <h2>🦷 Tooth Diagnosis</h2>
        <label>Tên đăng nhập<input value={form.username} onChange={set('username')} autoFocus /></label>
        <label>Mật khẩu<input type="password" value={form.password} onChange={set('password')} /></label>
        {error && <div className="error">{error}</div>}
        <button className="btn primary">Đăng nhập</button>
      </form>
    </div>
  )
}
