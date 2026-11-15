import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter, Navigate, NavLink, Outlet, Route, Routes, useLocation, useNavigate } from 'react-router-dom'
import { session } from './api'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import Diagnosis from './pages/Diagnosis'
import './styles.css'

function Layout() {
  const nav = useNavigate()
  const onDiag = useLocation().pathname === '/diagnosis'
  const [diagMounted, setDiagMounted] = useState(onDiag) // giữ trang Chẩn đoán sống sau lần mở đầu
  useEffect(() => { if (onDiag) setDiagMounted(true) }, [onDiag])
  if (!session.token) return <Navigate to="/login" replace />
  const logout = () => { session.clear(); nav('/login') }
  return (
    <div className="app">
      <header className="topbar">
        <span className="brand">🦷 Tooth Diagnosis</span>
        <nav>
          <NavLink to="/" end>📊 Thống kê</NavLink>
          <NavLink to="/diagnosis">🩺 Chẩn đoán</NavLink>
        </nav>
        <span className="spacer" />
        <span className="user">👤 {session.user}</span>
        <button className="btn ghost" onClick={logout}>Đăng xuất</button>
      </header>
      <main>
        {!onDiag && <Outlet />}
        {diagMounted && <div hidden={!onDiag}><Diagnosis active={onDiag} /></div>}
      </main>
    </div>
  )
}

ReactDOM.createRoot(document.getElementById('root')).render(
  <BrowserRouter>
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/diagnosis" element={null} />
      </Route>
      <Route path="*" element={<Navigate to="/" />} />
    </Routes>
  </BrowserRouter>,
)
