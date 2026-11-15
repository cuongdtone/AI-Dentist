export const session = {
  get token() { return localStorage.getItem('token') },
  get user() { return localStorage.getItem('user') },
  save(token, user) { localStorage.setItem('token', token); localStorage.setItem('user', user) },
  clear() { localStorage.removeItem('token'); localStorage.removeItem('user') },
}

export async function api(path, { method = 'GET', json, form } = {}) {
  const headers = { Authorization: `Bearer ${session.token}` }
  if (json) headers['Content-Type'] = 'application/json'
  const r = await fetch(`/api${path}`, { method, headers, body: form ?? (json && JSON.stringify(json)) })
  if (r.status === 401 && path !== '/auth/login') {
    session.clear()
    location.href = '/login'
  }
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || r.statusText)
  return data
}
