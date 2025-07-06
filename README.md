# Flask API Docker + Cloudflare Tunnel

A minimal Flask API containerized with Docker and securely published to the internet via Cloudflare Tunnel.

---

## 🚀 Quick Start

### 1️⃣ Prerequisites
- Docker Desktop installed
- Cloudflare account
- A domain added to Cloudflare and nameservers updated
- `cloudflared` installed

---

### 2️⃣ Build and Run the Flask API

```powershell
docker-compose up -d
```

✅ Test locally:
[http://localhost:5000](http://localhost:5000)

---

### 3️⃣ Cloudflare Tunnel Setup

**Authenticate:**
```powershell
cloudflared tunnel login
```

**Create a tunnel:**
```powershell
cloudflared tunnel create flask-tunnel
```

**Create config.yml:**
```
C:\Users\YourUsername\.cloudflared\config.yml
```
Content:
```yaml
tunnel: flask-tunnel
credentials-file: C:\Users\YourUsername\.cloudflared\<tunnel-UUID>.json

ingress:
  - hostname: api.yourdomain.com
    service: http://localhost:5000
  - service: http_status:404
```

**Create DNS route:**
```powershell
cloudflared tunnel route dns flask-tunnel api.yourdomain.com
```

**Run the tunnel:**
```powershell
cloudflared tunnel run flask-tunnel
```

✅ Your API is live at:
```
https://api.yourdomain.com
```

---

## 📂 Files

- `app.py` – Flask app
- `requirements.txt` – Python dependencies
- `Dockerfile` – Build instructions
- `docker-compose.yml` – Docker Compose configuration

---

## 🛠️ Useful Commands

**Start containers:**
```powershell
docker-compose up -d
```

**Stop containers:**
```powershell
docker-compose down
```

**Rebuild image:**
```powershell
docker-compose build
```

---

## ⚠️ Notes
- Make sure your Cloudflare domain is active (nameservers updated).
- You can install the tunnel as a Windows Service:
  ```powershell
  cloudflared service install
  net start CloudflareTunnel
  ```
- Use environment variables for sensitive settings.

---

## 📄 License

MIT License

---
