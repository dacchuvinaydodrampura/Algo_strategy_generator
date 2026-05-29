import os
import asyncio
import datetime
import shutil
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from app.config import load_settings, Settings
from app.jobs.daily_job import run_daily_job
from app.utils.log_setup import get_logger

logger = get_logger(__name__)

app = FastAPI(title="Market Microstructure Research Engine Dashboard")

# Global task state to track pipeline runs
pipeline_status = {
    "status": "idle", # idle | running | completed | failed
    "last_run": None,
    "last_archive": None,
    "error": None,
    "log_output": []
}

class ConfigUpdate(BaseModel):
    telegram_bot_token: str
    telegram_channel_id: int
    log_level: str
    log_format: str
    mongodb_uri: Optional[str] = ""

def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent

def write_dotenv(config: ConfigUpdate):
    root = get_project_root()
    env_path = root / ".env"
    lines = [
        "# Telegram Secrets",
        f"TELEGRAM_BOT_TOKEN={config.telegram_bot_token}",
        f"TELEGRAM_CHANNEL_ID={config.telegram_channel_id}",
        "",
        "# MongoDB Connection",
        f"MONGODB_URI={config.mongodb_uri or ''}",
        "",
        "# Logging",
        f"LOG_LEVEL={config.log_level}",
        f"LOG_FORMAT={config.log_format}",
        "",
        "# Storage",
        f"DATA_ROOT={root.as_posix()}/data"
    ]
    env_path.write_text("\n".join(lines), encoding="utf-8")
    
    # Update active environment variables
    os.environ["TELEGRAM_BOT_TOKEN"] = config.telegram_bot_token
    os.environ["TELEGRAM_CHANNEL_ID"] = str(config.telegram_channel_id)
    os.environ["MONGODB_URI"] = config.mongodb_uri or ""
    os.environ["LOG_LEVEL"] = config.log_level
    os.environ["LOG_FORMAT"] = config.log_format

def execute_pipeline(archive_name: str, session_date_str: Optional[str] = None):
    global pipeline_status
    pipeline_status["status"] = "running"
    pipeline_status["last_archive"] = archive_name
    pipeline_status["error"] = None
    pipeline_status["log_output"] = ["Pipeline started..."]
    
    try:
        settings = load_settings()
        archive_path = settings.archives_path / archive_name
        
        session_date = None
        if session_date_str:
            session_date = datetime.date.fromisoformat(session_date_str)
            
        pipeline_status["log_output"].append(f"Processing archive: {archive_path.name}")
        pipeline_status["log_output"].append(f"Inferred/Overridden Date: {session_date or 'Auto-parsed from filename'}")
        
        # Execute the pipeline orchestrator
        success = run_daily_job(
            settings=settings,
            archive_path=archive_path,
            session_date=session_date
        )
        
        if success:
            pipeline_status["status"] = "completed"
            pipeline_status["log_output"].append("Pipeline run completed successfully! PDF Report and JSON summary generated.")
        else:
            pipeline_status["status"] = "failed"
            pipeline_status["error"] = "Daily job orchestrator returned False. Check logs for validation or backtest errors."
            pipeline_status["log_output"].append("Pipeline run failed.")
            
    except Exception as e:
        pipeline_status["status"] = "failed"
        pipeline_status["error"] = str(e)
        pipeline_status["log_output"].append(f"Pipeline crashed with exception: {e}")
    finally:
        pipeline_status["last_run"] = datetime.datetime.now().isoformat()

@app.get("/api/config")
def get_config():
    settings = load_settings()
    return {
        "telegram_bot_token": settings.telegram_bot_token,
        "telegram_channel_id": settings.telegram_channel_id,
        "mongodb_uri": settings.mongodb_uri,
        "log_level": settings.log_level,
        "log_format": settings.log_format,
        "data_root": settings.data_root
    }

@app.post("/api/config")
def update_config(config: ConfigUpdate):
    try:
        write_dotenv(config)
        return {"status": "success", "message": "Environment configuration updated successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update config: {e}")

@app.get("/api/reports")
def list_reports():
    settings = load_settings()
    reports_dir = settings.reports_path
    if not reports_dir.exists():
        return []
    
    files = []
    for f in reports_dir.glob("*_research_report.pdf"):
        stat = f.stat()
        date_str = f.name.split("_")[0]
        summary_file = reports_dir / f"{date_str}_summary.json"
        
        summary_data = None
        if summary_file.exists():
            try:
                import json
                summary_data = json.loads(summary_file.read_text())
            except Exception:
                pass
                
        files.append({
            "filename": f.name,
            "size_bytes": stat.st_size,
            "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "date": date_str,
            "summary": summary_data
        })
    return sorted(files, key=lambda x: x["created_at"], reverse=True)

@app.get("/api/reports/{filename}")
def download_report(filename: str):
    settings = load_settings()
    file_path = settings.reports_path / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="PDF report not found")
    return FileResponse(file_path, media_type="application/pdf", filename=filename)

@app.get("/api/archives")
def list_archives():
    settings = load_settings()
    archives_dir = settings.archives_path
    if not archives_dir.exists():
        return []
    
    files = []
    for f in archives_dir.glob("*.tar.gz"):
        stat = f.stat()
        files.append({
            "filename": f.name,
            "size_bytes": stat.st_size,
            "created_at": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    return sorted(files, key=lambda x: x["filename"], reverse=True)

@app.post("/api/upload-archive")
async def upload_archive(file: UploadFile = File(...)):
    if not file.filename.endswith(".tar.gz"):
        raise HTTPException(status_code=400, detail="Only .tar.gz archives are supported")
        
    settings = load_settings()
    settings.ensure_directories()
    dest_path = settings.archives_path / file.filename
    
    try:
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"status": "success", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

@app.post("/api/run-job")
def trigger_job(archive_name: str, date: Optional[str] = None, background_tasks: BackgroundTasks = BackgroundTasks()):
    global pipeline_status
    if pipeline_status["status"] == "running":
        raise HTTPException(status_code=400, detail="Pipeline job is already running")
        
    settings = load_settings()
    archive_path = settings.archives_path / archive_name
    if not archive_path.exists():
        raise HTTPException(status_code=404, detail=f"Archive file '{archive_name}' not found in data/archives")
        
    background_tasks.add_task(execute_pipeline, archive_name, date)
    return {"status": "started", "message": "Pipeline run triggered in background."}

@app.get("/api/status")
def get_status():
    return pipeline_status

@app.get("/api/stats")
def get_db_stats():
    settings = load_settings()
    
    if settings.mongodb_uri:
        try:
            import pymongo
            client = pymongo.MongoClient(settings.mongodb_uri)
            try:
                db = client.get_default_database()
                if db is None or db.name == "test":
                    db = client["market_research"]
            except Exception:
                db = client["market_research"]
            
            stats = db.command("dbStats")
            collections = db.list_collection_names()
            
            # Count ticks per symbol from manifests
            symbols_count = {}
            cursor = db["archive_manifests"].find()
            for row in cursor:
                total_ticks = row.get("total_ticks", {})
                for sym, cnt in total_ticks.items():
                    symbols_count[sym] = symbols_count.get(sym, 0) + cnt
            
            client.close()
            
            limit_bytes = 512 * 1024 * 1024
            current_bytes = stats.get("storageSize", stats.get("dataSize", 0))
            
            return {
                "db_exists": True,
                "db_type": "MongoDB",
                "size_bytes": current_bytes,
                "limit_bytes": limit_bytes,
                "utilization_pct": round((current_bytes / limit_bytes) * 100, 2),
                "collections": collections,
                "symbol_ticks": symbols_count
            }
        except Exception as e:
            return {"db_exists": False, "db_type": "MongoDB", "error": str(e)}
            
    db_path = settings.db_path
    if not db_path.exists():
        return {"db_exists": False, "size_bytes": 0, "tables": [], "db_type": "DuckDB"}
        
    try:
        import duckdb
        conn = duckdb.connect(str(db_path))
        tables_df = conn.execute("SHOW TABLES").fetchall()
        tables = [t[0] for t in tables_df]
        
        symbols_count = {}
        if "ticks" in tables:
            sym_df = conn.execute("SELECT s, count(*) FROM ticks GROUP BY s").fetchall()
            symbols_count = {r[0]: r[1] for r in sym_df}
            
        conn.close()
        return {
            "db_exists": True,
            "db_type": "DuckDB",
            "size_bytes": db_path.stat().st_size,
            "tables": tables,
            "symbol_ticks": symbols_count
        }
    except Exception as e:
        return {"db_exists": True, "db_type": "DuckDB", "size_bytes": db_path.stat().st_size, "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Microstructure Research Engine Dashboard</title>
    <!-- Google Font -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    
    <style>
        :root {
            --bg-base: #0b0f19;
            --bg-surface: rgba(17, 24, 39, 0.7);
            --bg-card: rgba(30, 41, 59, 0.5);
            --border-color: rgba(255, 255, 255, 0.08);
            
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            
            --accent: #38bdf8;
            --accent-glow: rgba(56, 189, 248, 0.15);
            --success: #22c55e;
            --warning: #eab308;
            --danger: #ef4444;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Outfit', sans-serif;
            background-color: var(--bg-base);
            background-image: radial-gradient(circle at 10% 20%, rgba(56, 189, 248, 0.05) 0%, transparent 40%),
                              radial-gradient(circle at 90% 80%, rgba(13, 27, 42, 0.2) 0%, transparent 50%);
            background-attachment: fixed;
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.5;
            padding: 2rem;
        }

        .container {
            max-width: 1400px;
            margin: 0 auto;
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 2rem;
        }

        header {
            grid-column: 1 / -1;
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--border-color);
            margin-bottom: 1rem;
        }

        .logo-section h1 {
            font-size: 1.75rem;
            font-weight: 700;
            background: linear-gradient(135deg, #f8fafc 0%, #38bdf8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.02em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }
        
        .logo-section p {
            color: var(--text-secondary);
            font-size: 0.875rem;
            margin-top: 0.25rem;
        }

        .status-badge-container {
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }

        .badge {
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
        }

        .badge-idle { background: rgba(148, 163, 184, 0.15); color: #cbd5e1; }
        .badge-running { background: rgba(56, 189, 248, 0.15); color: #38bdf8; animation: pulse 2s infinite; }
        .badge-completed { background: rgba(34, 197, 94, 0.15); color: #4ade80; }
        .badge-failed { background: rgba(239, 68, 68, 0.15); color: #f87171; }

        @keyframes pulse {
            0% { opacity: 0.6; }
            50% { opacity: 1; }
            100% { opacity: 0.6; }
        }

        /* Sidebar Styling */
        .sidebar {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        .card {
            background: var(--bg-surface);
            backdrop-filter: blur(16px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        .card:hover {
            box-shadow: 0 4px 30px rgba(56, 189, 248, 0.04), 0 0 1px rgba(255, 255, 255, 0.1);
        }

        .card-title {
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            justify-content: space-between;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            padding-bottom: 0.5rem;
        }

        /* Form Controls */
        .form-group {
            margin-bottom: 1rem;
        }

        .form-group label {
            display: block;
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-bottom: 0.35rem;
            font-weight: 500;
        }

        .form-group input, .form-group select {
            width: 100%;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.9rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: var(--accent);
            box-shadow: 0 0 0 2px var(--accent-glow);
        }

        .btn {
            width: 100%;
            background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.7rem;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }

        .btn:hover {
            opacity: 0.95;
        }

        .btn:active {
            transform: scale(0.98);
        }
        
        .btn-secondary {
            background: rgba(255, 255, 255, 0.06);
            border: 1px solid var(--border-color);
            color: var(--text-primary);
        }
        
        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.1);
        }

        /* Drag and Drop Zone */
        .upload-zone {
            border: 2px dashed rgba(56, 189, 248, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            cursor: pointer;
            background: rgba(56, 189, 248, 0.02);
            transition: border-color 0.2s, background-color 0.2s;
        }

        .upload-zone.dragover {
            border-color: var(--accent);
            background: rgba(56, 189, 248, 0.08);
        }

        .upload-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }

        .upload-text {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        /* Main Workspace */
        .main-content {
            display: flex;
            flex-direction: column;
            gap: 2rem;
        }

        /* Log console */
        .console-container {
            background: rgba(10, 15, 30, 0.9);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            box-shadow: 0 4px 30px rgba(0,0,0,0.4);
            overflow: hidden;
            display: flex;
            flex-direction: column;
            height: 250px;
        }

        .console-header {
            background: rgba(255, 255, 255, 0.03);
            padding: 0.6rem 1.2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-secondary);
        }

        .console-body {
            padding: 1rem;
            overflow-y: auto;
            flex-grow: 1;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.85rem;
            color: #38bdf8;
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
        }
        
        .console-line-error {
            color: var(--danger);
        }
        
        .console-line-system {
            color: var(--text-muted);
        }

        /* Tables & Lists */
        .report-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.5rem;
        }

        .report-card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            position: relative;
            overflow: hidden;
        }
        
        .report-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--accent);
        }
        
        .report-card-accepted::before { background: var(--success); }
        .report-card-marginal::before { background: var(--warning); }
        .report-card-rejected::before { background: var(--danger); }

        .report-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }

        .report-title {
            font-size: 1.1rem;
            font-weight: 700;
            margin-bottom: 0.75rem;
        }

        .report-stats {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 8px;
            padding: 0.6rem;
            margin-bottom: 1rem;
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 0.5rem;
            font-size: 0.8rem;
        }

        .stat-item {
            display: flex;
            flex-direction: column;
        }

        .stat-label {
            color: var(--text-muted);
            font-size: 0.7rem;
        }

        .stat-value {
            font-weight: 600;
        }

        .report-actions {
            margin-top: auto;
        }

        /* Database Stats list */
        .db-stat-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
            border-bottom: 1px solid rgba(255,255,255,0.02);
            padding-bottom: 0.35rem;
        }
        
        .db-stat-row span:first-child {
            color: var(--text-secondary);
        }

        .db-stat-row span:last-child {
            font-weight: 600;
        }

        /* Archive listing styling */
        .archive-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255,255,255,0.02);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 0.6rem 0.8rem;
            margin-bottom: 0.5rem;
            font-size: 0.85rem;
        }
        
        .archive-item-info {
            display: flex;
            flex-direction: column;
        }
        
        .archive-item-size {
            font-size: 0.7rem;
            color: var(--text-muted);
        }

        .archive-actions {
            display: flex;
            gap: 0.35rem;
        }
        
        .archive-actions button {
            padding: 0.3rem 0.6rem;
            font-size: 0.75rem;
            border-radius: 6px;
        }

        @media (max-width: 1024px) {
            .container {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>

    <div class="container">
        <header>
            <div class="logo-section">
                <h1>⚡ NEURO FREQUENCY</h1>
                <p>Market Microstructure Research Engine v1.0</p>
            </div>
            <div class="status-badge-container">
                <span id="global-status-badge" class="badge badge-idle">Idle</span>
            </div>
        </header>

        <!-- Sidebar / Config / Control -->
        <div class="sidebar">
            <!-- App Controller / Upload -->
            <div class="card">
                <div class="card-title">Run Pipeline</div>
                <div class="form-group">
                    <label for="run-archive-select">Select Market Archive</label>
                    <select id="run-archive-select">
                        <option value="">No archives available</option>
                    </select>
                </div>
                <div class="form-group">
                    <label for="run-date-override">Date Override (Optional)</label>
                    <input type="date" id="run-date-override">
                </div>
                <button id="btn-run-pipeline" class="btn" style="margin-top: 0.5rem;">
                    🚀 Trigger Pipeline Run
                </button>
                
                <div style="margin: 1.5rem 0 1rem 0; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 1.5rem;">
                    <div class="upload-zone" id="upload-dropzone">
                        <div class="upload-icon">📦</div>
                        <div class="upload-text">Drag & drop or Click to upload archive<br><strong>.tar.gz</strong> format</div>
                        <input type="file" id="upload-file-input" style="display: none;" accept=".tar.gz">
                    </div>
                </div>
            </div>

            <!-- Settings / Config Override -->
            <div class="card">
                <div class="card-title">Environment Settings</div>
                <form id="config-form" onsubmit="saveConfig(event)">
                    <div class="form-group">
                        <label for="cfg-bot-token">Telegram Bot Token</label>
                        <input type="password" id="cfg-bot-token" required>
                    </div>
                    <div class="form-group">
                        <label for="cfg-channel-id">Telegram Channel ID</label>
                        <input type="text" id="cfg-channel-id" required>
                    </div>
                    <div class="form-group">
                        <label for="cfg-mongodb-uri">MongoDB URI (Optional)</label>
                        <input type="password" id="cfg-mongodb-uri" placeholder="mongodb+srv://...">
                    </div>
                    <div class="form-group">
                        <label for="cfg-log-level">Log Level</label>
                        <select id="cfg-log-level">
                            <option value="DEBUG">DEBUG</option>
                            <option value="INFO">INFO</option>
                            <option value="WARNING">WARNING</option>
                            <option value="ERROR">ERROR</option>
                        </select>
                    </div>
                    <div class="form-group">
                        <label for="cfg-log-format">Log Format</label>
                        <select id="cfg-log-format">
                            <option value="json">JSON</option>
                            <option value="console">Console Text</option>
                        </select>
                    </div>
                    <button type="submit" class="btn">💾 Save & Restart Config</button>
                </form>
            </div>
            
            <!-- DuckDB Statistics -->
            <div class="card">
                <div class="card-title">Database Status</div>
                <div id="db-stats-content">
                    <div class="db-stat-row">
                        <span>Database Found</span>
                        <span>Checking...</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Dashboard Workspace -->
        <div class="main-content">
            <!-- Console log area -->
            <div class="console-container">
                <div class="console-header">
                    <span>Pipeline Logging Stream</span>
                    <span id="log-status">Ready</span>
                </div>
                <div class="console-body" id="console-body">
                    <div class="console-line-system">System ready. Waiting for user action.</div>
                </div>
            </div>

            <!-- Uploaded archives listing -->
            <div class="card">
                <div class="card-title">Uploaded Archives</div>
                <div id="archives-list" style="max-height: 250px; overflow-y: auto;">
                    <div class="text-muted" style="text-align: center; padding: 1rem;">No archives uploaded yet.</div>
                </div>
            </div>

            <!-- Report list dashboard -->
            <div class="card" style="flex-grow: 1;">
                <div class="card-title">Generated PDF Research Reports</div>
                <div id="reports-grid" class="report-grid">
                    <div class="text-muted" style="text-align: center; padding: 2rem; grid-column: 1 / -1;">
                        No research reports generated yet. Trigger a run with an archive file!
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Poll status every 2 seconds
        setInterval(pollStatus, 2000);
        
        // Initial data fetches
        fetchConfig();
        fetchReports();
        fetchArchives();
        fetchStats();

        function updateGlobalBadge(status) {
            const badge = document.getElementById('global-status-badge');
            badge.className = 'badge';
            
            if (status === 'running') {
                badge.classList.add('badge-running');
                badge.innerText = 'Running';
            } else if (status === 'completed') {
                badge.classList.add('badge-completed');
                badge.innerText = 'Idle / Last Run Completed';
            } else if (status === 'failed') {
                badge.classList.add('badge-failed');
                badge.innerText = 'Last Run Failed';
            } else {
                badge.classList.add('badge-idle');
                badge.innerText = 'Idle';
            }
        }

        async function pollStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                updateGlobalBadge(data.status);
                
                const logBody = document.getElementById('console-body');
                const logStatus = document.getElementById('log-status');
                
                if (data.status === 'running') {
                    logStatus.innerText = 'Executing Pipeline';
                    logStatus.style.color = '#38bdf8';
                } else if (data.status === 'completed') {
                    logStatus.innerText = 'Completed';
                    logStatus.style.color = '#22c55e';
                    // Refresh files and stats
                    fetchReports();
                    fetchStats();
                } else if (data.status === 'failed') {
                    logStatus.innerText = 'Failed';
                    logStatus.style.color = '#ef4444';
                }
                
                if (data.log_output && data.log_output.length > 0) {
                    logBody.innerHTML = '';
                    data.log_output.forEach(line => {
                        const div = document.createElement('div');
                        if (line.includes('failed') || line.includes('crashed') || line.includes('Error')) {
                            div.className = 'console-line-error';
                        }
                        div.innerText = line;
                        logBody.appendChild(div);
                    });
                    // Auto scroll to bottom
                    logBody.scrollTop = logBody.scrollHeight;
                }
            } catch (err) {
                console.error('Error polling status', err);
            }
        }

        async function fetchConfig() {
            try {
                const response = await fetch('/api/config');
                const data = await response.json();
                document.getElementById('cfg-bot-token').value = data.telegram_bot_token;
                document.getElementById('cfg-channel-id').value = data.telegram_channel_id;
                document.getElementById('cfg-mongodb-uri').value = data.mongodb_uri || '';
                document.getElementById('cfg-log-level').value = data.log_level;
                document.getElementById('cfg-log-format').value = data.log_format;
            } catch (err) {
                console.error('Error fetching config', err);
            }
        }

        async function saveConfig(e) {
            e.preventDefault();
            const config = {
                telegram_bot_token: document.getElementById('cfg-bot-token').value,
                telegram_channel_id: parseInt(document.getElementById('cfg-channel-id').value),
                mongodb_uri: document.getElementById('cfg-mongodb-uri').value,
                log_level: document.getElementById('cfg-log-level').value,
                log_format: document.getElementById('cfg-log-format').value
            };
            
            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(config)
                });
                const result = await response.json();
                alert(result.message);
                fetchConfig();
            } catch (err) {
                alert('Failed to save config: ' + err);
            }
        }

        async function fetchReports() {
            try {
                const response = await fetch('/api/reports');
                const data = await response.json();
                const grid = document.getElementById('reports-grid');
                grid.innerHTML = '';
                
                if (data.length === 0) {
                    grid.innerHTML = `<div class="text-muted" style="text-align: center; padding: 2rem; grid-column: 1 / -1;">
                        No research reports generated yet. Trigger a run with an archive file!
                    </div>`;
                    return;
                }
                
                data.forEach(report => {
                    const card = document.createElement('div');
                    card.className = 'report-card';
                    
                    let pVerdict = 'REJECTED';
                    if (report.summary && report.summary.patterns && report.summary.patterns.length > 0) {
                        const verdicts = report.summary.patterns.map(p => p.verdict);
                        if (verdicts.includes('ACCEPTED')) {
                            pVerdict = 'ACCEPTED';
                            card.classList.add('report-card-accepted');
                        } else if (verdicts.includes('MARGINAL')) {
                            pVerdict = 'MARGINAL';
                            card.classList.add('report-card-marginal');
                        } else {
                            card.classList.add('report-card-rejected');
                        }
                    } else {
                        card.classList.add('report-card-rejected');
                    }
                    
                    const formattedDate = report.date;
                    const sizeMB = (report.size_bytes / (1024 * 1024)).toFixed(2);
                    
                    let statsHTML = '';
                    if (report.summary) {
                        statsHTML = `
                            <div class="report-stats">
                                <div class="stat-item">
                                    <span class="stat-label">Total Ticks</span>
                                    <span class="stat-value">${report.summary.total_ticks ? report.summary.total_ticks.toLocaleString() : 'N/A'}</span>
                                </div>
                                <div class="stat-item">
                                    <span class="stat-label">Validation</span>
                                    <span class="stat-value" style="color: ${report.summary.validation_passed ? '#22c55e' : '#ef4444'}">
                                        ${report.summary.validation_passed ? 'PASS' : 'FAIL'}
                                    </span>
                                </div>
                                <div class="stat-item" style="grid-column: 1 / -1; margin-top: 0.25rem;">
                                    <span class="stat-label">Patterns (Acc / Marg / Rej)</span>
                                    <span class="stat-value">
                                        ${report.summary.patterns ? report.summary.patterns.filter(p => p.verdict === 'ACCEPTED').length : 0} / 
                                        ${report.summary.patterns ? report.summary.patterns.filter(p => p.verdict === 'MARGINAL').length : 0} / 
                                        ${report.summary.patterns ? report.summary.patterns.filter(p => p.verdict === 'REJECTED').length : 0}
                                    </span>
                                </div>
                            </div>
                        `;
                    } else {
                        statsHTML = `
                            <div class="report-stats">
                                <div class="stat-item" style="grid-column: 1 / -1; text-align: center;">
                                    <span class="stat-label" style="color: var(--text-muted)">Metadata summary missing</span>
                                </div>
                            </div>
                        `;
                    }
                    
                    card.innerHTML = `
                        <div class="report-meta">
                            <span>${formattedDate}</span>
                            <span>${sizeMB} MB</span>
                        </div>
                        <div class="report-title">Research Report ${formattedDate}</div>
                        ${statsHTML}
                        <div class="report-actions">
                            <a href="/api/reports/${report.filename}" target="_blank" style="text-decoration: none;">
                                <button class="btn btn-secondary">📥 Open PDF Report</button>
                            </a>
                        </div>
                    `;
                    grid.appendChild(card);
                });
            } catch (err) {
                console.error('Error fetching reports', err);
            }
        }

        async function fetchArchives() {
            try {
                const response = await fetch('/api/archives');
                const data = await response.json();
                
                const list = document.getElementById('archives-list');
                const select = document.getElementById('run-archive-select');
                
                list.innerHTML = '';
                select.innerHTML = '<option value="">-- Choose Archive File --</option>';
                
                if (data.length === 0) {
                    list.innerHTML = `<div class="text-muted" style="text-align: center; padding: 1rem;">No archives uploaded yet.</div>`;
                    return;
                }
                
                data.forEach(archive => {
                    const sizeMB = (archive.size_bytes / (1024 * 1024)).toFixed(2);
                    // Add to list
                    const div = document.createElement('div');
                    div.className = 'archive-item';
                    div.innerHTML = `
                        <div class="archive-item-info">
                            <strong>${archive.filename}</strong>
                            <span class="archive-item-size">${sizeMB} MB  ·  Uploaded ${new Date(archive.created_at).toLocaleDateString()}</span>
                        </div>
                        <div class="archive-actions">
                            <button class="btn btn-secondary" style="width: auto; padding: 0.35rem 0.75rem;" onclick="selectArchiveForRun('${archive.filename}')">👉 Select</button>
                        </div>
                    `;
                    list.appendChild(div);
                    
                    // Add to select dropdown
                    const opt = document.createElement('option');
                    opt.value = archive.filename;
                    opt.innerText = `${archive.filename} (${sizeMB} MB)`;
                    select.appendChild(opt);
                });
            } catch (err) {
                console.error('Error fetching archives', err);
            }
        }
        
        function selectArchiveForRun(filename) {
            document.getElementById('run-archive-select').value = filename;
            
            // Auto parse date from filename if of format YYYY-MM-DD.tar.gz
            try {
                const datePart = filename.replace(".tar.gz", "");
                if (datePart.match(/^\d{4}-\d{2}-\d{2}$/)) {
                    document.getElementById('run-date-override').value = datePart;
                }
            } catch(e) {}
        }

        async function fetchStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                const container = document.getElementById('db-stats-content');
                container.innerHTML = '';
                
                if (!data.db_exists) {
                    container.innerHTML = `
                        <div class="db-stat-row">
                            <span>Database Status</span>
                            <span style="color: var(--danger)">NOT FOUND</span>
                        </div>
                        <p style="font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem;">
                            The database will be automatically created on the first successful pipeline run.
                        </p>
                    `;
                    return;
                }
                
                const dbSizeMB = (data.size_bytes / (1024 * 1024)).toFixed(2);
                const dbType = data.db_type || 'DuckDB';
                
                container.innerHTML += `
                    <div class="db-stat-row">
                        <span>Database Type</span>
                        <span style="color: var(--accent); font-weight: bold;">${dbType}</span>
                    </div>
                `;
                
                if (dbType === 'MongoDB') {
                    container.innerHTML += `
                        <div class="db-stat-row">
                            <span>Storage Utilised</span>
                            <span>${dbSizeMB} / 512.00 MB</span>
                        </div>
                        <div class="db-stat-row" style="margin-top: 0.25rem;">
                            <span>Space Utilisation</span>
                            <span style="color: ${data.utilization_pct >= 90 ? 'var(--danger)' : data.utilization_pct >= 75 ? 'var(--warning)' : 'var(--success)'}; font-weight: bold;">${data.utilization_pct}%</span>
                        </div>
                        <div style="background: rgba(255,255,255,0.03); border-radius: 4px; height: 6px; overflow: hidden; margin-top: 0.5rem; border: 1px solid var(--border-color); width: 100%;">
                            <div style="background: ${data.utilization_pct >= 90 ? 'var(--danger)' : data.utilization_pct >= 75 ? 'var(--warning)' : 'var(--accent)'}; width: ${Math.min(data.utilization_pct, 100)}%; height: 100%;"></div>
                        </div>
                    `;
                } else {
                    container.innerHTML += `
                        <div class="db-stat-row">
                            <span>Database Status</span>
                            <span style="color: var(--success)">FOUND</span>
                        </div>
                        <div class="db-stat-row">
                            <span>File Size</span>
                            <span>${dbSizeMB} MB</span>
                        </div>
                    `;
                }
                
                if (data.symbol_ticks && Object.keys(data.symbol_ticks).length > 0) {
                    container.innerHTML += `
                        <div style="font-size: 0.8rem; font-weight: 600; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.5rem; color: var(--accent);">
                            Stored Symbols Coverage:
                        </div>
                    `;
                    for (const [sym, count] of Object.entries(data.symbol_ticks)) {
                        container.innerHTML += `
                            <div class="db-stat-row" style="font-family: 'JetBrains Mono', monospace; font-size: 0.75rem; margin-top: 0.25rem;">
                                <span>${sym}</span>
                                <span>${count.toLocaleString()} ticks</span>
                            </div>
                        `;
                    }
                } else {
                    container.innerHTML += `
                        <div class="db-stat-row">
                            <span>Total Tables</span>
                            <span>${(data.tables && data.tables.join(', ')) || 'None'}</span>
                        </div>
                    `;
                }
            } catch (err) {
                console.error('Error fetching stats', err);
            }
        }

        // Trigger Run
        document.getElementById('btn-run-pipeline').addEventListener('click', async () => {
            const archiveSelect = document.getElementById('run-archive-select');
            const archive = archiveSelect.value;
            const dateOverride = document.getElementById('run-date-override').value;
            
            if (!archive) {
                alert('Please upload or select an archive file first.');
                return;
            }
            
            try {
                let url = `/api/run-job?archive_name=${encodeURIComponent(archive)}`;
                if (dateOverride) {
                    url += `&date=${encodeURIComponent(dateOverride)}`;
                }
                
                const response = await fetch(url, { method: 'POST' });
                if (!response.ok) {
                    const detail = await response.json();
                    throw new Error(detail.detail || 'Failed to start job');
                }
                const result = await response.json();
                
                const consoleBody = document.getElementById('console-body');
                consoleBody.innerHTML = '<div class="console-line-system">Triggering daily job on backend...</div>';
                
                // Switch focus to polling status immediately
                pollStatus();
            } catch (err) {
                alert('Error running pipeline: ' + err.message);
            }
        });

        // Drag and Drop implementation
        const dropzone = document.getElementById('upload-dropzone');
        const fileInput = document.getElementById('upload-file-input');

        dropzone.addEventListener('click', () => fileInput.click());

        dropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        });

        dropzone.addEventListener('dragleave', () => {
            dropzone.classList.remove('dragover');
        });

        dropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
            if (e.dataTransfer.files.length > 0) {
                uploadFile(e.dataTransfer.files[0]);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files.length > 0) {
                uploadFile(fileInput.files[0]);
            }
        });

        async function uploadFile(file) {
            if (!file.name.endsWith('.tar.gz')) {
                alert('Only .tar.gz archives are supported!');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            const logBody = document.getElementById('console-body');
            logBody.innerHTML = `<div class="console-line-system">Uploading ${file.name} (${(file.size / (1024*1024)).toFixed(2)} MB)...</div>`;
            
            try {
                const response = await fetch('/api/upload-archive', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const detail = await response.json();
                    throw new Error(detail.detail || 'Upload failed');
                }
                
                const result = await response.json();
                logBody.innerHTML += `<div class="console-line-system" style="color: #22c55e;">Upload successful! Select the archive from the dropdown to run.</div>`;
                
                fetchArchives();
            } catch (err) {
                logBody.innerHTML += `<div class="console-line-error">Upload failed: ${err.message}</div>`;
                alert('Upload failed: ' + err.message);
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
