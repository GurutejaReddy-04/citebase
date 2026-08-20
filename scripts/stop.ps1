# stop.ps1 - Cleanly stop CiteBase backend server on Windows
Write-Host "Checking for CiteBase process on port 8000..." -ForegroundColor Cyan
$pids = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique

if ($pids) {
    foreach ($procId in $pids) {
        $proc = Get-Process -Id $procId -ErrorAction SilentlyContinue
        Write-Host "Stopping PID $procId ($($proc.ProcessName))..." -ForegroundColor Yellow
        Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 500
    Write-Host "CiteBase server successfully stopped. Port 8000 is clean." -ForegroundColor Green
} else {
    Write-Host "No process is currently listening on port 8000." -ForegroundColor Gray
}
