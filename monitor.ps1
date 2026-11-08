while ($true) {
    Clear-Host
    Write-Host "=== CEMR Generation Monitor ===" -ForegroundColor Green
    Write-Host ""
    
    # Check process
    $proc = Get-WmiObject Win32_Process | Where-Object { $_.CommandLine -like "*cemr*generate*" } | Select-Object -First 1
    if ($proc) {
        Write-Host "Process: Running (ID: $($proc.ProcessId))" -ForegroundColor Green
    } else {
        Write-Host "Process: Not found" -ForegroundColor Red
    }
    
    Write-Host ""
    
    # Check conv.json size
    if (Test-Path "cov/gemini/conv.json") {
        $size = (Get-Item "cov/gemini/conv.json").Length
        Write-Host "conv.json size: {0:N2} MB" -f ($size/1MB)
    }
    
    # Check conversation count
    if (Test-Path "cov/gemini/conversations") {
        $count = (Get-ChildItem "cov/gemini/conversations" -Filter *.json | Measure-Object).Count
        Write-Host "Conversation files: $count"
    }
    
    # Show funnel if available
    if (Test-Path "cov/gemini/funnel.json") {
        try {
            $funnel = Get-Content "cov/gemini/funnel.json" | ConvertFrom-Json
            Write-Host ""
            Write-Host "Funnel status:" -ForegroundColor Yellow
            $funnel | Format-List
        } catch {}
    }
    
    Write-Host ""
    Write-Host "Press Ctrl+C to stop monitoring" -ForegroundColor DarkGray
    Start-Sleep 10
}
