Param(
    [string]$LogFile = "C:\Users\username\Desktop\test\log.txt",
    [string]$DnsServer = "8.8.8.8",
    [int]$IntervalSeconds = 4
)

# Ensure log directory exists
$logDir = Split-Path $LogFile -Parent
if (-not (Test-Path -Path $logDir)) {
    try {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    } catch {
        Write-Error "Unable to create log directory '$logDir': $_"
        exit 1
    }
}

# Funkcia na získanie dňa v týždni
function Get-DayOfWeekName {
    param ([datetime]$date)
    switch ($date.DayOfWeek) {
        "Monday"    { return "Pondelok" }
        "Tuesday"   { return "Utorok" }
        "Wednesday" { return "Streda" }
        "Thursday"  { return "Štvrtok" }
        "Friday"    { return "Piatok" }
        "Saturday"  { return "Sobota" }
        "Sunday"    { return "Nedeľa" }
        default     { return "Nedefinovaný" }
    }
}

# Funkcia na získanie čísla týždňa
function Get-WeekNumber {
    param ([datetime]$date)
    $culture = [System.Globalization.CultureInfo]::CurrentCulture
    $calendar = $culture.Calendar
    $calendar.GetWeekOfYear($date, $culture.DateTimeFormat.CalendarWeekRule, $culture.DateTimeFormat.FirstDayOfWeek)
}

# Funkcia na logovanie výsledkov
function Log-PingResult {
    param (
        [string]$status,
        [string]$pingTime
    )
    $currentDate = Get-Date
    $dayOfWeek = Get-DayOfWeekName -date $currentDate
    $weekNumber = Get-WeekNumber -date $currentDate
    $latencyText = if ($pingTime -eq 'N/A') { 'N/A' } else { "$pingTime ms" }
    $logEntry = "[$dayOfWeek, $weekNumber. týždeň, $($currentDate.ToString('dd. MM. yyyy, HH:mm:ss'))] - $DnsServer - $status, latency: $latencyText"
    Write-Output $logEntry  # Výpis do konzoly
    try {
        Add-Content -Path $LogFile -Value $logEntry -Encoding UTF8  # Zápis do logu
    } catch {
        Write-Error "Failed to write to log file '$LogFile': $_"
    }
}

# Začiatok logovacej relácie
$sessionStart = "========== new session: $(Get-Date) =========="
Write-Output $sessionStart
try {
    Add-Content -Path $LogFile -Value $sessionStart -Encoding UTF8
} catch {
    Write-Error "Failed to write session start to log file: $_"
}

# Nekonečná slučka
while ($true) {
    try {
        $reply = Test-Connection -ComputerName $DnsServer -Count 1 -ErrorAction Stop
        if ($reply) {
            $pingTime = $reply.ResponseTime
            Log-PingResult -status "Dostupné" -pingTime $pingTime
        } else {
            Log-PingResult -status "Nedostupné" -pingTime "N/A"
        }
    } catch {
        Log-PingResult -status "Nedostupné" -pingTime "N/A"
    }

    # Pauza medzi testami
    Start-Sleep -Seconds $IntervalSeconds
}

