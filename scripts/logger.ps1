# Cesta k logovaciemu súboru
$logFile = "C:\Users\username\Desktop\test\log.txt"
$dnsServer = "8.8.8.8"

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
    $logEntry = "[$dayOfWeek, $weekNumber. týždeň, $($currentDate.ToString("dd. MM. yyyy, HH:mm:ss"))] - $dnsServer - $status, latency: $pingTime ms"
    Write-Output $logEntry  # Výpis do konzoly
    Add-Content -Path $logFile -Value $logEntry  # Zápis do logu
}

# Začiatok logovacej relácie
$sessionStart = "========== new session: $(Get-Date) =========="
Write-Output $sessionStart
Add-Content -Path $logFile -Value $sessionStart

# Nekonečná slučka
while ($true) {
    # Spustenie príkazu ping
    $pingOutput = ping -n 1 $dnsServer

    # Spracovanie výstupu
    $pingTime = $null
    foreach ($line in $pingOutput) {
        if ($line -match "time=(\d+)ms") {
            $pingTime = $matches[1]
            break
        }
    }

    # Logovanie výsledku
    if ($pingTime) {
        Log-PingResult -status "Dostupné" -pingTime $pingTime
    } else {
        Log-PingResult -status "Nedostupné" -pingTime "N/A"
    }

    # Pauza medzi testami
    Start-Sleep -Seconds 4
}

