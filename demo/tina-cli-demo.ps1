[CmdletBinding()]
param(
    [ValidateSet("en", "zh")]
    [string]$Language = "en",
    [int]$TerminalWidth = 0,
    [string]$AssistantId = "Tina",
    [string[]]$Commands,
    [switch]$Demo,
    [ValidateSet("auto", "sixel", "text", "none")]
    [string]$Splash = "auto",
    [ValidateSet("right", "closed")]
    [string]$WaterFlow = "right",
    [switch]$ToolShowcase,
    [switch]$EditShowcase,
    [switch]$TodoShowcase,
    [switch]$GitShowcase,
    [switch]$CommandShowcase,
    [switch]$TideLoop,
    [switch]$NoColor,
    [switch]$NoMotion,
    [switch]$NoClear
)

<#
    Tina CLI shell preview.

    This is intentionally model-free: it does not load an artifact, create a
    CUDA context, invoke a tool host, or mutate the repository. It is a UX
    preview for the public command surface and safety copy.
#>

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$script:AutoTerminalWidth = $TerminalWidth -eq 0
$script:MinimumTerminalWidth = 40
$script:MaximumTerminalWidth = 120
$script:FallbackTerminalWidth = 80

function Get-DetectedTerminalWidth {
    try {
        return [int]$Host.UI.RawUI.WindowSize.Width
    } catch {
        return $script:FallbackTerminalWidth
    }
}

function Sync-TerminalWidth {
    if (-not $script:AutoTerminalWidth) { return }
    $detected = Get-DetectedTerminalWidth
    $script:TerminalWidth = [Math]::Min(
        $script:MaximumTerminalWidth,
        [Math]::Max($script:MinimumTerminalWidth, $detected)
    )
}

if ($script:AutoTerminalWidth) {
    $TerminalWidth = Get-DetectedTerminalWidth
}
$TerminalWidth = [Math]::Min(
    $script:MaximumTerminalWidth,
    [Math]::Max($script:MinimumTerminalWidth, $TerminalWidth)
)

$script:UseColor = -not $NoColor -and $env:NO_COLOR -ne "1"
$script:UseMotion = -not $NoMotion -and -not [Console]::IsOutputRedirected
$script:Esc = [char]27
$script:Reset = if ($script:UseColor) { "$($script:Esc)[0m" } else { "" }
$script:Inverse = if ($script:UseColor) { "$($script:Esc)[7m" } else { "" }
$script:NoInverse = if ($script:UseColor) { "$($script:Esc)[27m" } else { "" }
$script:DeepWater = if ($script:UseColor) { "$($script:Esc)[38;2;50;82;139m" } else { "" }
$script:Tide = if ($script:UseColor) { "$($script:Esc)[38;2;80;159;214m" } else { "" }
$script:Accent = if ($script:UseColor) { "$($script:Esc)[38;2;126;202;237m" } else { "" }
$script:Moon = if ($script:UseColor) { "$($script:Esc)[38;2;205;228;250m" } else { "" }
$script:MoonBright = if ($script:UseColor) { "$($script:Esc)[1;38;2;245;249;255m" } else { "" }
$script:Border = $script:MoonBright
$script:EditBorder = $script:Border
$script:TinaBorder = $script:Border
$script:UserBorder = $script:Border
$script:Reply = $script:MoonBright
$script:Lilac = if ($script:UseColor) { "$($script:Esc)[38;2;178;166;230m" } else { "" }
$script:Muted = if ($script:UseColor) { "$($script:Esc)[38;2;108;130;159m" } else { "" }
$script:Green = if ($script:UseColor) { "$($script:Esc)[38;2;125;211;160m" } else { "" }
$script:DiffGreen = if ($script:UseColor) { "$($script:Esc)[1;38;2;0;220;72m" } else { "" }
$script:Yellow = if ($script:UseColor) { "$($script:Esc)[38;2;235;202;125m" } else { "" }
$script:Red = if ($script:UseColor) { "$($script:Esc)[38;2;242;142;142m" } else { "" }
$script:DiffRed = if ($script:UseColor) { "$($script:Esc)[1;38;2;248;45;45m" } else { "" }
$script:LastUiLineWasBlank = $false
$script:MajorGapLines = 2
$script:ToolGapLines = 3
$script:SlashPopupRows = 8
$script:SlashCommands = @(
    [pscustomobject]@{ Name = "/agent"; Aliases = @("/代理"); English = "show Agent status and access mode"; Chinese = "查看 Agent 状态与权限模式" },
    [pscustomobject]@{ Name = "/goal"; Aliases = @("/目标"); English = "show, set, or clear the session goal"; Chinese = "查看、设置或清除会话目标" },
    [pscustomobject]@{ Name = "/thinking"; Aliases = @("/思考"); English = "choose off, medium, high, or max reasoning"; Chinese = "选择 off、medium、high 或 max 思考级别" },
    [pscustomobject]@{ Name = "/status"; Aliases = @("/状态"); English = "show runtime, context, cache, and Agent status"; Chinese = "查看运行、上下文、缓存和 Agent 状态" },
    [pscustomobject]@{ Name = "/context"; Aliases = @("/上下文"); English = "show the 258K activity-context budget"; Chinese = "查看 258K 活动上下文预算" },
    [pscustomobject]@{ Name = "/help"; Aliases = @("/帮助", "/指令"); English = "show all commands and argument forms"; Chinese = "查看全部指令与参数格式" },
    [pscustomobject]@{ Name = "/tools"; Aliases = @("/工具"); English = "show enabled tools and safety boundaries"; Chinese = "查看已启用工具与安全边界" },
    [pscustomobject]@{ Name = "/environment"; Aliases = @("/环境", "/终端"); English = "show the OS, shell, terminal, and cwd"; Chinese = "查看 OS、shell、终端与工作目录" },
    [pscustomobject]@{ Name = "/diff"; Aliases = @("/差异"); English = "show staged and unstaged workspace changes"; Chinese = "查看工作树已暂存与未暂存改动" },
    [pscustomobject]@{ Name = "/review"; Aliases = @("/审查"); English = "run a read-only review of current changes"; Chinese = "只读审查当前改动" },
    [pscustomobject]@{ Name = "/todo"; Aliases = @("/待办", "/任务"); English = "show or update the session TodoList"; Chinese = "查看或更新会话 TodoList" },
    [pscustomobject]@{ Name = "/sessions"; Aliases = @("/会话"); English = "list resumable conversation archives"; Chinese = "列出可恢复的会话归档" },
    [pscustomobject]@{ Name = "/resume"; Aliases = @("/恢复"); English = "switch to latest or a full archive ID"; Chinese = "切换到最新归档或完整归档 ID" },
    [pscustomobject]@{ Name = "/retry"; Aliases = @("/重试"); English = "retry the last user turn on a new branch"; Chinese = "在新分支重试上一用户回合" },
    [pscustomobject]@{ Name = "/new"; Aliases = @("/新会话", "/重置", "/清空"); English = "start a fresh session and retain the archive"; Chinese = "开始新会话并保留旧归档" },
    [pscustomobject]@{ Name = "/compact"; Aliases = @("/压缩"); English = "create a manual context handoff"; Chinese = "主动创建上下文交接" },
    [pscustomobject]@{ Name = "/export"; Aliases = @("/导出"); English = "export the active branch as JSONL or Markdown"; Chinese = "将活动分支导出为 JSONL 或 Markdown" },
    [pscustomobject]@{ Name = "/memory"; Aliases = @("/记忆"); English = "show pinned conversation memory"; Chinese = "查看已固定的会话记忆" },
    [pscustomobject]@{ Name = "/remember"; Aliases = @("/记住"); English = "pin a fact in this conversation"; Chinese = "在当前会话固定一条事实" },
    [pscustomobject]@{ Name = "/project-memory"; Aliases = @("/项目记忆"); English = "show Git-backed project memory"; Chinese = "查看 Git 持久化项目记忆" },
    [pscustomobject]@{ Name = "/project-remember"; Aliases = @("/项目记住"); English = "persist a project fact across sessions"; Chinese = "跨会话保存项目事实" },
    [pscustomobject]@{ Name = "/project-forget"; Aliases = @("/项目忘记"); English = "forget a project fact through the delete gate"; Chinese = "经删除拦截忘记项目事实" },
    [pscustomobject]@{ Name = "/plugins"; Aliases = @("/插件"); English = "list declarative plugins and readiness"; Chinese = "列出声明式插件及就绪状态" },
    [pscustomobject]@{ Name = "/assistant"; Aliases = @("/助手"); English = "show or change the assistant ID"; Chinese = "查看或修改助手 ID" },
    [pscustomobject]@{ Name = "/language"; Aliases = @("/语言"); English = "show or change the terminal language"; Chinese = "查看或切换终端界面语言" },
    [pscustomobject]@{ Name = "/agent-restart"; Aliases = @("/重启工具"); English = "restart the bounded Agent tool host"; Chinese = "重启受控 Agent tool-host" },
    [pscustomobject]@{ Name = "/delete"; Aliases = @(); English = "preview the destructive-file safety gate"; Chinese = "预览破坏性文件删除拦截" },
    [pscustomobject]@{ Name = "/exit"; Aliases = @("/退出"); English = "close Tina and retain the conversation archive"; Chinese = "关闭 Tina 并保留会话归档" }
)

function T {
    param(
        [Parameter(Mandatory = $true)][string]$English,
        [Parameter(Mandatory = $true)][string]$Chinese
    )
    if ($script:State.Language -eq "zh") { return $Chinese }
    return $English
}

function Write-Ui {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [string]$Color = ""
    )
    # Keep ordinary rows compact. Larger surfaces use Write-UiGap so the
    # transcript, tool batch, and composer do not visually collapse together.
    $isBlank = $Text.Length -eq 0
    if ($isBlank -and $script:LastUiLineWasBlank) { return }
    Write-Host "$Color$Text$($script:Reset)"
    $script:LastUiLineWasBlank = $isBlank
}

function Write-UiGap {
    param(
        [ValidateRange(0, 8)][int]$Lines = $script:MajorGapLines
    )
    if ($Lines -le 0) { return }
    $alreadyBlank = if ($script:LastUiLineWasBlank) { 1 } else { 0 }
    $needed = [Math]::Max(0, $Lines - $alreadyBlank)
    for ($index = 0; $index -lt $needed; $index++) {
        Write-Host ""
    }
    $script:LastUiLineWasBlank = $true
}

function Write-UiInline {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [string]$Color = ""
    )
    Write-Host "$Color$Text$($script:Reset)" -NoNewline
    if ($Text.Length -gt 0) { $script:LastUiLineWasBlank = $false }
}

function Get-CellWidth {
    param([AllowEmptyString()][string]$Text)
    $plain = [regex]::Replace($Text, '\x1b\[[0-?]*[ -/]*[@-~]', '')
    $width = 0
    foreach ($character in $plain.ToCharArray()) {
        $code = [int]$character
        $isWide =
            ($code -ge 0x1100 -and $code -le 0x115f) -or
            ($code -ge 0x2e80 -and $code -le 0xa4cf) -or
            ($code -ge 0xac00 -and $code -le 0xd7a3) -or
            ($code -ge 0xf900 -and $code -le 0xfaff) -or
            ($code -ge 0xfe10 -and $code -le 0xfe6f) -or
            ($code -ge 0xff01 -and $code -le 0xff60) -or
            ($code -ge 0xffe0 -and $code -le 0xffe6)
        $width += if ($isWide) { 2 } else { 1 }
    }
    return $width
}

function Fit-UiText {
    param(
        [AllowEmptyString()][string]$Text,
        [int]$Width
    )
    if ($Width -le 0) { return "" }
    if ((Get-CellWidth $Text) -le $Width) { return $Text }
    $ellipsis = if ($Width -ge 3) { "..." } else { "." * $Width }
    $ellipsisWidth = Get-CellWidth $ellipsis
    $result = [System.Text.StringBuilder]::new()
    $used = 0
    foreach ($character in $Text.ToCharArray()) {
        $cellWidth = Get-CellWidth ([string]$character)
        if ($used + $cellWidth + $ellipsisWidth -gt $Width) { break }
        [void]$result.Append($character)
        $used += $cellWidth
    }
    [void]$result.Append($ellipsis)
    return $result.ToString()
}

function Join-UiColumns {
    param(
        [Parameter(Mandatory = $true)][string]$Left,
        [Parameter(Mandatory = $true)][string]$Right,
        [int]$Width = $TerminalWidth
    )
    $rightWidth = Get-CellWidth $Right
    $left = Fit-UiText $Left ([Math]::Max(1, $Width - $rightWidth - 1))
    $gap = [Math]::Max(1, $Width - (Get-CellWidth $left) - $rightWidth)
    return $left + (" " * $gap) + $Right
}

function Center-UiText {
    param(
        [AllowEmptyString()][string]$Text,
        [int]$Width = $TerminalWidth
    )
    $text = Fit-UiText $Text $Width
    $padding = [Math]::Max(0, [int][Math]::Floor(($Width - (Get-CellWidth $text)) / 2))
    return (" " * $padding) + $text
}

function Get-WelcomeFrameLines {
    param(
        [Parameter(Mandatory = $true)][string]$Title,
        [Parameter(Mandatory = $true)][string]$Badge,
        [Parameter(Mandatory = $true)][string[]]$Rows
    )
    $borderWidth = [Math]::Max(1, $TerminalWidth - 2)
    $right = " $Badge ─"
    $titleWidth = [Math]::Max(1, $borderWidth - (Get-CellWidth $right) - 3)
    $title = Fit-UiText $Title $titleWidth
    $left = "─ $title "
    $fill = [Math]::Max(0, $borderWidth - (Get-CellWidth $left) - (Get-CellWidth $right))
    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("╭$left$("─" * $fill)$right╮")

    $contentWidth = [Math]::Max(1, $TerminalWidth - 4)
    foreach ($item in $Rows) {
        $row = Fit-UiText $item $contentWidth
        $remaining = [Math]::Max(0, $contentWidth - (Get-CellWidth $row))
        $leftPadding = [int][Math]::Floor($remaining / 2)
        $rightPadding = $remaining - $leftPadding
        $lines.Add("│ $(" " * $leftPadding)$row$(" " * $rightPadding) │")
    }
    $lines.Add("╰$("─" * $borderWidth)╯")
    return $lines.ToArray()
}

function Get-SlashCommandToken {
    param([AllowEmptyString()][string]$Text)
    $firstLine = ([regex]::Split($Text, '\r?\n', 2))[0]
    if (-not $firstLine.StartsWith("/") -or $firstLine.StartsWith("//")) {
        return $null
    }
    return ([regex]::Split($firstLine, '\s+', 2))[0]
}

function Get-SlashCommandCandidates {
    param([AllowEmptyString()][string]$Text)
    $token = Get-SlashCommandToken $Text
    if ($null -eq $token) { return @() }

    $useAlias = $token -match '[^\x00-\x7f]'
    $tokenLower = $token.ToLowerInvariant()
    $exact = [System.Collections.Generic.List[object]]::new()
    $prefixed = [System.Collections.Generic.List[object]]::new()
    foreach ($command in $script:SlashCommands) {
        $displayName = $command.Name
        if ($useAlias) {
            $alias = $command.Aliases | Where-Object {
                $_.ToLowerInvariant().StartsWith($tokenLower)
            } | Select-Object -First 1
            if ($null -eq $alias) { continue }
            $displayName = $alias
        }

        $nameLower = $displayName.ToLowerInvariant()
        if (-not $nameLower.StartsWith($tokenLower)) { continue }
        $candidate = [pscustomobject]@{
            Name = $displayName
            Description = if ($script:State.Language -eq "zh") { $command.Chinese } else { $command.English }
        }
        if ($nameLower -eq $tokenLower) {
            $exact.Add($candidate)
        } else {
            $prefixed.Add($candidate)
        }
    }
    return @($exact) + @($prefixed)
}

function Show-CommandPopup {
    param(
        [AllowEmptyString()][string]$Prefix,
        [int]$SelectedIndex = 0
    )
    $candidates = @(Get-SlashCommandCandidates $Prefix)
    if ($candidates.Count -eq 0) {
        Write-Ui (T "  no matching commands" "  没有匹配的指令") $script:Muted
        return
    }

    $selected = [Math]::Min([Math]::Max(0, $SelectedIndex), $candidates.Count - 1)
    $visibleCount = [Math]::Min($script:SlashPopupRows, $candidates.Count)
    $start = [Math]::Min(
        [Math]::Max(0, $selected - [int][Math]::Floor($visibleCount / 2)),
        $candidates.Count - $visibleCount
    )
    $visible = @($candidates[$start..($start + $visibleCount - 1)])
    $nameLimit = [Math]::Max(8, [Math]::Min(22, [int][Math]::Floor($TerminalWidth / 2)))
    $nameWidth = 1
    foreach ($candidate in $visible) {
        $nameWidth = [Math]::Max($nameWidth, [Math]::Min($nameLimit, (Get-CellWidth $candidate.Name)))
    }
    $descriptionWidth = [Math]::Max(1, $TerminalWidth - 6 - $nameWidth)

    for ($offset = 0; $offset -lt $visible.Count; $offset++) {
        $candidate = $visible[$offset]
        $index = $start + $offset
        $name = Fit-UiText $candidate.Name $nameWidth
        $namePadding = " " * [Math]::Max(0, $nameWidth - (Get-CellWidth $name))
        $description = Fit-UiText $candidate.Description $descriptionWidth
        $marker = if ($index -eq $selected) { "›" } else { " " }
        $line = "  $marker $name$namePadding  $description"
        $line += " " * [Math]::Max(0, $TerminalWidth - (Get-CellWidth $line))
        if ($index -eq $selected) {
            Write-Ui $line ($script:Inverse + $script:MoonBright)
        } else {
            Write-UiInline "    $name$namePadding  " $script:Moon
            Write-Ui ($description + (" " * [Math]::Max(0, $TerminalWidth - 6 - $nameWidth - (Get-CellWidth $description)))) $script:Muted
        }
    }
}

function Get-TideLine {
    param(
        [int]$Width,
        [double]$Phase = 0.0,
        [double]$Strength = 1.0
    )
    $glyphs = @(" ", "·", ".", "~", "≈")
    $line = [System.Text.StringBuilder]::new([Math]::Max(0, $Width))
    for ($x = 0; $x -lt $Width; $x++) {
        # Two related waves produce a deterministic interference pattern. The
        # phase changes only while work is active, so the idle UI stays still.
        $wave =
            [Math]::Sin(($x * 0.31) + $Phase) +
            (0.45 * [Math]::Sin(($x * 0.59) - (0.70 * $Phase)))
        $normalized = (($wave * $Strength) + 1.45) / 2.90
        $index = [Math]::Min(
            $glyphs.Count - 1,
            [Math]::Max(0, [int][Math]::Floor($normalized * $glyphs.Count))
        )
        [void]$line.Append($glyphs[$index])
    }
    return $line.ToString()
}

function Get-FlowLine {
    param(
        [int]$Width,
        [double]$Phase = 0.0
    )
    $glyphs = @(" ", "·", ".", "~", "≈")
    $line = [System.Text.StringBuilder]::new([Math]::Max(0, $Width))
    $span = [Math]::Max(1, $Width - 1)
    for ($x = 0; $x -lt $Width; $x++) {
        # The rising bias makes the wave read left-to-right instead of looking
        # like a static divider. It remains deterministic for transcript tests.
        $trend = 0.72 * ($x / $span)
        $wave =
            [Math]::Sin(($x * 0.31) + $Phase) +
            (0.45 * [Math]::Sin(($x * 0.59) - (0.70 * $Phase))) +
            $trend - 0.15
        $normalized = ($wave + 1.60) / 3.25
        $index = [Math]::Min(
            $glyphs.Count - 1,
            [Math]::Max(0, [int][Math]::Floor($normalized * $glyphs.Count))
        )
        [void]$line.Append($glyphs[$index])
    }
    return $line.ToString()
}

function Get-RegularTideLine {
    param(
        [int]$Width,
        [double]$Phase = 0.0,
        [double]$Amplitude = 1.0
    )
    $glyphs = @("·", "·", ".", "~", "≈")
    $line = [System.Text.StringBuilder]::new([Math]::Max(0, $Width))
    $wavelength = 18.0
    $amplitude = [Math]::Min(1.0, [Math]::Max(0.0, $Amplitude))
    for ($x = 0; $x -lt $Width; $x++) {
        # One dominant travelling sine keeps each crest and trough evenly
        # spaced. Amplitude is controlled separately for calm entry/settling.
        $wave = 0.5 + (0.5 * [Math]::Sin(((2.0 * [Math]::PI * $x) / $wavelength) - $Phase))
        $level = $amplitude * $wave
        $index = [Math]::Min(
            $glyphs.Count - 1,
            [Math]::Max(0, [int][Math]::Round($level * ($glyphs.Count - 1)))
        )
        [void]$line.Append($glyphs[$index])
    }
    return $line.ToString()
}

function Test-SixelAllowed {
    if ([Console]::IsOutputRedirected -or $TerminalWidth -lt 72) { return $false }
    if ($env:TERM_PROGRAM -and $env:TERM_PROGRAM.Equals("vscode", [StringComparison]::OrdinalIgnoreCase)) {
        return $false
    }
    if ($env:VSCODE_INJECTION -in @("1", "true")) { return $false }
    return $true
}

function Test-SixelTerminal {
    if (-not (Test-SixelAllowed)) { return $false }
    if ($env:WT_SESSION) { return $true }
    if ($env:TERM -and $env:TERM.ToLowerInvariant().Contains("sixel")) { return $true }
    if ($env:TERM_PROGRAM -and $env:TERM_PROGRAM.ToLowerInvariant() -in @("contour", "foot", "konsole", "mintty", "mlterm", "rio", "wezterm")) {
        return $true
    }
    return $false
}

function Write-SixelSplash {
    $assetName = if ($TerminalWidth -ge 108) { "luna_splash_960.sixel" } else { "luna_splash.sixel" }
    $asset = Join-Path $PSScriptRoot "..\crates\cq-infer\assets\$assetName"
    if (-not (Test-Path -LiteralPath $asset)) { return $false }
    try {
        $output = [Console]::OpenStandardOutput()
        $prefix = [Text.Encoding]::UTF8.GetBytes("`n$($script:Esc)[?2026h$($script:Esc)[?7l$($script:Esc)[?25l")
        $payload = [IO.File]::ReadAllBytes((Resolve-Path -LiteralPath $asset))
        $suffix = [Text.Encoding]::UTF8.GetBytes("$($script:Esc)[0m`n$($script:Esc)[?7h$($script:Esc)[?25h$($script:Esc)[?2026l`n")
        $output.Write($prefix, 0, $prefix.Length)
        $output.Write($payload, 0, $payload.Length)
        $output.Write($suffix, 0, $suffix.Length)
        $output.Flush()
        return $true
    } catch {
        return $false
    }
}

function Show-TidalTextSplash {
    Write-Ui ""
    Write-Ui (Join-UiColumns "  T I D A L   L U N A" "☾  ") $script:MoonBright
    Write-Ui (Join-UiColumns "  Tina Engine" "UI PREVIEW  ") $script:Accent
    Write-Ui ("  " + (Get-TideLine ([Math]::Max(1, $TerminalWidth - 4)) 1.7 0.92)) $script:Tide
    Write-Ui (Center-UiText "tidal luna" $TerminalWidth) $script:DeepWater
}

function Show-StartupArt {
    if ($Splash -eq "none") { return $false }
    $useSixel = (Test-SixelAllowed) -and ($Splash -eq "sixel" -or ($Splash -eq "auto" -and (Test-SixelTerminal)))
    if ($useSixel -and (Write-SixelSplash)) { return $true }
    Show-TidalTextSplash
    return $false
}

function Write-Rule {
    $rule = "─" * $TerminalWidth
    Write-Ui $rule $script:Border
}

function Write-Row {
    param(
        [Parameter(Mandatory = $true)][string]$Key,
        [Parameter(Mandatory = $true)][string]$Value,
        [string]$ValueColor = ""
    )
    # A delimiter is more stable than fixed padding when Chinese and Latin
    # labels share the same terminal row.
    Write-Ui ("  {0}: {1}" -f $Key, $Value) $ValueColor
}

function Set-AssistantId {
    param([string]$Value)
    $candidate = $Value.Trim()
    if ([string]::IsNullOrWhiteSpace($candidate) -or $candidate.Length -gt 32) {
        Write-Ui (T "Assistant ID must be 1-32 characters." "助手 ID 必须为 1-32 个字符。") $script:Red
        return
    }
    $script:State.AssistantId = $candidate
    Write-Ui (T "Assistant ID set to '$candidate'." "助手 ID 已改为 '$candidate'。") $script:Green
}

function Set-Thinking {
    param([string]$Value)
    $normalized = $Value.Trim().ToLowerInvariant()
    $level = switch ($normalized) {
        { $_ -in @("off", "non-thinking", "nonthinking", "chat", "none") } { "off"; break }
        { $_ -in @("medium", "med", "low") } { "medium"; break }
        "high" { "high"; break }
        { $_ -in @("max", "xhigh") } { "max"; break }
        default { $null }
    }
    if ($null -eq $level) {
        Write-Ui (T "Use: /thinking off|medium|high|max" "用法：/thinking off|medium|high|max") $script:Red
        return
    }
    $script:State.Thinking = $level
    Write-Ui (T "Thinking level: $level." "思考级别：$level。") $script:Green
}

function Write-ComposerTop {
    $label = " User "
    $fill = [Math]::Max(0, $TerminalWidth - 3 - (Get-CellWidth $label))
    Write-UiInline "╭─" $script:UserBorder
    Write-UiInline $label $script:MoonBright
    Write-Ui (("─" * $fill) + "╮") $script:UserBorder
}

function Write-ComposerLine {
    param([AllowEmptyString()][string]$Text)
    $contentWidth = [Math]::Max(1, $TerminalWidth - 4)
    $content = Fit-UiText $Text $contentWidth
    $padding = [Math]::Max(0, $contentWidth - (Get-CellWidth $content))
    Write-UiInline "│ " $script:UserBorder
    Write-UiInline ($content + (" " * $padding)) $script:Moon
    Write-Ui " │" $script:UserBorder
}

function Write-ComposerQueuedText {
    param([Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text)
    $contentWidth = [Math]::Max(3, $TerminalWidth - 4)
    $messageWidth = [Math]::Max(1, $contentWidth - 2)
    $wrapped = @(Split-UiText $Text $messageWidth)
    if ($wrapped.Count -eq 0) { $wrapped = @($Text) }
    for ($index = 0; $index -lt $wrapped.Count; $index++) {
        $prefix = if ($index -eq 0) { "› " } else { "  " }
        Write-ComposerLine ($prefix + $wrapped[$index])
    }
}

function Write-ComposerBottom {
    $left = if ($script:State.Language -eq "zh") {
        " Agent $($script:State.Agent) · 思考 $($script:State.Thinking) "
    } else {
        " Agent $($script:State.Agent) · think $($script:State.Thinking) "
    }
    $right = " $($script:State.Context)/258K "
    $available = [Math]::Max(1, $TerminalWidth - 2 - (Get-CellWidth $right))
    $left = Fit-UiText $left $available
    $fill = [Math]::Max(0, $TerminalWidth - 2 - (Get-CellWidth $left) - (Get-CellWidth $right))
    Write-UiInline "╰" $script:UserBorder
    Write-UiInline $left $script:DeepWater
    Write-UiInline ("─" * $fill) $script:UserBorder
    Write-UiInline $right $script:DeepWater
    Write-Ui "╯" $script:UserBorder
}

function Read-Composer {
    param(
        [AllowEmptyString()][string]$QueuedLine,
        [switch]$FromQueue
    )
    Write-UiGap $script:MajorGapLines
    Write-ComposerTop
    if ($FromQueue) {
        Write-ComposerQueuedText $QueuedLine
        $line = $QueuedLine
    } else {
        Write-UiInline "│ " $script:UserBorder
        Write-UiInline "› " $script:MoonBright
        $line = [Console]::ReadLine()
        if ($null -eq $line) { $line = "/exit" }
    }
    $slashToken = Get-SlashCommandToken $line
    if ($null -ne $slashToken -and (-not $FromQueue -or $CommandShowcase -or $slashToken -eq "/")) {
        Show-CommandPopup $line
        $candidates = @(Get-SlashCommandCandidates $line)
        if ($slashToken -ne "/" -and $line.Trim() -eq $slashToken -and $candidates.Count -eq 1) {
            $line = $candidates[0].Name
        }
    }
    Write-ComposerBottom
    return $line
}

function Show-TideWait {
    $label = if ($script:State.Language -eq "zh") { "Tina 正在观察潮汐" } else { "Tina is reading the tide" }
    if (-not $script:UseMotion) {
        Write-Ui ("  $label  " + (Get-TideLine 18 1.2 0.85)) $script:Tide
        return
    }
    $motionWidth = 2 + (Get-CellWidth $label) + 2 + 22
    for ($frame = 0; $frame -lt 10; $frame++) {
        $progress = [Math]::Min(1.0, $frame / 3.0)
        $amplitude = $progress * $progress * (3.0 - (2.0 * $progress))
        $wave = Get-RegularTideLine 22 ($frame * 0.52) $amplitude
        Write-MotionLine "$($script:Tide)  $label  $wave$($script:Reset)" $motionWidth
        Start-Sleep -Milliseconds 125
    }
    Clear-MotionLine $motionWidth
}

function Split-UiText {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [int]$Width
    )
    $hasWideGlyph = $false
    foreach ($character in $Text.ToCharArray()) {
        if ((Get-CellWidth ([string]$character)) -gt 1) {
            $hasWideGlyph = $true
            break
        }
    }
    # CJK prose has no reliable whitespace boundaries, so fill it by terminal
    # cells; pure Latin prose keeps word boundaries for readability.
    if ($hasWideGlyph) {
        return Split-UiTextCellGreedy $Text $Width
    }

    $lines = [System.Collections.Generic.List[string]]::new()
    $current = ""
    foreach ($match in [regex]::Matches($Text, '\S+')) {
        $word = $match.Value
        $candidate = if ($current) { "$current $word" } else { $word }
        if ((Get-CellWidth $candidate) -le $Width) {
            $current = $candidate
            continue
        }
        if ($current) {
            $lines.Add($current)
            $current = ""
        }
        if ((Get-CellWidth $word) -le $Width) {
            $current = $word
            continue
        }

        $chunk = [System.Text.StringBuilder]::new()
        $chunkWidth = 0
        foreach ($character in $word.ToCharArray()) {
            $cellWidth = Get-CellWidth ([string]$character)
            if ($chunkWidth + $cellWidth -gt $Width -and $chunk.Length -gt 0) {
                $lines.Add($chunk.ToString())
                [void]$chunk.Clear()
                $chunkWidth = 0
            }
            [void]$chunk.Append($character)
            $chunkWidth += $cellWidth
        }
        $current = $chunk.ToString()
    }
    if ($current) { $lines.Add($current) }
    return $lines.ToArray()
}

function Split-UiTextCellGreedy {
    param(
        [Parameter(Mandatory = $true)][string]$Text,
        [int]$Width
    )
    if ($Width -le 0) { return @() }

    $lines = [System.Collections.Generic.List[string]]::new()
    $current = [System.Text.StringBuilder]::new()
    $currentWidth = 0
    $pendingSpace = $false

    foreach ($character in $Text.ToCharArray()) {
        if ([char]::IsWhiteSpace($character)) {
            if ($current.Length -gt 0) { $pendingSpace = $true }
            continue
        }

        $cellWidth = Get-CellWidth ([string]$character)
        $spaceWidth = if ($pendingSpace -and $current.Length -gt 0) { 1 } else { 0 }
        if ($current.Length -gt 0 -and $currentWidth + $spaceWidth + $cellWidth -gt $Width) {
            $lines.Add($current.ToString())
            [void]$current.Clear()
            $currentWidth = 0
            $pendingSpace = $false
            $spaceWidth = 0
        }

        if ($spaceWidth -gt 0) {
            [void]$current.Append(' ')
            $currentWidth++
        }
        [void]$current.Append($character)
        $currentWidth += $cellWidth
        $pendingSpace = $false
    }

    if ($current.Length -gt 0) { $lines.Add($current.ToString()) }
    return $lines.ToArray()
}

function Write-ToolPreview {
    $readLabel = if ($script:State.Language -eq "zh") { "读取" } else { "Read" }
    $shapeLabel = if ($script:State.Language -eq "zh") { "布局" } else { "Shape" }
    Write-Ui ("  ◇ {0,-7} crates/cq-infer/src/tina_cli.rs  · preview" -f $readLabel) $script:Muted
    Write-Ui ("  ◇ {0,-7} {1} columns · scrollback retained" -f $shapeLabel, $TerminalWidth) $script:Muted
    Write-UiGap $script:MajorGapLines
}

function Write-ToolRow {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Target,
        [ValidateSet("queued", "running", "waiting", "done", "blocked")]
        [string]$State = "done",
        [string]$Details = ""
    )
    $glyph = switch ($State) {
        "queued" { "·" }
        "running" { "≈" }
        "waiting" { "◇" }
        "done" { "✓" }
        "blocked" { "!" }
    }
    $suffix = if ($Details) { " · $Details" } else { "" }
    $line = "  $glyph $Name  $Target$suffix"
    $color = switch ($State) {
        "running" { $script:Tide }
        "waiting" { $script:Yellow }
        "done" { $script:Green }
        "blocked" { $script:Red }
        default { $script:Muted }
    }
    Write-Ui (Fit-UiText $line $TerminalWidth) $color
}

function Show-ToolRunning {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Target,
        [double]$Phase = 0.0
    )
    if (-not $script:UseMotion) {
        Write-ToolRow $Name $Target "running" "active"
        return
    }
    for ($frame = 0; $frame -lt 8; $frame++) {
        $progress = [Math]::Min(1.0, $frame / 3.0)
        $amplitude = 0.72 * $progress * $progress * (3.0 - (2.0 * $progress))
        $wave = Get-RegularTideLine 16 ($Phase + ($frame * 0.52)) $amplitude
        $line = Fit-UiText ("  ≈ $Name  $Target  $wave") $TerminalWidth
        Write-MotionLine "$($script:Tide)$line$($script:Reset)" $TerminalWidth
        Start-Sleep -Milliseconds 125
    }
    Clear-MotionLine $TerminalWidth
}

function Show-DiffPreview {
    Write-Ui (T "  Proposed patch · preview only" "  拟议修改 · 仅预览") $script:Accent
    Write-Ui (T "    scripts/tina-cli-demo.ps1" "    scripts/tina-cli-demo.ps1") $script:Moon
    Write-Ui "    - closed reflection line" $script:Lilac
    Write-Ui "    + right-open water flow" $script:Green
    Write-Ui "    + diff retained after the edit" $script:Green
    Write-Ui (T "  diff summary: +2 / -1 · no file was written" "  diff 摘要：+2 / -1 · 没有写入文件") $script:Muted
}

function Write-EditTop {
    param([Parameter(Mandatory = $true)][string]$Target)
    $label = " edit · $(Fit-UiText $Target ([Math]::Max(12, $TerminalWidth - 18))) "
    $fill = [Math]::Max(0, $TerminalWidth - 3 - (Get-CellWidth $label))
    Write-UiInline "╭─" $script:EditBorder
    Write-UiInline $label $script:Accent
    Write-Ui (("─" * $fill) + "╮") $script:EditBorder
}

function Get-EditBodyParts {
    param(
        [int]$LineNumber,
        [ValidateSet(" ", "+", "-")][string]$Marker,
        [Parameter(Mandatory = $true)][string]$Text
    )
    $bodyWidth = [Math]::Max(12, $TerminalWidth - 4)
    $number = if ($LineNumber -gt 0) { ("{0,4}" -f $LineNumber) } else { "    " }
    $content = Fit-UiText ("$number $Marker $Text") $bodyWidth
    $padding = [Math]::Max(0, $bodyWidth - (Get-CellWidth $content))
    return [pscustomobject]@{
        Content = $content
        Padding = $padding
    }
}

function Write-EditBodyLine {
    param(
        [int]$LineNumber,
        [ValidateSet(" ", "+", "-")][string]$Marker,
        [Parameter(Mandatory = $true)][string]$Text,
        [string]$ChangedText = ""
    )
    $parts = Get-EditBodyParts $LineNumber $Marker $Text
    $content = $parts.Content + (" " * $parts.Padding)
    $color = switch ($Marker) {
        "+" { $script:DiffGreen }
        "-" { $script:DiffRed }
        default { $script:Muted }
    }
    if ($script:UseColor -and $ChangedText -and $content.Contains($ChangedText)) {
        # Oh My Pi uses the diff foreground plus inverse on the changed token.
        # On dark terminals this becomes black text over a red/green swatch.
        $highlight = "$($script:Inverse)$ChangedText$($script:NoInverse)"
        $content = $content.Replace($ChangedText, $highlight)
    }
    Write-UiInline "│ " $script:EditBorder
    Write-UiInline $content $color
    Write-Ui " │" $script:EditBorder
}

function Write-EditBottom {
    param(
        [int]$Added,
        [int]$Removed,
        [Parameter(Mandatory = $true)][string]$State
    )
    $right = " $($script:State.Context)/258K "
    $prefix = "─ +$Added -$Removed · "
    $stateWidth = [Math]::Max(
        1,
        $TerminalWidth - 2 - (Get-CellWidth $right) - (Get-CellWidth $prefix) - 1
    )
    $left = "$prefix$(Fit-UiText $State $stateWidth) "
    $fill = [Math]::Max(0, $TerminalWidth - 2 - (Get-CellWidth $left) - (Get-CellWidth $right))
    Write-UiInline "╰" $script:EditBorder
    Write-UiInline "─" $script:EditBorder
    Write-UiInline $left.Substring(1) $script:DeepWater
    Write-UiInline ("─" * $fill) $script:EditBorder
    Write-UiInline $right $script:DeepWater
    Write-Ui "╯" $script:EditBorder
}

function Write-WorkSurfaceTop {
    param([Parameter(Mandatory = $true)][string]$Label)
    $label = " $(Fit-UiText $Label ([Math]::Max(12, $TerminalWidth - 6))) "
    $fill = [Math]::Max(0, $TerminalWidth - 3 - (Get-CellWidth $label))
    Write-UiInline "╭─" $script:Border
    Write-UiInline $label $script:Accent
    Write-Ui (("─" * $fill) + "╮") $script:Border
}

function Write-WorkSurfaceColumns {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Left,
        [AllowEmptyString()][string]$Right = "",
        [string]$LeftColor = "",
        [string]$RightColor = ""
    )
    $innerWidth = [Math]::Max(16, $TerminalWidth - 6)
    $right = Fit-UiText $Right ([Math]::Max(0, $innerWidth - 2))
    $separatorWidth = if ($right) { 1 } else { 0 }
    $left = Fit-UiText $Left ([Math]::Max(1, $innerWidth - (Get-CellWidth $right) - $separatorWidth))
    $gap = [Math]::Max(0, $innerWidth - (Get-CellWidth $left) - (Get-CellWidth $right))
    Write-UiInline "│  " $script:Border
    Write-UiInline $left $LeftColor
    Write-UiInline (" " * $gap)
    if ($right) { Write-UiInline $right $RightColor }
    Write-Ui "  │" $script:Border
}

function Write-WorkSurfaceBottom {
    param(
        [Parameter(Mandatory = $true)][string]$Summary,
        [Parameter(Mandatory = $true)][string]$Right
    )
    $right = " $Right "
    $tailWidth = 2
    $available = [Math]::Max(1, $TerminalWidth - 2 - $tailWidth - (Get-CellWidth $right))
    $summary = Fit-UiText $Summary ([Math]::Max(1, $available - 3))
    $left = "─ $summary "
    $fill = [Math]::Max(0, $available - (Get-CellWidth $left))
    Write-UiInline "╰" $script:Border
    Write-UiInline "─" $script:Border
    Write-UiInline $left.Substring(1) $script:DeepWater
    Write-UiInline ("─" * $fill) $script:Border
    Write-UiInline $right $script:DeepWater
    Write-UiInline ("─" * $tailWidth) $script:Border
    Write-Ui "╯" $script:Border
}

function Show-EditDiff {
    param([Parameter(Mandatory = $true)][string]$State)
    $target = "scripts/tina-cli-demo.ps1"
    Write-EditTop $target
    Write-EditBodyLine 238 " " '$phase += 0.52'
    Write-EditBodyLine 239 "-" '$flow = Get-FlowLine $flowWidth $phase' 'Get-FlowLine'
    Write-EditBodyLine 239 "+" '$flow = Get-RegularTideLine $flowWidth $phase $amplitude' 'Get-RegularTideLine'
    Write-EditBodyLine 240 "+" '$amplitude = calm + raised_cosine_envelope' 'raised_cosine_envelope'
    Write-EditBodyLine 241 " " 'Write-Ui $line $script:Tide'
    Write-EditBottom 2 1 $State
}

function Show-EditShowcase {
    Write-UiGap $script:MajorGapLines
    Write-ToolRow "edit" "scripts/tina-cli-demo.ps1" "queued" "full access · diff retained"
    Show-ToolRunning "edit" "scripts/tina-cli-demo.ps1" 1.6
    Write-ToolRow "edit" "scripts/tina-cli-demo.ps1" "done" "applied · 18 ms"
    Write-UiGap $script:MajorGapLines
    Show-EditDiff (T "applied · diff archived" "已应用 · diff 已归档")
    $policy = T `
        "Ordinary edits run directly; only rm/delete file operations stop at the environment gate." `
        "普通编辑直接执行；只有 rm/delete 删除文件会在环境边界停下拦截。"
    foreach ($line in (Split-UiText $policy ([Math]::Max(16, $TerminalWidth - 2)))) {
        Write-Ui "  $line" $script:Muted
    }
    Write-UiGap $script:ToolGapLines
}

function Show-ToolShowcase {
    Write-UiGap $script:MajorGapLines
    Write-Ui (T "  Agent trace" "  Agent 轨迹") $script:Accent
    Write-ToolRow "search" "crates/cq-infer/src/tina_cli.rs" "queued" "workspace scan"
    Show-ToolRunning "search" "crates/cq-infer/src/tina_cli.rs" 0.4
    Write-ToolRow "search" "crates/cq-infer/src/tina_cli.rs" "done" "12 matches · 24 ms"
    Show-ToolRunning "patch" "scripts/tina-cli-demo.ps1" 2.1
    Write-ToolRow "patch" "scripts/tina-cli-demo.ps1" "done" "applied · diff retained"
    Show-DiffPreview
    # Deletion is deliberately motionless. The user must be able to distinguish
    # the environment safety gate from an active tool without reading animation.
    Write-ToolRow "rm" "build/obsolete.bin" "blocked" "DELETE confirmation required"
    Write-Ui (T "  delete paused · no file was removed" "  删除已暂停 · 没有移除任何文件") $script:Yellow
    Write-UiGap $script:ToolGapLines
}

function Show-AssistantFrameTop {
    $label = " $($script:State.AssistantId) ☾ "
    $fill = [Math]::Max(0, $TerminalWidth - 3 - (Get-CellWidth $label))
    Write-UiInline "╭─" $script:TinaBorder
    Write-UiInline $label $script:Reply
    Write-Ui (("─" * $fill) + "╮") $script:TinaBorder
}

function Write-StreamingPreview {
    param([Parameter(Mandatory = $true)][string]$Text)
    $bodyWidth = [Math]::Max(16, $TerminalWidth - 6)
    $lines = [System.Collections.Generic.List[string]]::new()
    foreach ($paragraph in [regex]::Split($Text, '\r?\n')) {
        if ($paragraph.Length -eq 0) {
            $lines.Add("")
            continue
        }
        foreach ($wrappedLine in (Split-UiText $paragraph $bodyWidth)) {
            $lines.Add($wrappedLine)
        }
    }
    foreach ($line in $lines) {
        Write-UiInline "│  " $script:TinaBorder
        $padding = [Math]::Max(0, $bodyWidth - (Get-CellWidth $line))
        if (-not $script:UseMotion) {
            Write-UiInline ("$line" + (" " * $padding)) $script:Reply
            Write-Ui "  │" $script:TinaBorder
            continue
        }
        Write-Host $script:Reply -NoNewline
        foreach ($character in $line.ToCharArray()) {
            Write-Host $character -NoNewline
            if (-not [char]::IsWhiteSpace($character)) {
                Start-Sleep -Milliseconds 7
            }
        }
        Write-Host (" " * $padding) -NoNewline
        Write-Host "$($script:Reset)$($script:TinaBorder)  │$($script:Reset)"
    }
}

function Write-MotionLine {
    param(
        [Parameter(Mandatory = $true)][AllowEmptyString()][string]$Text,
        [int]$Width = 0
    )
    $visibleWidth = Get-CellWidth $Text
    $padding = if ($Width -gt $visibleWidth) { " " * ($Width - $visibleWidth) } else { "" }
    # A single write plus carriage return lets the terminal repaint in place.
    [Console]::Write("`r$Text$padding")
}

function Clear-MotionLine {
    param([int]$Width)
    $blank = " " * [Math]::Max(1, $Width)
    [Console]::Write("`r$blank`r")
}

function Rewind-MotionLine {
    [Console]::Write("`r")
}

function Write-TideBorderFrame {
    param([Parameter(Mandatory = $true)][string]$Flow)
    # Keep the animated wave expressive while reserving the structural border
    # glyphs for the same moon-white color as every other UI surface.
    $frame = "$($script:TinaBorder)╰─$($script:Reset)$($script:Tide)$Flow$($script:Reset)$($script:TinaBorder)─╯$($script:Reset)"
    Write-MotionLine $frame ((Get-CellWidth $Flow) + 4)
}

function Show-TurnReflection {
    $status = "preview · 44.1 tok/s · $($script:State.Context)/258K"
    if ($WaterFlow -eq "right") {
        if ($script:UseMotion) {
            $flowWidth = [Math]::Max(1, $TerminalWidth - 4)
            $phase = 3.8
            $amplitudes = @(0.00, 0.08, 0.30, 0.62, 0.88, 1.00, 1.00, 1.00, 1.00, 0.74, 0.42, 0.16, 0.00)
            for ($frame = 0; $frame -lt $amplitudes.Count; $frame++) {
                $flow = Get-RegularTideLine $flowWidth $phase $amplitudes[$frame]
                Write-TideBorderFrame $flow
                $phase += 0.52
                if ($frame -lt 9) {
                    Start-Sleep -Milliseconds 110
                } else {
                    Start-Sleep -Milliseconds (145 + (($frame - 9) * 45))
                }
            }
            Rewind-MotionLine
        }
        $left = "─ preview · 44.1 tok/s "
        $right = " $($script:State.Context)/258K "
        $tailWidth = 2
        $fill = [Math]::Max(0, $TerminalWidth - 2 - $tailWidth - (Get-CellWidth $left) - (Get-CellWidth $right))
        Write-UiInline "╰" $script:TinaBorder
        Write-UiInline "─" $script:TinaBorder
        Write-UiInline $left.Substring(1) $script:Muted
        Write-UiInline ("─" * $fill) $script:TinaBorder
        Write-UiInline $right $script:Moon
        Write-UiInline ("─" * $tailWidth) $script:TinaBorder
        Write-Ui "╯" $script:TinaBorder
    } else {
        $tideWidth = 12
        $status = " $status "
        $used = 2 + $tideWidth + 1 + (Get-CellWidth $status) + 1
        $fill = [Math]::Max(0, $TerminalWidth - $used)
        Write-UiInline "╰─" $script:TinaBorder
        Write-UiInline (Get-TideLine $tideWidth 4.2 0.68) $script:Tide
        Write-UiInline " "
        Write-UiInline $status $script:Muted
        Write-Ui (("─" * $fill) + "╯") $script:TinaBorder
    }
    Write-UiGap $script:MajorGapLines
}

function Show-ContinuousTidePreview {
    if (-not $script:UseMotion) {
        throw "-TideLoop requires an interactive terminal with motion enabled."
    }

    $script:State.Context = 77
    Write-Ui ""
    Show-AssistantFrameTop
    $sample = T `
        "Moon-white text stays still while the tide keeps a calm, regular rhythm." `
        "月白色正文保持静止，底部潮水以平稳、规则的节奏持续起落。"
    Write-StreamingPreview $sample

    $flowWidth = [Math]::Max(1, $TerminalWidth - 4)
    $phase = 0.0
    $frame = 0
    $oldControlMode = [Console]::TreatControlCAsInput
    [Console]::TreatControlCAsInput = $true
    try {
        while ($true) {
            if ([Console]::KeyAvailable) {
                $key = [Console]::ReadKey($true)
                $controlC =
                    $key.Key -eq [ConsoleKey]::C -and
                    ($key.Modifiers -band [ConsoleModifiers]::Control)
                if ($controlC) { break }
            }

            # Phase always advances, while a slow raised-cosine envelope makes
            # the same tide emerge and settle repeatedly. The 40% floor keeps
            # a visible travelling ripple even at the calm point.
            $cycleFrames = 36.0
            $cyclePosition = ($frame % [int]$cycleFrames) / $cycleFrames
            $envelope = 0.5 - (0.5 * [Math]::Cos(2.0 * [Math]::PI * $cyclePosition))
            $amplitude = 0.40 + (0.60 * $envelope)
            $flow = Get-RegularTideLine $flowWidth $phase $amplitude
            Write-TideBorderFrame $flow
            Start-Sleep -Milliseconds 110
            $phase += 0.52
            $frame++
        }
    } finally {
        [Console]::TreatControlCAsInput = $oldControlMode
        Clear-MotionLine $TerminalWidth
    }

    $left = "─ preview · 44.1 tok/s "
    $right = " $($script:State.Context)/258K "
    $tailWidth = 2
    $fill = [Math]::Max(0, $TerminalWidth - 2 - $tailWidth - (Get-CellWidth $left) - (Get-CellWidth $right))
    Write-UiInline "╰" $script:TinaBorder
    Write-UiInline "─" $script:TinaBorder
    Write-UiInline $left.Substring(1) $script:Muted
    Write-UiInline ("─" * $fill) $script:TinaBorder
    Write-UiInline $right $script:Moon
    Write-UiInline ("─" * $tailWidth) $script:TinaBorder
    Write-Ui "╯" $script:TinaBorder
    Write-UiGap $script:MajorGapLines
}

function Show-Welcome {
    if (-not $NoClear) {
        try { Clear-Host } catch { }
    }
    [void](Show-StartupArt)
    $rows = @(
        (T "offline · no model · no tools" "离线 · 无模型 · 无工具"),
        "0731-main · single-sequence · no-MTP/no-D-Spark",
        (T `
            ("{0} · {1} · 0/258K · cache off · think off" -f $script:State.OS, $script:State.Shell) `
            ("{0} · {1} · 0/258K · 缓存 off · 思考 off" -f $script:State.OS, $script:State.Shell))
    )
    foreach ($line in Get-WelcomeFrameLines "TINA ENGINE · $($script:State.AssistantId)" "UI PREVIEW" $rows) {
        Write-Ui $line $script:MoonBright
    }
    Write-UiGap $script:MajorGapLines
}

function Show-Help {
    Write-Rule
    Write-Ui (T "Core commands" "核心指令") $script:Accent
    Write-Ui (T "  /agent                 Agent status and access mode" "  /agent                 Agent 状态与权限模式")
    Write-Ui (T "  /goal [set|clear]      Session goal" "  /goal [set|clear]      会话目标")
    Write-Ui (T "  /thinking [level]      off|medium|high|max" "  /thinking [级别]       off|medium|high|max")
    Write-Ui (T "  /status                Runtime, context, and cache" "  /status                运行、上下文与缓存状态")
    Write-Ui (T "  /help                  Command list" "  /help                  指令列表")
    Write-Ui (T "  /exit                  Close the session" "  /exit                  关闭会话")
    Write-Ui ""
    Write-Ui (T "Product controls" "产品控制") $script:Accent
    Write-Ui (T "  /context  /environment  /assistant [id]  /language [en|zh]" "  /context  /environment  /assistant [id]  /language [en|zh]")
    Write-Ui (T "  /todo  /diff  /review  /plugins  /tools" "  /todo  /diff  /review  /plugins  /tools")
    Write-Ui (T "  /sessions  /resume  /retry  /new  /compact  /export" "  /sessions  /resume  /retry  /new  /compact  /export")
    Write-Ui (T "  /memory  /remember  /project-memory" "  /memory  /remember  /project-memory")
    Write-Ui (T "  /delete <path>         Safety interceptor preview" "  /delete <路径>         删除安全拦截预览")
    Write-Rule
}

function Show-Status {
    Write-Rule
    Write-Ui (T "Tina session status" "Tina 会话状态") $script:Accent
    Write-Row (T "engine" "引擎") "fast (demo)" $script:Green
    Write-Row (T "model" "模型") "DeepSeek-V4-Flash-0731 · CQTensors" $script:Muted
    Write-Row (T "device" "设备") "simulation · no CUDA context" $script:Muted
    Write-Row (T "thinking" "思考") $script:State.Thinking
    Write-Row (T "context" "上下文") "$($script:State.Context) / 258K · normal" $script:Green
    Write-Row (T "KV cache" "KV 缓存") "off · exact full-prompt opt-in" $script:Muted
    Write-Row (T "Agent" "Agent") $script:State.Agent $script:Green
    Write-Row (T "session" "会话") $script:State.Session $script:Muted
    Write-Rule
}

function Show-Agent {
    Write-Rule
    Write-Ui (T "Agent preview" "Agent 预览") $script:Accent
    Write-Row (T "host" "主机") "$($script:State.Agent) (demo)" $script:Green
    Write-Row (T "scope" "权限") "workspace-bounded" $script:Muted
    Write-Row (T "tools" "工具") "read · search · git · patch · write · mkdir · rename · delete" $script:Muted
    Write-Row (T "network" "网络") "off by default · explicit provider only" $script:Muted
    Write-Row (T "delete" "删除") "intercepted · requires DELETE" $script:Yellow
    Write-Ui (T "Use /agent full to preview the warning; this demo never grants access." "输入 /agent full 可预览警告；此 demo 永不实际开放权限。") $script:Yellow
    Write-Rule
}

function Show-Environment {
    Write-Rule
    Write-Ui (T "Injected terminal environment" "注入模型的终端环境") $script:Accent
    Write-Row "OS" $script:State.OS
    Write-Row "shell" $script:State.Shell
    $terminalName = if ($env:TERM) { $env:TERM } else { "Windows Terminal / PowerShell" }
    Write-Row "terminal" $terminalName
    Write-Row "cwd" (Get-Location).Path
    $languageName = if ($env:TINA_LANG) { $env:TINA_LANG } else { "auto" }
    Write-Row "TINA_LANG" $languageName
    Write-Rule
}

function Show-Todos {
    Write-Ui ""
    Write-WorkSurfaceTop (T "Todo · Tina beta" "Todo · Tina beta")
    Write-WorkSurfaceColumns `
        (T "✓ 01  Freeze CLI interaction surface" "✓ 01  冻结 CLI 交互界面") `
        (T "done" "完成") $script:Green $script:Muted
    Write-WorkSurfaceColumns `
        (T "✓ 02  Close unsafe trust boundaries" "✓ 02  闭合 unsafe 信任边界") `
        (T "done" "完成") $script:Green $script:Muted
    Write-WorkSurfaceColumns `
        (T "≈ 03  Connect cache-hit product path" "≈ 03  接入缓存命中产品路径") `
        (T "active  ~≈~" "进行中  ~≈~") $script:Reply $script:Tide
    Write-WorkSurfaceColumns `
        (T "· 04  Run public-source qualification" "· 04  运行公开源码资格测试") `
        (T "pending" "待处理") $script:Muted $script:Muted
    Write-WorkSurfaceColumns `
        (T "· 05  Publish the beta release" "· 05  发布 Beta 版本") `
        (T "pending" "待处理") $script:Muted $script:Muted
    $summary = T "2/5 done · 1 active · 2 pending" "完成 2/5 · 进行中 1 · 待处理 2"
    Write-WorkSurfaceBottom $summary "$($script:State.Context)/258K"
    Write-Ui ""
}

function Show-GitWorkspace {
    Write-Ui ""
    Write-WorkSurfaceTop "Git · feature/cli-agent-kv-persistence"
    Write-WorkSurfaceColumns "M  scripts/tina-cli-demo.ps1" (T "unstaged" "未暂存") $script:Yellow $script:Yellow
    Write-WorkSurfaceColumns "HEAD 68b40234" "feat(cli): refine Tina startup surface" $script:Muted $script:Moon
    Write-WorkSurfaceColumns (T "+2 additions" "+2 新增") (T "-1 deletion" "-1 删除") $script:DiffGreen $script:DiffRed
    $summary = T "preview · dirty · staged 0 · unstaged 1" "预览 · 有修改 · 已暂存 0 · 未暂存 1"
    Write-WorkSurfaceBottom $summary "$($script:State.Context)/258K"
    Write-Ui ""
}

function Show-Plugins {
    Write-Rule
    Write-Ui (T "Plugins" "插件") $script:Accent
    Write-Ui "  ccswitch        loopback API bridge       ready (demo)" $script:Green
    Write-Ui "  /v1/anthropic   Anthropic-compatible route ready (demo)" $script:Green
    Write-Ui "  /v1/openai      OpenAI-compatible route    ready (demo)" $script:Green
    Write-Ui "  image-ocr       zh/en/ja extraction        preview" $script:Muted
    Write-Rule
}

function Show-Sessions {
    Write-Rule
    Write-Ui (T "Conversation archives" "会话归档") $script:Accent
    Write-Row (T "active" "当前") "$($script:State.Session) · epoch 1" $script:Green
    Write-Row (T "latest" "最新") "$($script:State.Session) · demo archive" $script:Muted
    Write-Ui (T "Use /resume latest or a full archive ID in the product CLI." "产品 CLI 可使用 /resume latest 或完整归档 ID。") $script:Muted
    Write-Rule
}

function Show-Memory {
    Write-Rule
    Write-Ui (T "Conversation memory" "会话记忆") $script:Accent
    Write-Row (T "pinned facts" "固定事实") "0 · preview"
    Write-Row (T "archive" "归档") "append-only · retained" $script:Green
    Write-Ui (T "This model-free demo does not persist facts." "这个无模型 demo 不会持久化事实。") $script:Muted
    Write-Rule
}

function Show-ProjectMemory {
    Write-Rule
    Write-Ui (T "Git-backed project memory" "Git 持久化项目记忆") $script:Accent
    Write-Row (T "workspace" "工作区") (Get-Location).Path $script:Muted
    Write-Row (T "entries" "条目") "0 · preview"
    Write-Row (T "delete" "删除") "intercepted · DELETE required" $script:Yellow
    Write-Rule
}

function Show-ReviewPreview {
    Write-Ui (T "  Read-only review · preview" "  只读审查 · 预览") $script:Accent
    Show-DiffPreview
    Write-Ui (T "  No tool or file was changed." "  没有调用工具，也没有修改文件。") $script:Muted
}

function Confirm-DeletePreview {
    param([string]$Target)
    Write-Ui (T "DELETE INTERCEPT" "删除拦截") $script:Yellow
    Write-Ui (T "Requested path: $Target" "请求路径：$Target") $script:Yellow
    Write-Ui (T "This is destructive. Type DELETE to confirm (demo never deletes files):" "这是破坏性操作。输入 DELETE 确认（demo 永不删除文件）：") $script:Yellow
    if ($script:CommandQueue.Count -gt 0) {
        Write-Ui (T "Blocked: interactive confirmation is required." "已拦截：必须在交互终端中确认。") $script:Red
        return
    }
    $confirmation = Read-Host "  >"
    if ($confirmation -ceq "DELETE") {
        Write-Ui (T "Demo accepted the confirmation, but performed no deletion." "demo 接受了确认，但没有执行删除。") $script:Green
    } else {
        Write-Ui (T "Blocked. No file was changed." "已拦截。没有修改任何文件。") $script:Green
    }
}

function Handle-Command {
    param([string]$Line)
    $match = [regex]::Match($Line.Trim(), '^(?<name>\S+)(?:\s+(?<arg>.*))?$')
    $name = $match.Groups["name"].Value.ToLowerInvariant()
    $arg = $match.Groups["arg"].Value.Trim()
    $aliases = @{
        "/commands" = "/help"; "/帮助" = "/help"; "/指令" = "/help"
        "/代理" = "/agent"; "/目标" = "/goal"; "/思考" = "/thinking"
        "/状态" = "/status"; "/上下文" = "/context"; "/env" = "/environment"
        "/环境" = "/environment"; "/终端" = "/environment"
        "/name" = "/assistant"; "/助手" = "/assistant"; "/语言" = "/language"
        "/待办" = "/todo"; "/任务" = "/todo"; "/差异" = "/diff"
        "/审查" = "/review"; "/插件" = "/plugins"; "/工具" = "/tools"
        "/会话" = "/sessions"; "/恢复" = "/resume"; "/重试" = "/retry"
        "/reset" = "/new"; "/clear" = "/new"; "/新会话" = "/new"
        "/重置" = "/new"; "/清空" = "/new"; "/压缩" = "/compact"
        "/导出" = "/export"; "/记忆" = "/memory"; "/记住" = "/remember"
        "/项目记忆" = "/project-memory"; "/项目记住" = "/project-remember"
        "/项目忘记" = "/project-forget"; "/重启工具" = "/agent-restart"
        "/退出" = "/exit"
    }
    if ($aliases.ContainsKey($name)) { $name = $aliases[$name] }
    switch ($name) {
        "/" { return $true }
        "/help" { Show-Help; return $true }
        "/agent" { if ($arg -eq "full") { Write-Ui (T "FULL_ACCESS warning: demo only; no access is granted." "FULL_ACCESS 警告：仅演示，不会开放权限。") $script:Yellow }; Show-Agent; return $true }
        "/goal" { if ($arg -match "^(?i)set\s+(.+)$") { $script:State.Goal = $Matches[1]; Write-Ui (T "Goal saved for this demo session." "目标已保存到本 demo 会话。") $script:Green } elseif ($arg -match "^(?i)clear$") { $script:State.Goal = $null; Write-Ui (T "Goal cleared." "目标已清除。") $script:Green } elseif ($script:State.Goal) { Write-Ui ("  " + $script:State.Goal) $script:Moon } else { Write-Ui (T "No goal set. Use /goal set <text>." "尚未设置目标。使用 /goal set <目标>。") $script:Muted }; return $true }
        "/thinking" { if ($arg) { Set-Thinking $arg } else { Write-Ui (T "Thinking level: $($script:State.Thinking)." "思考级别：$($script:State.Thinking)。") $script:Moon }; return $true }
        "/status" { Show-Status; return $true }
        "/context" { Write-Rule; Write-Row (T "context" "上下文") "0 / 258K · normal" $script:Green; Write-Row (T "handoff" "交接") "not needed" $script:Muted; Write-Rule; return $true }
        "/environment" { Show-Environment; return $true }
        "/assistant" { if ($arg) { Set-AssistantId $arg } else { Write-Ui (T "Assistant ID: $($script:State.AssistantId)" "助手 ID：$($script:State.AssistantId)") $script:Moon }; return $true }
        "/language" { if ($arg -in @("en", "zh")) { $script:State.Language = $arg; Write-Ui (T "UI language switched." "界面语言已切换。") $script:Green } else { Write-Ui (T "Use: /language en|zh" "用法：/language en|zh") $script:Muted }; return $true }
        "/todo" { Show-Todos; return $true }
        "/diff" { Show-GitWorkspace; return $true }
        "/review" { Show-ReviewPreview; return $true }
        "/plugins" { Show-Plugins; return $true }
        "/tools" { Show-Agent; return $true }
        "/agent-restart" { Write-Ui (T "Agent tool-host restart is available in the product CLI; this demo has no host." "产品 CLI 支持重启 Agent tool-host；此 demo 没有 host。") $script:Muted; return $true }
        "/sessions" { Show-Sessions; return $true }
        "/resume" { if ($arg) { Write-Ui (T "Resume preview: '$arg' would close the current owner before switching archives." "恢复预览：切换到 '$arg' 前会先关闭当前 owner。") $script:Green } else { Write-Ui (T "Use: /resume latest|<full archive ID>" "用法：/resume latest|<完整归档 ID>") $script:Muted }; return $true }
        "/retry" { Write-Ui (T "Retry preview: a new immutable branch would be created from the last user turn." "重试预览：会从上一用户回合创建新的不可变分支。") $script:Green; return $true }
        "/memory" { Show-Memory; return $true }
        "/remember" { if ($arg) { Write-Ui (T "Memory preview: this fact would be pinned to the conversation archive." "记忆预览：该事实会固定到会话归档。") $script:Green } else { Write-Ui (T "Use: /remember <fact>" "用法：/remember <事实>") $script:Muted }; return $true }
        "/project-memory" { Show-ProjectMemory; return $true }
        "/project-remember" { if ($arg) { Write-Ui (T "Project-memory preview: Git-backed persistence is disabled in this demo." "项目记忆预览：此 demo 未启用 Git 持久化。") $script:Green } else { Write-Ui (T "Use: /project-remember <fact>" "用法：/project-remember <事实>") $script:Muted }; return $true }
        "/project-forget" { if ($arg) { Confirm-DeletePreview $arg } else { Write-Ui (T "Use: /project-forget <ID>" "用法：/project-forget <ID>") $script:Muted }; return $true }
        "/export" { if ($arg) { Write-Ui (T "Export preview: '$arg' was not written by this demo." "导出预览：此 demo 未写入 '$arg'。") $script:Green } else { Write-Ui (T "Use: /export [jsonl|md] <path>" "用法：/export [jsonl|md] <路径>") $script:Muted }; return $true }
        "/delete" { if ($arg) { Confirm-DeletePreview $arg } else { Write-Ui (T "Use: /delete <path>" "用法：/delete <路径>") $script:Muted }; return $true }
        "/new" { $script:State.Session = "demo-" + (Get-Random -Minimum 1000 -Maximum 9999); $script:State.Context = 0; Write-Ui (T "New session opened; previous archive retained." "已打开新会话；旧归档保留。") $script:Green; return $true }
        "/compact" { Write-Ui (T "Demo handoff created at 95% policy boundary (no model state changed)." "已在 95% 策略边界创建 demo 交接（未改变模型状态）。") $script:Green; return $true }
        "/exit" { return $false }
        "/quit" { return $false }
        ":q" { return $false }
        default { Write-Ui (T "Unknown command. Use /help." "未知指令。输入 /help 查看帮助。") $script:Red; return $true }
    }
}

function Handle-Message {
    param([string]$Message)
    Write-UiGap $script:MajorGapLines
    Show-TideWait
    if ($EditShowcase -or ($Demo -and -not $ToolShowcase)) {
        Show-EditShowcase
    } elseif ($ToolShowcase) {
        Show-ToolShowcase
    } else {
        Write-ToolPreview
    }
    Show-AssistantFrameTop
    $response = T `
        "The artwork appears once at startup, then yields the screen to the task. Turn metrics feel like a reflection settling quietly beneath the answer." `
        "原画只在启动时出现一次，随后把屏幕完整让给任务。回合指标则像倒影一样安静沉到正文下方。"
    Write-StreamingPreview $response
    $estimatedTokens = [Math]::Max(1, [int][Math]::Ceiling($Message.Length / 3.5))
    $script:State.Context = [Math]::Min(258000, $script:State.Context + $estimatedTokens + 65)
    Show-TurnReflection
    Write-UiGap $script:MajorGapLines
}

$script:State = [ordered]@{
    Language = $Language
    AssistantId = if ([string]::IsNullOrWhiteSpace($AssistantId)) { "Tina" } else { $AssistantId.Trim() }
    Thinking = "off"
    Goal = $null
    Session = "demo-0001"
    Agent = "off"
    Context = 0
    OS = if ($env:OS -eq "Windows_NT" -or [Environment]::OSVersion.Platform -eq [PlatformID]::Win32NT) { "Windows" } else { "Unix" }
    Shell = "PowerShell"
}
$script:CommandQueue = [System.Collections.Generic.Queue[string]]::new()
if ($Demo) {
    if ($CommandShowcase) {
        $script:CommandQueue.Enqueue("/")
        $script:CommandQueue.Enqueue("/status")
        $script:CommandQueue.Enqueue("/exit")
    } elseif ($TodoShowcase -or $GitShowcase) {
        $script:State.Context = 77
        if ($TodoShowcase) { $script:CommandQueue.Enqueue("/todo") }
        if ($GitShowcase) { $script:CommandQueue.Enqueue("/diff") }
        $script:CommandQueue.Enqueue("/exit")
    } else {
        $samplePrompt = if ($Language -eq "zh") {
            "帮我审计这套 CLI，让界面有 Tidal Luna 的气质，但不要妨碍编码；同时检查 Edit、Todo、Git 和上下文状态是否清晰一致。"
        } else {
            "Audit this CLI. Give it a Tidal Luna identity without getting in the way of coding, and verify that Edit, Todo, Git, and context status remain clear and consistent."
        }
        $tour = if ($EditShowcase -or $ToolShowcase) {
            @($samplePrompt, "/exit")
        } else {
            @($samplePrompt, "/todo", "/diff", "/status", "/exit")
        }
        $tour | ForEach-Object { $script:CommandQueue.Enqueue($_) }
    }
} elseif ($null -ne $Commands) {
    $Commands | ForEach-Object { $script:CommandQueue.Enqueue($_) }
}

Show-Welcome
if ($TideLoop) {
    Show-ContinuousTidePreview
    return
}
while ($true) {
    Sync-TerminalWidth
    if ($script:CommandQueue.Count -gt 0) {
        $line = Read-Composer -QueuedLine $script:CommandQueue.Dequeue() -FromQueue
    } else {
        $line = Read-Composer -QueuedLine ""
    }
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.TrimStart().StartsWith("/")) {
        Write-UiGap $script:MajorGapLines
        if (-not (Handle-Command $line)) { break }
        Write-UiGap $script:MajorGapLines
    } else {
        Handle-Message $line
    }
}
Write-Ui (T "Session closed. No model or files were touched." "会话已关闭。没有加载模型，也没有修改文件。") $script:Muted
