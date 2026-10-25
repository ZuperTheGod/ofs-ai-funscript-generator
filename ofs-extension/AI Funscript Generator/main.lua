-- AI Funscript Generator for OpenFunscripter 3
-- Generates a funscript for the open video with the AI (or plain motion tracking), loads it into the selected
-- track, and can check a script for stretches that may be upside down (100 at OUT instead of IN).
-- Needs the generator installed (see the project README); set its folder below once.

Settings = {
    generatorDir = "",
    method = 1,          -- 1 = AI, 2 = motion tracking only
    model = 1,           -- 1 = v4, 2 = v3
    actThr = 0.20,
    anchorConf = 0.65,
    slowHole = 2.2,
    sound = true,
    rest = true,
    beatLock = false,
}
MusicMode = false        -- per video, never saved
State = { busy = false, task = "", poll = 0, status = "", log = "", flags = {}, havePoints = false, video = "" }

local function sep() return package.config:sub(1, 1) end
local function workDir() return ofs.ExtensionDir() .. sep() .. "work" end
local function settingsFile() return ofs.ExtensionDir() .. sep() .. "settings.txt" end

local function exists(p)
    local f = io.open(p, "r")
    if f then f:close() return true end
    return false
end

local function readAll(p)
    local f = io.open(p, "r")
    if not f then return nil end
    local s = f:read("a")
    f:close()
    return s
end

local function jsonStr(s)
    s = s:gsub("\\", "\\\\"):gsub('"', '\\"')
    return '"' .. s .. '"'
end

local function saveSettings()
    local f = io.open(settingsFile(), "w")
    if not f then return end
    for k, v in pairs(Settings) do f:write(k .. "=" .. tostring(v) .. "\n") end
    f:close()
end

local function loadSettings()
    local f = io.open(settingsFile(), "r")
    if not f then return end
    for line in f:lines() do
        local k, v = line:match("^(%w+)=(.*)$")
        if k and Settings[k] ~= nil then
            if type(Settings[k]) == "boolean" then Settings[k] = (v == "true")
            elseif type(Settings[k]) == "number" then Settings[k] = tonumber(v) or Settings[k]
            else Settings[k] = v end
        end
    end
    f:close()
end

local function python()
    return Settings.generatorDir .. sep() .. ".venv" .. sep() .. "Scripts" .. sep() .. "python.exe"
end

local function start(task, scriptPath)
    local video = player.CurrentVideo() or ""
    if video == "" then State.status = "Open a video first." return end
    if Settings.generatorDir == "" or not exists(python()) then
        State.status = "Set the generator folder (where you ran '1 - Setup') first."
        return
    end
    os.execute('mkdir "' .. workDir() .. '" >nul 2>nul')
    local out = workDir()
    local settings = string.format('{"act_thr": %.3f, "anchor_conf": %.3f, "slow_hole_factor": %.2f, "sound": %s, ' ..
        '"rest": %s, "beat_lock": %s, "beat_lock_share": 0.7, "beat_lock_near_ms": 60, "music_mode": %s}',
        Settings.actThr, Settings.anchorConf, Settings.slowHole, tostring(Settings.sound), tostring(Settings.rest),
        tostring(Settings.beatLock), tostring(MusicMode))
    local req = "{" ..
        '"task": ' .. jsonStr(task) .. ', "video": ' .. jsonStr(video) .. ', "out_dir": ' .. jsonStr(out) ..
        ', "settings": ' .. settings .. ', "model": ' .. jsonStr(Settings.model == 2 and "v3" or "v4") ..
        ', "motion_only": ' .. tostring(Settings.method == 2) ..
        ', "script": ' .. jsonStr(scriptPath or "") .. "}"
    local reqPath = out .. sep() .. "request.json"
    local f = io.open(reqPath, "w")
    f:write(req)
    f:close()
    os.remove(out .. sep() .. "done.txt")
    -- OFS's own process API: no console window
    State.proc = Process.new(python(), Settings.generatorDir .. sep() .. "run_ofs_request.py", reqPath)
    if State.proc then State.proc:detach() end
    State.busy, State.task, State.havePoints = true, task, false
    State.status = task == "check" and "Checking direction ..." or "Generating ... (first run of a video takes a few minutes)"
end

local function loadPoints()
    local p = workDir() .. sep() .. "points.txt"
    local f = io.open(p, "r")
    if not f then State.status = "No generated script found." return end
    local script = ofs.Script(ofs.ActiveIdx())
    script.actions:clear()
    local n = 0
    for line in f:lines() do
        local t, pos = line:match("^([%d%.]+)%s+(%d+)")
        if t then
            script.actions:add(Action.new(tonumber(t), tonumber(pos)))
            n = n + 1
        end
    end
    f:close()
    script:commit()
    State.status = string.format("Loaded %d points into track '%s' (Edit > Undo or the Undo button reverts it).",
        n, ofs.ScriptName(ofs.ActiveIdx()) or "?")
end

local function readFlags()
    State.flags = {}
    local s = readAll(workDir() .. sep() .. "flags.txt") or ""
    for a, b, lvl in s:gmatch("([%d%.]+)%s+([%d%.]+)%s+(%S+)") do
        table.insert(State.flags, { start = tonumber(a), stop = tonumber(b), likely = lvl == "likely_flipped" })
    end
end

function init()
    loadSettings()
end

function update(delta)
    local v = player.CurrentVideo() or ""
    if v ~= State.video then
        State.video, MusicMode, State.flags = v, false, {}
    end
    if not State.busy then return end
    State.poll = State.poll + delta
    if State.poll < 0.5 then return end
    State.poll = 0
    local logText = readAll(workDir() .. sep() .. "log.txt")
    if logText then State.log = logText:sub(-1500) end
    local done = readAll(workDir() .. sep() .. "done.txt")
    if done then
        State.busy = false
        if done:find("ok") then
            if State.task == "check" then
                readFlags()
                State.status = #State.flags == 0 and "Direction check: nothing suspicious found." or
                    string.format("Direction check: %d stretches to look at (below).", #State.flags)
            else
                State.havePoints = true
                State.status = "Done. Select an EMPTY track (or a new one) and press 'Load into selected track'."
            end
        else
            State.status = "Failed - see the log below."
        end
    end
end

local function fmt(t) return string.format("%d:%02d", math.floor(t / 60), math.floor(t % 60)) end

function gui()
    local changed
    ofs.Text("Generator folder (where you ran '1 - Setup'):")
    Settings.generatorDir, changed = ofs.Input("##gendir", Settings.generatorDir)
    if changed then saveSettings() end
    ofs.Separator()
    local c
    Settings.method, c = ofs.Combo("Method", Settings.method, { "AI (learned)", "Motion tracking only (loops)" }); changed = c
    Settings.model, c = ofs.Combo("Model", Settings.model, { "v4 (default)", "v3 (previous)" }); changed = changed or c
    Settings.actThr, c = ofs.Slider("Scripting threshold", Settings.actThr, 0.05, 0.7); changed = changed or c
    Settings.anchorConf, c = ofs.Slider("Follow visible strokes", Settings.anchorConf, 0.3, 1.0); changed = changed or c
    Settings.slowHole, c = ofs.Slider("Fill gaps wider than x rhythm", Settings.slowHole, 0.0, 4.0); changed = changed or c
    Settings.sound, c = ofs.Checkbox("Use thrust sounds (videos without music)", Settings.sound); changed = changed or c
    Settings.rest, c = ofs.Checkbox("Rest points (stretches start and end at 0)", Settings.rest); changed = changed or c
    Settings.beatLock, c = ofs.Checkbox("Beat lock (INs on the beat where strokes follow it)", Settings.beatLock); changed = changed or c
    if changed then saveSettings() end
    MusicMode = ofs.Checkbox("MUSIC MODE for this video (every IN on a beat)", MusicMode)
    ofs.Separator()
    if State.busy then
        ofs.Text(State.status)
    else
        if ofs.Button("Generate for this video") then start("generate") end
        if State.havePoints then
            ofs.SameLine()
            if ofs.Button("Load into selected track") then loadPoints() end
            ofs.SameLine()
            if ofs.Button("Undo load") then ofs.Undo() end
        end
        if ofs.Button("Check direction of the selected track (save it first)") then
            start("check", ofs.Script(ofs.ActiveIdx()).path)
        end
        ofs.Text(State.status)
    end
    if #State.flags > 0 then
        ofs.Separator()
        ofs.Text("Stretches that may be upside down (click to jump):")
        for i, f in ipairs(State.flags) do
            if ofs.Button((f.likely and "LIKELY FLIPPED  " or "check  ") .. fmt(f.start) .. " - " .. fmt(f.stop) .. "##f" .. i) then
                player.Seek(f.start)
            end
        end
    end
    if State.log ~= "" then
        ofs.Separator()
        ofs.Text(State.log:match("([^\n]*\n?[^\n]*\n?[^\n]*\n?[^\n]*)$") or "")
    end
end

function scriptChange(scriptIdx)
end
