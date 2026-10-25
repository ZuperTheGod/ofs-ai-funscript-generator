# OFS-AI: OpenFunscripter with the AI Funscript Generator built in

OFS-AI is [OpenFunscripter](https://github.com/OpenFunscripter/OFS) (GPL-3.0) with these additions:

- **AI Generator** menu / panel: generate with every setting, Check direction (red/orange timeline marks + jump list)
- **Why view**: a band on the timeline showing what the AI saw and what decided each stroke
- **Highlights**: right-click the timeline to colour and tag sections (searchable tag gallery), saved as chapters too
- audio classification / voice separation lanes and a CV tracker (earlier additions in this fork)

A ready-built Windows version is in the release zip (`OFS-AI` folder). This folder holds its source changes, as the
GPL requires.

## Building it yourself

Based on OFS commit `5b0c699b84b871eff5534303ce07adeef82679b6`.

```
git clone --recursive https://github.com/OpenFunscripter/OFS
cd OFS
git checkout 5b0c699b84b871eff5534303ce07adeef82679b6
git apply <this folder>/OFS-AI.patch
xcopy /e /i /y "<this folder>\new-files" .
cmake -S . -B build -G "Visual Studio 17 2022"
cmake --build build --config RelWithDebInfo --target OpenFunscripter
```

`OFS-AI.patch` changes existing OFS files; `new-files/` are files OFS-AI adds. The generator itself (Python) is the
rest of this repository; the panel finds it automatically when OFS-AI sits in a folder inside it, otherwise set the
folder in the panel once.
