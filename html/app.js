// Web IDE frontend
console.log('web IDE loaded');

const api = {
  status: () => fetch('/status').then(r=>r.json()),
  // updated to include shots so server can run with requested shots
  run: (code, shots=1024) => fetch('/run', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({code, shots})}).then(r=>r.json()),
  installHelp: () => fetch('/install-help').then(r=>r.json()),
  install: (pkg) => fetch('/install', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({tool:pkg})}).then(r=>r.json()),
  save: (name, content) => fetch('/save', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({path:name, content})}).then(r=>r.json()),
  load: (name) => fetch(`/load?path=${encodeURIComponent(name)}`).then(r=>r.json()),
  list: () => fetch('/list-files').then(r=>r.json())
}

const rawTextarea = document.getElementById('editor')
const output = document.getElementById('output')
const statusOut = document.getElementById('status_out')
const filesList = document.getElementById('files')

let cm = null
let term = null
let xtermAvailable = false

function initEditor(){
  if(typeof CodeMirror === 'undefined'){
    console.warn('CodeMirror not loaded');
    return;
  }
  cm = CodeMirror.fromTextArea(rawTextarea, {
    mode: 'javascript', theme: 'dracula', lineNumbers: true, indentUnit:2, tabSize:2, autofocus:true
  })
  cm.setSize('100%','100%')
}

async function refreshStatus(){
  try{
    const s = await api.status();
    if(statusOut) statusOut.textContent = JSON.stringify(s, null, 2);
  }catch(e){ if(statusOut) statusOut.textContent = 'status error: '+String(e) }
}

async function refreshFiles(){
  try{
    const files = await api.list();
    if(filesList) filesList.innerHTML = '';
    files.forEach(f=>{
      const li = document.createElement('li');
      li.textContent = f;
      li.addEventListener('click', async ()=>{
        const d = await api.load(f);
        if(d && d.content!==undefined){ if(cm) cm.setValue(d.content); else rawTextarea.value = d.content }
      });
      if(filesList) filesList.appendChild(li);
    });
  }catch(e){ console.warn('list-files failed', e); if(output) output.textContent = 'file list failed' }
}

// wire UI
window.addEventListener('load', ()=>{
  initEditor();
  refreshStatus();
  refreshFiles();

  // initialize xterm.js terminal if available
  const xtermContainer = document.getElementById('xterm-container');
  if(window.Terminal && xtermContainer){
    try{
      term = new Terminal({cols: 80, rows: 18, convertEol: true, fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Monaco, "Roboto Mono", monospace', fontSize: 13, theme: { background: '#0b0710', foreground: '#dcd6ff' } });
      term.open(xtermContainer);
      xtermAvailable = true;
      term.writeln('qubo terminal ready');
    }catch(e){ console.warn('xterm init failed', e); xtermAvailable = false }
  }

  // populate gates list (left pane)
  const gates = ['H','X','Y','Z','CNOT','Measure'];
  const gatesEl = document.getElementById('gates');
  if(gatesEl){
    gates.forEach(g=>{
      const li = document.createElement('li');
      li.textContent = g;
      li.addEventListener('click', ()=>{
        let insert = '';
        if(g === 'CNOT') insert = "qc.add_gate('CNOT', targets=[0,1])\n";
        else if(g === 'Measure') insert = "qc.add_gate('M', targets=[0])\n";
        else insert = `qc.add_gate('${g}', targets=[0])\n`;
        if(cm){ const pos = cm.getCursor(); cm.replaceRange(insert, pos); cm.focus(); } else { rawTextarea.value += '\n' + insert }
      });
      gatesEl.appendChild(li);
    })
  }

  // bottom tab switching
  document.querySelectorAll('.tab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
      btn.classList.add('active');
      const tab = btn.getAttribute('data-tab');
      document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
      const sel = document.getElementById('panel-'+tab);
      if(sel) sel.classList.add('active');
    })
  })

  // open/save simple handlers
  const openBtn = document.getElementById('open');
  const saveBtn = document.getElementById('save');
  if(saveBtn){ saveBtn.addEventListener('click', async ()=>{
    const name = prompt('Save as', 'script.py'); if(!name) return;
    const content = cm ? cm.getValue() : rawTextarea.value;
    const r = await api.save(name, content);
    if(output) output.textContent = JSON.stringify(r, null, 2);
    refreshFiles();
  }) }
  if(openBtn){ openBtn.addEventListener('click', async ()=>{
    const name = prompt('Open file', 'script.py'); if(!name) return;
    const d = await api.load(name);
    if(d && d.content!==undefined){ if(cm) cm.setValue(d.content); else rawTextarea.value = d.content }
  }) }

  const btnDemo = document.getElementById('btn_demo');
  if(btnDemo){ btnDemo.addEventListener('click', ()=>{
    const demo = `from qubo import *\n\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nprint(qc)\n`;
    if(cm) cm.setValue(demo); else rawTextarea.value = demo;
  }) }

  // demo small button
  const demoBtn = document.getElementById('btn_demo_small');
  if(demoBtn){ demoBtn.addEventListener('click', ()=>{
    const demo = `from qubo import *\n\nqc = QuantumCircuit(2)\nqc.h(0)\nqc.cx(0,1)\nprint(qc)\n`;
    if(cm) cm.setValue(demo); else rawTextarea.value = demo;
  }) }

  // run handlers (top and simple)
  const runTop = document.getElementById('btn_run_top');
  const runSimple = document.getElementById('btn_run');
  const graphCanvas = document.getElementById('graph_canvas');

  // WebSocket real-time terminal support (client stub)
  let ws = null;
  function startWsRun(code, shots){
    if(!('WebSocket' in window)) return false;
    try{
      ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws-run');
      if(xtermAvailable && term) term.reset();
      ws.onopen = ()=>{
        ws.send(JSON.stringify({code, shots}));
      };
      ws.onmessage = (ev)=>{
        try{ const obj = JSON.parse(ev.data); if(obj.type === 'stdout'){ if(xtermAvailable) term.write(obj.data.replace(/\n/g,'\r\n')); else { const fb = document.getElementById('terminal_out'); if(fb) fb.textContent += obj.data } } else if(obj.type === 'done'){ if(xtermAvailable) term.writeln('\r\n[done]'); } }catch(e){ if(xtermAvailable) term.writeln(ev.data); }
      };
      ws.onclose = ()=>{ ws=null }
      return true;
    }catch(e){ console.warn('ws run failed', e); ws=null; return false }
  }

  async function handleRun(code, shots){
    const usedWs = startWsRun(code, shots);
    if(usedWs) return;
    if(xtermAvailable){ term.reset(); term.writeln('Running...'); } else { const fallback = document.getElementById('terminal_out'); if(fallback) fallback.textContent = 'Running...\n'; }
    try{
      const res = await api.run(code, shots);
      const text = JSON.stringify(res, null, 2);
      if(xtermAvailable){ term.writeln(text.replace(/\n/g,'\r\n')); }
      else { const fallback = document.getElementById('terminal_out'); if(fallback) fallback.textContent += text }

      // draw graph if numeric array present
      try{
        const obj = res.result || res;
        let arr = null;
        if(Array.isArray(obj)) arr = obj;
        else if(obj && typeof obj === 'object'){
          for(const k of Object.keys(obj)) if(Array.isArray(obj[k])) { arr = obj[k]; break }
        }
        if(arr && graphCanvas){
          const ctx = graphCanvas.getContext('2d');
          ctx.clearRect(0,0,graphCanvas.width, graphCanvas.height);
          const w = graphCanvas.width / arr.length;
          const max = Math.max(...arr);
          arr.forEach((v,i)=>{
            const h = (v/max) * (graphCanvas.height-20);
            ctx.fillStyle = '#b58cff';
            ctx.fillRect(i*w+4, graphCanvas.height - h -10, w-8, h);
          })
        }
      }catch(e){ console.warn('graph draw failed', e) }
    }catch(e){ if(xtermAvailable) term.writeln('Run error: '+String(e)); else { const fallback = document.getElementById('terminal_out'); if(fallback) fallback.textContent = 'Run error: '+String(e) } }
  }

  if(runTop){ runTop.addEventListener('click', async ()=>{
    const code = cm ? cm.getValue() : rawTextarea.value;
    const shotsInput = document.getElementById('shots_input');
    const shots = shotsInput ? parseInt(shotsInput.value || '1024', 10) : 1024;
    await handleRun(code, shots);
  }) }
  if(runSimple){ runSimple.addEventListener('click', async ()=>{
    const code = cm ? cm.getValue() : rawTextarea.value;
    const shotsInput = document.getElementById('shots_input');
    const shots = shotsInput ? parseInt(shotsInput.value || '1024', 10) : 1024;
    await handleRun(code, shots);
  }) }

  // right-side tab switching
  document.querySelectorAll('.rtab').forEach(btn=>{
    btn.addEventListener('click', ()=>{
      document.querySelectorAll('.rtab').forEach(t=>t.classList.remove('active'));
      btn.classList.add('active');
      const panel = btn.getAttribute('data-panel');
      document.querySelectorAll('.rpanel').forEach(p=>p.classList.remove('active'));
      const el = document.getElementById('panel-'+panel);
      if(el) el.classList.add('active');
    })
  })

  // Mirror editor into previews and terminal
  function mirrorToPreviews(){
    const txt = cm ? cm.getValue() : rawTextarea.value;
    const p1 = document.getElementById('preview1');
    const p2 = document.getElementById('preview2');
    if(p1) p1.textContent = txt;
    if(p2) p2.textContent = txt;
    // update xterm mirror (show last portion)
    const out = txt.slice(-20000);
    if(xtermAvailable && term){
      term.reset();
      // write preserving newlines
      term.write(out.replace(/\n/g,'\r\n'));
    }else{
      const fallback = document.getElementById('terminal_out'); if(fallback) fallback.textContent = out;
    }
  }
  if(cm){ cm.on('change', mirrorToPreviews); mirrorToPreviews(); }

  // add a vertical resize handle for terminal panel (single instance)
  (function addTerminalResizeHandle(){
    const panelTerminal = document.getElementById('panel-terminal');
    if(!panelTerminal) return;
    const handle = document.createElement('div');
    handle.className = 'terminal-resize-handle';
    panelTerminal.parentNode.insertBefore(handle, panelTerminal);
    let dragging = false;
    handle.addEventListener('mousedown', (e)=>{ dragging = true; document.body.style.cursor='ns-resize'; e.preventDefault(); });
    window.addEventListener('mousemove', (e)=>{
      if(!dragging) return;
      const newH = Math.max(120, window.innerHeight - e.clientY - 120);
      panelTerminal.style.minHeight = newH + 'px';
      if(xtermAvailable && term) term.resize && term.resize(80, Math.max(10, Math.floor(newH/18)));
    });
    window.addEventListener('mouseup', ()=>{ if(dragging){ dragging=false; document.body.style.cursor=''; } });
  })();

  // cursor-follow light
  const light = document.getElementById('cursor_light');
  if(light){
    document.addEventListener('mousemove', e => {
      const w = light.offsetWidth/2;
      const h = light.offsetHeight/2;
      light.style.transform = `translate(${e.clientX - w}px, ${e.clientY - h}px)`;
    });
  }

  // theme toggle
  const themeBtn = document.getElementById('btn_theme_toggle');
  if(themeBtn){ themeBtn.addEventListener('click', ()=>{ document.documentElement.classList.toggle('light-theme'); }) }

  // install / save UI buttons
  const btnInstallHelp = document.getElementById('btn_install_help');
  if(btnInstallHelp) btnInstallHelp.addEventListener('click', async ()=>{
    const data = await api.installHelp(); if(output) output.textContent = JSON.stringify(data, null, 2);
  })

  const btnInstall = document.getElementById('btn_install');
  if(btnInstall) btnInstall.addEventListener('click', async ()=>{
    const pkgEl = document.getElementById('install_pkg');
    const pkg = pkgEl ? pkgEl.value || 'opt_einsum' : 'opt_einsum';
    if(output) output.textContent = 'Installing '+pkg+'...';
    const data = await api.install(pkg);
    if(output) output.textContent = JSON.stringify(data, null, 2);
  })

  const btnSave = document.getElementById('btn_save');
  if(btnSave) btnSave.addEventListener('click', async ()=>{
    const nameEl = document.getElementById('save_name');
    const name = nameEl ? nameEl.value || 'script.py' : 'script.py';
    const content = cm ? cm.getValue() : rawTextarea.value;
    const res = await api.save(name, content);
    if(output) output.textContent = JSON.stringify(res, null, 2);
    refreshFiles();
  })

  // settings panel wiring
  const settingsBtn = document.getElementById('btn_settings');
  const settingsPanel = document.getElementById('settings_panel');
  const settingsClose = document.getElementById('settings_close');
  const optCursor = document.getElementById('opt_cursor_light');
  const optBg = document.getElementById('opt_bg_anim');
  const optReduced = document.getElementById('opt_reduced_motion');
  const optTermFont = document.getElementById('opt_term_font');
  const optRealtime = document.getElementById('opt_realtime');

  if(settingsBtn && settingsPanel){
    settingsBtn.addEventListener('click', ()=>{ settingsPanel.setAttribute('aria-hidden','false'); });
    if(settingsClose) settingsClose.addEventListener('click', ()=>{ settingsPanel.setAttribute('aria-hidden','true'); });

    // initialize controls from current UI
    if(optCursor) optCursor.checked = !!light;
    if(optBg) optBg.checked = true;
    if(optReduced) optReduced.checked = false;
    if(optTermFont) optTermFont.value = 13;

    // toggles
    if(optCursor){ optCursor.addEventListener('change', ()=>{ if(!light) return; light.style.display = optCursor.checked ? 'block' : 'none'; }) }
    if(optBg){ optBg.addEventListener('change', ()=>{ const bg = document.querySelector('.bg-anim'); if(bg) bg.style.display = optBg.checked ? 'block' : 'none'; }) }
    if(optReduced){ optReduced.addEventListener('change', ()=>{ if(optReduced.checked) document.documentElement.style.setProperty('--reduced-motion','1'); else document.documentElement.style.removeProperty('--reduced-motion'); }) }
    if(optTermFont){ optTermFont.addEventListener('input', ()=>{ const v = parseInt(optTermFont.value,10)||13; if(xtermAvailable && term) term.setOption('fontSize', v); else { const t = document.getElementById('terminal_out'); if(t) t.style.fontSize = v+'px' } }) }

    // realtime websocket toggle
    optRealtime && optRealtime.addEventListener('change', async ()=>{
      if(optRealtime.checked){
        // try to open websocket to server
        try{
          ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws');
          ws.onopen = ()=>{ console.log('ws open'); if(xtermAvailable) term.writeln('realtime connected'); }
          ws.onmessage = (ev)=>{ if(xtermAvailable) term.writeln(ev.data.replace(/\n/g,'\r\n')); else { const t = document.getElementById('terminal_out'); if(t) t.textContent += '\n'+ev.data } }
          ws.onclose = ()=>{ console.log('ws closed'); if(xtermAvailable) term.writeln('realtime disconnected'); }
        }catch(e){ console.warn('ws connect failed', e); optRealtime.checked = false }
      }else{
        if(ws){ ws.close(); ws = null }
      }
    })
  }

  // Assistant panel: create dynamically and wire to /assistant
  (function initAssistantPanel(){
    if(document.getElementById('assistant_panel')) return;
    const panel = document.createElement('div');
    panel.id = 'assistant_panel';
    panel.className = 'assistant-panel';
    panel.setAttribute('aria-hidden','true');
    panel.innerHTML = `
      <div class="settings-card card" style="width:720px; max-width:92%; color: inherit;">
        <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
          <h3 style="margin:0">Assistant</h3>
          <div>
            <button id="assistant_close" class="icon-btn">Close</button>
          </div>
        </div>
        <div style="margin-top:10px">
          <textarea id="assistant_prompt" placeholder="Ask the assistant (e.g. 'Find bug', 'Refactor to use list comprehensions', 'Add measurements')" 
            style="width:100%;height:110px;background:transparent;color:inherit;border:1px solid rgba(255,255,255,0.04);padding:10px;border-radius:8px;font-family:inherit"></textarea>
        </div>
        <div style="display:flex;gap:8px;margin-top:10px;justify-content:flex-end">
          <button id="assistant_send" class="run">Send</button>
          <button id="assistant_insert" class="icon-btn">Insert</button>
          <button id="assistant_insert_run" class="run">Insert & Run</button>
        </div>
        <pre id="assistant_response" style="margin-top:12px;max-height:300px;overflow:auto;background:linear-gradient(180deg, rgba(0,0,0,0.5), rgba(6,4,12,0.6));padding:12px;border-radius:8px;color:#eae6ff"></pre>
      </div>
    `;
    document.body.appendChild(panel);

    const btnOpen = document.getElementById('btn_assistant');
    // if there's a topbar assistant button, wire it; otherwise create a small floating trigger
    let trigger = btnOpen;
    if(!trigger){
      trigger = document.createElement('button');
      trigger.id = 'btn_assistant';
      trigger.className = 'icon-btn';
      trigger.textContent = 'Assistant';
      trigger.style.position = 'fixed';
      trigger.style.right = '18px';
      trigger.style.bottom = '18px';
      trigger.style.zIndex = 1000;
      document.body.appendChild(trigger);
    }

    const openPanel = ()=>{ panel.setAttribute('aria-hidden','false'); }
    const closePanel = ()=>{ panel.setAttribute('aria-hidden','true'); }
    trigger.addEventListener('click', openPanel);
    document.getElementById('assistant_close').addEventListener('click', closePanel);

    const promptEl = document.getElementById('assistant_prompt');
    const respEl = document.getElementById('assistant_response');
    const sendBtn = document.getElementById('assistant_send');
    const insertBtn = document.getElementById('assistant_insert');
    const insertRunBtn = document.getElementById('assistant_insert_run');

    async function callAssistant(promptText, codeContext){
      // Clear previous response
      respEl.textContent = '';
      // Try websocket streaming first
      if('WebSocket' in window){
        try{
          const ws = new WebSocket((location.protocol==='https:'?'wss://':'ws://') + location.host + '/ws-assistant');
          let finished = false;
          let finalResult = { ok: false };
          ws.onopen = ()=>{
            ws.send(JSON.stringify({ prompt: promptText, code: codeContext }));
          };
          ws.onmessage = (ev)=>{
            try{
              const msg = JSON.parse(ev.data);
              if(msg.type === 'delta'){
                respEl.textContent += msg.data;
                // auto-scroll
                respEl.scrollTop = respEl.scrollHeight;
              } else if(msg.type === 'done'){
                respEl.textContent += '\n';
                finalResult = { ok: true, text: msg.text, extracted_code: msg.extracted_code };
                finished = true;
                try{ ws.close(); }catch(e){}
              } else if(msg.type === 'error'){
                respEl.textContent = 'Assistant error: ' + (msg.error || JSON.stringify(msg));
                finalResult = { ok: false, error: msg.error || 'upstream_error' };
                finished = true;
                try{ ws.close(); }catch(e){}
              }
            }catch(err){
              // non-json payload: append raw
              respEl.textContent += ev.data;
            }
          };
          ws.onerror = (e)=>{ console.warn('assistant ws error', e); };
          // wait for completion or timeout
          const timeoutMs = 60 * 1000;
          const start = Date.now();
          await new Promise((resolve)=>{
            const iv = setInterval(()=>{
              if(finished) { clearInterval(iv); resolve(); return; }
              if(Date.now() - start > timeoutMs){
                // timeout: close and resolve
                try{ ws.close(); }catch(e){}
                clearInterval(iv);
                resolve();
              }
            }, 50);
          });
          return finalResult;
        }catch(e){
          console.warn('ws-assistant failed, falling back to POST', e);
          respEl.textContent = 'Connecting to assistant failed, falling back...\n';
        }
      }

      // Fallback: synchronous POST
      respEl.textContent += 'Thinking...\n';
      try{
        const body = { prompt: promptText, code: codeContext };
        const r = await fetch('/assistant', { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body) });
        const j = await r.json();
        if(!j.ok){
          respEl.textContent = 'Assistant error:\n' + (j.error || JSON.stringify(j));
          return j;
        }
        const text = j.text || (j.response && JSON.stringify(j.response, null, 2)) || '';
        respEl.textContent = text;
        return j;
      }catch(e){
        respEl.textContent = 'Assistant request failed: ' + String(e);
        return { ok:false, error: String(e) };
      }
    }

    sendBtn.addEventListener('click', async ()=>{
      const p = promptEl.value || '';
      const code = cm ? cm.getValue() : rawTextarea.value;
      await callAssistant(p, code);
    });

    // helper to extract suggested code from assistant response
    function extractSuggestedCode(result){
      if(!result) return '';
      if(result.extracted_code) return result.extracted_code;
      if(result.text) return result.text;
      if(result.response){
        try{
          const resp = result.response;
          if(typeof resp === 'string') return resp;
          if(resp.candidates && Array.isArray(resp.candidates)){
            return resp.candidates.map(c => (c.content||c)).join('\n');
          }
          if(resp.output) return resp.output;
          return JSON.stringify(resp, null, 2);
        }catch(e){ return '' }
      }
      return '';
    }

    insertBtn.addEventListener('click', async ()=>{
      const p = promptEl.value || '';
      const code = cm ? cm.getValue() : rawTextarea.value;
      const res = await callAssistant(p, code);
      const suggestion = extractSuggestedCode(res);
      if(!suggestion) return;
      if(cm){ const pos = cm.getCursor(); cm.replaceRange('\n' + suggestion + '\n', pos); cm.focus(); }
      else { rawTextarea.value += '\n' + suggestion; }
    });

    insertRunBtn.addEventListener('click', async ()=>{
      const p = promptEl.value || '';
      const code = cm ? cm.getValue() : rawTextarea.value;
      const res = await callAssistant(p, code);
      const suggestion = extractSuggestedCode(res);
      if(suggestion){
        if(cm){ const pos = cm.getCursor(); cm.replaceRange('\n' + suggestion + '\n', pos); cm.focus(); }
        else { rawTextarea.value += '\n' + suggestion; }
        // run the updated code
        const fullCode = cm ? cm.getValue() : rawTextarea.value;
        const shotsInput = document.getElementById('shots_input');
        const shots = shotsInput ? parseInt(shotsInput.value || '1024', 10) : 1024;
        await handleRun(fullCode, shots);
      }
    });

  })();

})
