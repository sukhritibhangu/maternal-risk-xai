(() => {
  'use strict';
  const byId = id => document.getElementById(id);
  const panel = byId('chat-panel'), launch = byId('chat-launch');
  const messages = byId('chat-messages'), question = byId('chat-question');
  const include = byId('chat-include'), send = byId('chat-send');
  const status = byId('chat-status'), error = byId('chat-error');
  const welcomeMarkup = messages.innerHTML;
  let enabled = false, busy = false, assessment = null, history = [];
  let controller = null, revision = 0, messageNumber = 0;

  function sync() {
    question.disabled = !enabled || busy;
    send.disabled = !enabled || busy;
    include.disabled = !assessment || busy;
    panel.querySelectorAll('[data-chat-prompt]').forEach(button => {
      button.disabled = !enabled || busy || (button.hasAttribute('data-chat-assessment') && !assessment);
    });
    send.textContent = busy ? 'Finding…' : 'Send';
    const critical = include.checked && assessment ? assessment.critical : [];
    byId('chat-critical').hidden = !critical.length;
    byId('chat-critical').textContent = critical.length ? 'Existing assessment alerts: ' + critical.join(' ') : '';
  }
  function scrollEnd() {
    messages.scrollTo({top:messages.scrollHeight,behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});
  }
  function reset(note = '') {
    revision++;
    if (controller) controller.abort();
    controller = null;
    busy = false;
    history = [];
    messages.innerHTML = welcomeMarkup;
    error.textContent = '';
    status.textContent = note || (enabled ? 'Ready. Answers include supporting evidence.' : 'Chat is off. An administrator can enable it in Knowledge base.');
    sync();
    wirePrompts();
  }
  function show() {
    panel.hidden = false;
    launch.setAttribute('aria-expanded','true');
    (enabled ? question : byId('chat-close')).focus();
  }
  function close() {
    panel.hidden = true;
    launch.setAttribute('aria-expanded','false');
    launch.focus();
  }
  launch.addEventListener('click', () => panel.hidden ? show() : close());
  byId('chat-close').addEventListener('click', close);
  panel.addEventListener('keydown', event => {
    if (event.key === 'Escape') { event.stopPropagation(); close(); }
  });
  byId('chat-clear').addEventListener('click', () => { reset(); question.value=''; if(enabled)question.focus(); });
  include.addEventListener('change', () => {
    reset(include.checked ? 'New chat with the current assessment included.' : 'New chat using documents and application explanations.');
  });
  window.addEventListener('matcare:meta', event => {
    const next = event.detail.chatbot === true;
    if (next !== enabled) { enabled=next; reset(); }
    else sync();
    if (!enabled) status.textContent='Chat is off. An administrator can enable it in Knowledge base.';
  });
  window.addEventListener('matcare:assessment', event => {
    assessment = event.detail.token ? event.detail : null;
    include.checked = false;
    byId('chat-context-label').textContent = assessment
      ? 'Current model score: '+assessment.score+'. Check the box to share its explanation.'
      : 'Run an assessment to make its explanation available.';
    reset('New assessment available. Choose whether to include it in chat.');
  });
  window.addEventListener('matcare:assessment-stale', () => {
    if (!assessment) return;
    assessment = null;
    include.checked = false;
    byId('chat-context-label').textContent='Inputs changed. Assess again before including a result.';
    reset('Assessment context cleared because the inputs changed.');
  });

  function element(tag, text, className) {
    const node=document.createElement(tag);
    if(text!==undefined)node.textContent=text;
    if(className)node.className=className;
    return node;
  }
  function message(role) {
    const welcome=byId('chat-welcome'); if(welcome)welcome.remove();
    const article=element('article',undefined,'chat-message '+role);
    article.append(element('p',role==='user'?'You':'Matcare','chat-message-label'));
    const bubble=element('div',undefined,'chat-bubble');article.append(bubble);messages.append(article);
    return bubble;
  }
  function safeURL(value) {
    try { const url=new URL(value);return ['http:','https:'].includes(url.protocol)?url.href:null; }
    catch { return null; }
  }
  function renderAnswer(data) {
    const bubble=message('assistant'), prefix='chat-'+(++messageNumber)+'-';
    for(const part of data.paragraphs||[]) {
      const p=element('p',part.text);
      for(const ref of part.references||[]) {
        const button=element('button','['+ref+']','chat-ref');
        button.type='button';button.setAttribute('aria-label','Read evidence '+ref);
        button.addEventListener('click',()=>{
          const source=byId(prefix+ref); if(!source)return;
          source.closest('details').open=true;
          source.scrollIntoView({block:'nearest',behavior:matchMedia('(prefers-reduced-motion:reduce)').matches?'instant':'smooth'});
          source.focus({preventScroll:true});
        });
        p.append(button);
      }
      bubble.append(p);
    }
    if(data.gap)bubble.append(element('p',data.gap,'source-gap'));
    if(data.sources?.length) {
      const details=element('details',undefined,'chat-evidence');
      details.append(element('summary','Evidence and quotations ('+data.sources.length+')'));
      if(data.status==='fallback')details.open=true;
      for(const source of data.sources) {
        const box=element('div',undefined,'chat-source');box.id=prefix+source.id;box.tabIndex=-1;
        box.append(element('h4','['+source.id+'] '+source.title));
        let label=source.kind==='document'?'Admin-verified document':source.kind==='assessment'?'Current assessment':'Application explanation';
        if(source.page)label+=' · PDF page '+source.page;
        if(source.section)label+=' · '+source.section;
        box.append(element('p',label,'note'));
        for(const quote of source.quotes||[])box.append(element('blockquote',quote));
        box.append(element('p',source.text));
        const url=safeURL(source.source);
        if(url){const link=element('a','Open source');link.href=url;link.target='_blank';link.rel='noopener';box.append(link);}
        details.append(box);
      }
      bubble.append(details);
    }
    bubble.append(element('p','AI-generated. Evidence links identify the source text; review the answer against it.','note'));
    scrollEnd();
  }
  function wirePrompts() {
    messages.querySelectorAll('[data-chat-prompt]').forEach(button => {
      button.addEventListener('click', () => {
        if(button.hasAttribute('data-chat-assessment')){
          if(!assessment)return;
          include.checked=true;reset('Current assessment included.');
        }
        question.value=button.dataset.chatPrompt;
        byId('chat-form').requestSubmit();
      });
    });
    sync();
  }
  question.addEventListener('keydown',event=>{
    if(event.key==='Enter'&&!event.shiftKey&&!event.isComposing){
      event.preventDefault();if(!busy&&enabled)byId('chat-form').requestSubmit();
    }
  });
  byId('chat-form').addEventListener('submit',async event=>{
    event.preventDefault();
    const text=question.value.trim();if(!enabled||busy||!text)return;
    const currentRevision=revision;
    controller=new AbortController();
    const activeController=controller;
    const timeout=setTimeout(()=>activeController.abort(),60000);
    const payload={message:text,history:history.slice(-6)};
    if(include.checked&&assessment)payload.assessment_token=assessment.token;
    message('user').append(element('p',text));
    error.textContent='';busy=true;sync();status.textContent='Finding evidence and preparing an answer…';scrollEnd();
    try{
      const response=await fetch('/api/chat',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify(payload),signal:controller.signal});
      const data=await response.json().catch(()=>({error:'The server returned an unreadable response. Please try again.'}));
      if(currentRevision!==revision)return;
      if(!response.ok){
        if(data.code==='assessment_expired'){
          assessment=null;include.checked=false;
          byId('chat-context-label').textContent='Assessment expired. Assess again to include a result.';
        }
        throw new Error(data.error||'Chat could not complete this request.');
      }
      renderAnswer(data);
      const answer=(data.paragraphs||[]).map(p=>p.text).concat(data.gap||[]).join('\n\n');
      history.push({role:'user',content:text},{role:'assistant',content:(answer||'No supported answer was available.').slice(0,2000)});
      history=history.slice(-6);
      question.value='';
      status.textContent=data.status==='ready'
        ?'Answer ready · '+(data.search==='hybrid'?'Keyword and semantic evidence search':data.search==='keyword'?'Keyword evidence search':'Application and assessment explanation')
        :data.status==='insufficient'?'More applicable verified evidence is needed.':'AI answer unavailable. Source excerpts remain available.';
    }catch(exc){
      if(currentRevision!==revision)return;
      error.textContent=exc.name==='AbortError'?'Chat timed out. Please try again.':exc.message;
      status.textContent='Your question was kept so you can retry.';
    }finally{
      clearTimeout(timeout);
      if(currentRevision===revision){busy=false;controller=null;sync();}
    }
  });
  wirePrompts();
})();
