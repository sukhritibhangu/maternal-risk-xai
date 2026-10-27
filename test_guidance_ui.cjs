/* Offline rendering checks; no browser, API key or package install required. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync(__dirname + '/static/index.html', 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const definitions = script.slice(script.indexOf('const safeURL='), script.indexOf('// global'));
const context = vm.createContext({
  esc: value => String(value).replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
  pts: value => String(value),
  URL,
});
vm.runInContext(definitions + ';globalThis.renderSuggestions=suggest;', context);
const p = {citation:'S1', doc_id:'d1', title:'Source', source:'https://example.org',
           text:'An exact evidence sentence.', verified:true, page:7, section:'Antenatal care'};
const c = {group:'Haemoglobin', level:'info', reasons:[], shap:0,
           passages:[p], highlights:[{text:p.text,citation:'S1'}],
           ai_guidance:{suggestion:'<img src=x onerror=alert(1)>',
                        applicability:'Trimester unknown.',gap:'',conflict:false}};
const result = context.renderSuggestions({
  cards:[c,{...c,group:'General information'}], only_verified:true,
  loop:{status:'ready',search:'hybrid',rounds:2,generated_cards:2}, ai:{}
});
assert(result.includes('AI suggestion'));
assert(result.includes('keyword and semantic search'));
assert(result.includes('PDF page 7'));
assert(result.includes('Trimester unknown.'));
assert(result.includes('&lt;img'));
assert(!result.includes('<img src=x'));
assert.equal((result.match(/id="evidence-S1"/g)||[]).length,1);
assert(result.includes('href="#evidence-S1"'));
const fallback = context.renderSuggestions({cards:[],only_verified:false,
  loop:{status:'fallback',reason:'no_verified_sources'},ai:{}});
assert(fallback.includes('Add and verify source documents'));
assert(fallback.includes('Showing the original source excerpts'));
console.log('Guidance UI checks passed: citations, escaping, page labels and fallback.');
