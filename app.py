import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit.components.v1 as components
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import json
from fpdf import FPDF
from constants import (
    DHI_THRESHOLD_MILD, DHI_THRESHOLD_MODERATE,
    DHI_PHYSICAL_QS, DHI_EMOTIONAL_QS, DHI_FUNCTIONAL_QS,
)

_DIR = Path(__file__).parent

# ── MUST be first Streamlit call ──
st.set_page_config(
    page_title="PPPD Clinical CDSS",
    layout="wide",
)

# ── Google Fonts ──
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

# ── ALL STYLES (banner + tabs + cards + everything) ──
st.markdown(
    "<style>" + (_DIR / "main.css").read_text(encoding="utf-8") + "</style>",
    unsafe_allow_html=True,
)

# ── Banner ──
st.markdown(
    '<div class="fixed-banner">PPPD Clinical Decision Support System</div>'
    '<div class="hamburger-icon" id="hamburger-icon">'
    '<span></span><span></span><span></span>'
    '</div>'
    '<div class="menu-backdrop" id="menu-backdrop"></div>',
    unsafe_allow_html=True,
)

# ── Hamburger JS ──
components.html("""
<script>
(function(){
  var doc;
  try { doc = window.parent.document; } catch(e) { doc = document; }

  function setup() {
    var hamburger = doc.getElementById('hamburger-icon');
    var backdrop  = doc.getElementById('menu-backdrop');
    if (!hamburger || !backdrop) {
      setTimeout(setup, 300);
      return;
    }
    if (hamburger._bound) return;
    hamburger._bound = true;
    hamburger.addEventListener('click', function(e) {
      e.stopPropagation();
      doc.body.classList.toggle('menu-open');
    });
    doc.addEventListener('click', function(e) {
      if (!doc.body.classList.contains('menu-open')) return;
      if (e.target.closest && e.target.closest('.hamburger-icon')) return;
      // Close menu when clicking a tab button or anywhere outside the popup
      if (e.target.closest && e.target.closest('[data-baseweb="tab-list"] button')) {
        doc.body.classList.remove('menu-open');
      } else if (!e.target.closest || !e.target.closest('[data-baseweb="tab-list"]')) {
        doc.body.classList.remove('menu-open');
      }
    });
  }
  if (doc.readyState === 'complete') setup();
  else doc.addEventListener('DOMContentLoaded', setup);
  setTimeout(setup, 500);
})();
</script>
""", height=0, scrolling=False)


# ── Load model + dataset ──
try:
    model = joblib.load(_DIR / "pppd_vrt_model.pkl")
except Exception:
    model = None
    st.error("Model not found. Run train_model.py first.")

try:
    participants_df = pd.read_csv(_DIR / "participants.tsv", sep="\t")
except Exception:
    participants_df = None


# ── Layered Scroll Component (stacking sections + horizontal panels) ──
def layered_scroll(sections_data, height=2000):
    """Sections stack vertically; within each, panels scroll horizontally."""
    sections_html = ""
    sec_dots_html = ""
    nav_tabs_html = ""
    bgs = ["#f5f0eb", "#ffffff", "#f0ede8", "#faf7f2"]

    for si, sec in enumerate(sections_data):
        bg = bgs[si % len(bgs)]

        if "html" in sec:
            sections_html += (
                '<div class="section" data-si="'
                + str(si)
                + '" style="background:'
                + bg
                + '">'
                '  <div class="sec-header">'
                '    <h2 class="sec-title">' + sec.get("title", "") + "</h2>"
                '    <p class="sec-sub">' + sec.get("subtitle", "") + "</p>"
                '    <div class="sec-divider"></div>'
                "  </div>" + sec["html"] + "</div>"
            )
        else:
            panels_html = ""
            panel_dots = ""
            num_panels = len(sec["panels"])
            for pi, p in enumerate(sec["panels"]):
                tag = p.get("tag", "")
                num = str(pi + 1).zfill(2)
                # Add prev arrow on all panels except the first
                prev_arrow = ""
                if pi > 0:
                    prev_arrow = (
                        '<div class="panel-prev-btn" data-si="'
                        + str(si)
                        + '" data-pi="'
                        + str(pi - 1)
                        + '">'
                        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
                        '<circle cx="12" cy="12" r="11"/>'
                        '<path d="M14 8l-4 4 4 4"/>'
                        '</svg>'
                        '</div>'
                    )
                # Add next arrow on all panels except the last
                next_arrow = ""
                if pi < num_panels - 1:
                    next_arrow = (
                        '<div class="panel-next-btn" data-si="'
                        + str(si)
                        + '" data-pi="'
                        + str(pi + 1)
                        + '">'
                        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
                        '<circle cx="12" cy="12" r="11"/>'
                        '<path d="M10 8l4 4-4 4"/>'
                        '</svg>'
                        '</div>'
                    )
                hide_meta = p.get("hide_meta", False)
                meta_top = (
                    '    <div class="panel-num">'
                    + num
                    + '<span class="panel-line"></span></div>'
                    '    <span class="panel-tag">' + tag + "</span>"
                ) if not hide_meta else ""
                panels_html += (
                    '<div class="panel">'
                    '  <div class="panel-img-wrap">'
                    '    <div class="panel-img"><img src="'
                    + p["img"]
                    + '" alt="'
                    + p["title"]
                    + '"></div>'
                    "  </div>"
                    '  <div class="panel-meta">'
                    + meta_top
                    + '    <h3 class="panel-title">' + p["title"] + "</h3>"
                    '    <div class="panel-body">' + p["body"] + "</div>"
                    "  </div>"
                    + prev_arrow
                    + next_arrow
                    + "</div>"
                )
                pact = " active" if pi == 0 else ""
                panel_dots += (
                    '<span class="pdot'
                    + pact
                    + '" data-si="'
                    + str(si)
                    + '" data-pi="'
                    + str(pi)
                    + '"></span>'
                )

            sections_html += (
                '<div class="section" data-si="'
                + str(si)
                + '" style="background:'
                + bg
                + '">'
                '  <div class="sec-header">'
                '    <h2 class="sec-title">' + sec["title"] + "</h2>"
                '    <p class="sec-sub">' + sec["subtitle"] + "</p>"
                '    <div class="sec-divider"></div>'
                "  </div>"
                '  <div class="track-wrap">'
                '    <div class="track" data-si="'
                + str(si)
                + '">'
                + panels_html
                + "</div>"
                "  </div>"
                '  <div class="panel-dots" data-si="'
                + str(si)
                + '">'
                + panel_dots
                + "</div>"
                "</div>"
            )

        sact = " active" if si == 0 else ""
        sec_dots_html += '<span class="sdot' + sact + '"></span>'
        nav_tabs_html += (
            '<span class="nav-tab' + sact + '" data-si="' + str(si) + '">'
            + sec.get("title", "") + '</span>'
        )

    css = (_DIR / "scroll.css").read_text(encoding="utf-8")

    n = len(sections_data)
    panel_counts = [
        len(sec["panels"]) if "panels" in sec else 1 for sec in sections_data
    ]
    panel_counts_js = "[" + ",".join(str(c) for c in panel_counts) + "]"

    js = r"""
(function(){
  var sections = document.querySelectorAll('.section');
  var sdots = document.querySelectorAll('.sdot');
  var navTabs = document.querySelectorAll('.nav-tab');
  var counter = document.getElementById('counter');
  var N = sections.length;
  if(!N) return;
  var cont = document.querySelector('.container');
  var vh = cont.clientHeight;
  var maxVScroll = (N - 1) * vh;
  var panelCounts = PANEL_COUNTS;

  /* keep vh in sync after iframe resize */
  function refreshVh(){
    vh = cont.clientHeight;
    maxVScroll = (N - 1) * vh;
    /* re-snap to nearest section so content stays aligned */
    var curIdx = Math.round(vTarget / (vh || 1));
    vTarget = curIdx * vh;
    vCurrent = vTarget;
    /* re-snap horizontal tracks */
    for(var s = 0; s < N; s++){
      var track = sections[s].querySelector('.track');
      if(track){
        var tw = track.clientWidth;
        var curP = Math.round(hTargets[s] / (tw || 1));
        hTargets[s] = curP * tw;
        hCurrents[s] = hTargets[s];
      }
    }
  }
  try{
    var f = window.frameElement;
    function resizeIframe(){
      var r = f.getBoundingClientRect();
      f.style.height = (window.parent.innerHeight - r.top) + 'px';
      setTimeout(refreshVh, 50);
    }
    resizeIframe();
    window.parent.addEventListener('resize', resizeIframe);
    /* handle orientation change on mobile */
    if(window.parent.screen && window.parent.screen.orientation){
      window.parent.screen.orientation.addEventListener('change', function(){
        setTimeout(resizeIframe, 150);
      });
    }
    window.addEventListener('orientationchange', function(){ setTimeout(resizeIframe, 200); });
    setTimeout(refreshVh, 200);
  }catch(e){}

  /* per-section horizontal state */
  var hTargets = new Array(N);
  var hCurrents = new Array(N);
  for(var s = 0; s < N; s++){ hTargets[s] = 0; hCurrents[s] = 0; }

  var vTarget = 0, vCurrent = 0;
  var snapTimer = null;
  var hSnapTimers = new Array(N);
  for(var s = 0; s < N; s++) hSnapTimers[s] = null;

  /* snap vTarget to nearest section boundary */
  function scheduleSnap(){
    if(snapTimer) clearTimeout(snapTimer);
    snapTimer = setTimeout(function(){
      vTarget = Math.round(vTarget / vh) * vh;
    }, 180);
  }
  /* snap horizontal to nearest panel */
  function scheduleHSnap(si){
    if(hSnapTimers[si]) clearTimeout(hSnapTimers[si]);
    hSnapTimers[si] = setTimeout(function(){
      var track = sections[si].querySelector('.track');
      if(track) hTargets[si] = Math.round(hTargets[si] / track.clientWidth) * track.clientWidth;
    }, 180);
  }

  /* z-index: later sections on top */
  for(var i = 0; i < N; i++) sections[i].style.zIndex = i + 1;

  /* figure out active section index from vCurrent */
  function activeIdx(){ return Math.round(vCurrent / vh); }

  /* check if section has scrollable content that hasn't reached boundary */
  function sectionCanScroll(si, direction){
    var scrollEl = sections[si].querySelector('.intro-content') || sections[si];
    if(scrollEl.scrollHeight <= scrollEl.clientHeight + 2) return false;
    if(direction > 0) return scrollEl.scrollTop < scrollEl.scrollHeight - scrollEl.clientHeight - 2;
    return scrollEl.scrollTop > 2;
  }

  /* ── wheel ── */
  document.addEventListener('wheel', function(e){
    /* let modals scroll natively */
    if(e.target.closest && e.target.closest('.sym-modal')) return;

    var ai = activeIdx();
    var track = sections[ai].querySelector('.track');

    /* let scrollable sections scroll internally first */
    if(panelCounts[ai] <= 1 && sectionCanScroll(ai, e.deltaY)){
      return; /* allow native scroll */
    }
    e.preventDefault();

    /* try horizontal first if section has multiple panels */
    if(panelCounts[ai] > 1){
      var hMax = (panelCounts[ai] - 1) * track.clientWidth;
      if(e.deltaY > 0 && hTargets[ai] < hMax - 5){
        hTargets[ai] = Math.min(hMax, hTargets[ai] + Math.abs(e.deltaY) * 1.6);
        scheduleHSnap(ai);
        return;
      }
      if(e.deltaY < 0 && hTargets[ai] > 5){
        hTargets[ai] = Math.max(0, hTargets[ai] - Math.abs(e.deltaY) * 1.6);
        scheduleHSnap(ai);
        return;
      }
    }

    /* otherwise vertical stacking */
    if(vTarget >= maxVScroll && e.deltaY > 0) return;
    if(vTarget <= 0 && e.deltaY < 0) return;
    vTarget = Math.max(0, Math.min(maxVScroll, vTarget + e.deltaY * 1.2));
    scheduleSnap();
  }, {passive:false});

  /* ── touch ── */
  var tx=0, ty=0, ts=0, ths=0, tdir=null, tai=0;
  document.addEventListener('touchstart', function(e){
    tx = e.touches[0].clientX; ty = e.touches[0].clientY;
    tai = activeIdx(); ts = vTarget; ths = hTargets[tai]; tdir = null;
  });
  document.addEventListener('touchmove', function(e){
    /* let modals scroll natively */
    if(e.target.closest && e.target.closest('.sym-modal')) return;

    var dx = tx - e.touches[0].clientX;
    var dy = ty - e.touches[0].clientY;

    /* direction lock: require 8px movement and bias toward vertical */
    if(!tdir){
      var adx = Math.abs(dx), ady = Math.abs(dy);
      if(adx < 8 && ady < 8) return;
      tdir = (adx > ady * 1.4) ? 'h' : 'v';
    }

    /* let scrollable intro sections scroll natively */
    if(tdir === 'v' && panelCounts[tai] <= 1 && sectionCanScroll(tai, dy)){
      return;
    }

    /* allow panels to scroll internally on mobile if content overflows */
    if(tdir === 'v' && panelCounts[tai] > 1){
      var track = sections[tai].querySelector('.track');
      var tw = track ? track.clientWidth : 1;
      var curPanel = Math.round(hCurrents[tai] / tw);
      var panel = track ? track.querySelectorAll('.panel')[curPanel] : null;
      if(panel && panel.scrollHeight > panel.clientHeight + 4){
        var atTop = panel.scrollTop <= 2;
        var atBot = panel.scrollTop >= panel.scrollHeight - panel.clientHeight - 2;
        if((dy > 0 && !atBot) || (dy < 0 && !atTop)){
          return; /* let the panel scroll natively */
        }
      }
    }
    e.preventDefault();

    if(tdir === 'h'){
      var track = sections[tai].querySelector('.track');
      var hMax = (panelCounts[tai] - 1) * track.clientWidth;
      hTargets[tai] = Math.max(0, Math.min(hMax, ths + dx * 2));
      scheduleHSnap(tai);
    } else {
      vTarget = Math.max(0, Math.min(maxVScroll, ts + dy * 2));
      scheduleSnap();
    }
  }, {passive:false});

  /* ── animate ── */
  function animate(){
    vCurrent += (vTarget - vCurrent) * 0.06;
    if(Math.abs(vCurrent - vTarget) < 0.5) vCurrent = vTarget;
    var curIdx = Math.round(vCurrent / vh);

    for(var i = 0; i < N; i++){
      var offset = i * vh - vCurrent;
      var progress = offset / vh;

      /* vertical stacking */
      if(offset <= 0){
        var depth = Math.abs(progress);
        var sc = Math.max(0.92, 1 - depth * 0.04);
        var op = Math.max(0.15, 1 - depth * 0.45);
        var br = Math.max(0.65, 1 - depth * 0.18);
        sections[i].style.transform = 'translate3d(0,0,0) scale('+sc+')';
        sections[i].style.opacity = op;
        sections[i].style.filter = 'brightness('+br+')';
        sections[i].style.borderRadius = (depth * 12) + 'px';
      } else {
        sections[i].style.transform = 'translate3d(0,'+offset+'px,0)';
        sections[i].style.opacity = 1;
        sections[i].style.filter = 'none';
        sections[i].style.borderRadius = '0';
      }

      /* horizontal track */
      hCurrents[i] += (hTargets[i] - hCurrents[i]) * 0.07;
      if(Math.abs(hCurrents[i] - hTargets[i]) < 0.5) hCurrents[i] = hTargets[i];
      var track = sections[i].querySelector('.track');
      if(track){
        track.style.transform = 'translate3d('+ (-hCurrents[i]) +'px,0,0)';
        var tw = track.clientWidth;
        /* per-panel parallax */
        var panels = track.querySelectorAll('.panel');
        for(var p = 0; p < panels.length; p++){
          var pOff = p * tw - hCurrents[i];
          var pProg = pOff / tw;
          var imgW = panels[p].querySelector('.panel-img-wrap');
          var meta = panels[p].querySelector('.panel-meta');
          if(imgW){
            if(Math.abs(pProg) < 0.01){
              imgW.style.transform = 'none';
            } else {
              imgW.style.transform = 'translate3d('+(pProg*80)+'px,'+(pProg*15)+'px,0)';
            }
          }
          if(meta){
            var mop = Math.max(0.08, 1 - Math.abs(pProg) * 0.7);
            if(Math.abs(pProg) < 0.01){
              meta.style.transform = 'none';
              meta.style.opacity = 1;
            } else {
              meta.style.transform = 'translate3d('+(pProg*50)+'px,'+(pProg*-10)+'px,0)';
              meta.style.opacity = mop;
            }
          }
        }
      }

      /* panel dots */
      var pdots = sections[i].querySelectorAll('.pdot');
      var curP = tw ? Math.round(hCurrents[i] / tw) : 0;
      for(var d = 0; d < pdots.length; d++){
        pdots[d].classList.toggle('active', d === curP);
      }
    }

    /* section dots */
    for(var d = 0; d < sdots.length; d++){
      sdots[d].classList.toggle('active', d === curIdx);
    }
    /* nav tabs */
    for(var d = 0; d < navTabs.length; d++){
      navTabs[d].classList.toggle('active', d === curIdx);
    }
    counter.textContent = String(curIdx+1).padStart(2,'0')+' / '+String(N).padStart(2,'0');

    requestAnimationFrame(animate);
  }
  animate();

  /* section dot click */
  sdots.forEach(function(d, i){
    d.addEventListener('click', function(){ vTarget = i * vh; });
  });
  /* nav tab click */
  navTabs.forEach(function(t){
    t.addEventListener('click', function(){
      var si = parseInt(t.getAttribute('data-si'));
      vTarget = si * vh;
    });
  });
  /* panel dot click */
  document.querySelectorAll('.pdot').forEach(function(d){
    d.addEventListener('click', function(){
      var si = parseInt(d.getAttribute('data-si'));
      var pi = parseInt(d.getAttribute('data-pi'));
      var track = sections[si].querySelector('.track');
      hTargets[si] = pi * track.clientWidth;
    });
  });
  /* panel next button click */
  document.querySelectorAll('.panel-next-btn').forEach(function(btn){
    btn.addEventListener('click', function(){
      var si = parseInt(btn.getAttribute('data-si'));
      var pi = parseInt(btn.getAttribute('data-pi'));
      var track = sections[si].querySelector('.track');
      hTargets[si] = pi * track.clientWidth;
    });
  });
  /* panel prev button click */
  document.querySelectorAll('.panel-prev-btn').forEach(function(btn){
    btn.addEventListener('click', function(){
      var si = parseInt(btn.getAttribute('data-si'));
      var pi = parseInt(btn.getAttribute('data-pi'));
      var track = sections[si].querySelector('.track');
      hTargets[si] = pi * track.clientWidth;
    });
  });
})();
"""

    js = js.replace("PANEL_COUNTS", panel_counts_js)

    # ── Symptoms modal HTML ──
    symptoms_modal = (
        '<div id="symptoms-modal" class="sym-modal" onclick="if(event.target===this)this.style.display=\'none\'">'
        '<div class="sym-modal-content">'
        '<span class="sym-modal-close" onclick="document.getElementById(\'symptoms-modal\').style.display=\'none\'">&times;</span>'
        '<h2 class="sym-modal-title">Full PPPD Symptoms Breakdown</h2>'
        '<div class="sym-modal-scroll">'
        '<table class="sym-table">'
        '<thead><tr>'
        '<th>Symptom Domain</th><th>Description</th><th>What It Feels Like</th><th>When It Worsens</th><th>Clinical Notes</th>'
        '</tr></thead><tbody>'
        '<tr><td>Core dizziness</td><td>Non\u2011spinning, rocking, swaying, or floating dizziness</td>'
        '<td>Feeling like being on a boat, walking on a trampoline, or \u201cinternally moving\u201d even when still</td>'
        '<td>Prolonged standing/walking, fatigue, stress</td>'
        '<td>Present most days for \u22653\u00a0months, often for hours; may wax and wane</td></tr>'
        '<tr><td>Unsteadiness</td><td>Swaying, wobbling, or veering when walking</td>'
        '<td>Sense that the body may tip, drift sideways, or \u201cwalk like drunk\u201d</td>'
        '<td>Standing, walking, turning quickly, open/uneven spaces</td>'
        '<td>Patients may avoid walking unaided or in unfamiliar environments</td></tr>'
        '<tr><td>Postural dependence</td><td>Symptoms worsen when upright</td>'
        '<td>Standing or sitting feels \u201cheavy,\u201d precarious, or more \u201cfloaty\u201d than lying down</td>'
        '<td>Upright posture, queues, showering, standing at a counter</td>'
        '<td>Lying down often gives partial relief, highlighting postural pattern</td></tr>'
        '<tr><td>Visual motion sensitivity</td><td>Dizziness triggered by complex or moving visual scenes</td>'
        '<td>Feeling dizzy, off\u2011balance, or \u201coverwhelmed\u201d in visually busy places</td>'
        '<td>Supermarkets, crowds, escalators, scrolling screens, patterned floors</td>'
        '<td>Often called \u201cvisual vertigo\u201d; patients may fixate on the floor for stability</td></tr>'
        '<tr><td>Self\u2011motion sensitivity</td><td>Dizziness when moving head or body</td>'
        '<td>Rocking or \u201cinside motion\u201d when bending, looking up, turning, or riding in a car</td>'
        '<td>Rapid head turns, exercise, sudden position changes, travel</td>'
        '<td>Differs from BPPV because episodes are prolonged, not brief positional spins</td></tr>'
        '<tr><td>Elevator\u2011drop sensation</td><td>Sudden brief \u201celevator\u2011drop\u201d or sinking feeling</td>'
        '<td>Momentary feeling the floor drops away or body suddenly sinks</td>'
        '<td>Sudden posture changes, fatigue, stress, sometimes at rest</td>'
        '<td>Can be startling; overlaps with internal motion and autonomic sensations</td></tr>'
        '<tr><td>Diurnal pattern</td><td>Persistent, fluctuating symptoms over the day</td>'
        '<td>\u201cBackground\u201d dizziness most of the time, often milder in morning, worse later</td>'
        '<td>End of day, after cognitive/social load, stressful day</td>'
        '<td>Chronic but not uniformly severe; flare\u2011ups after exertion, stress, or poor sleep</td></tr>'
        '<tr><td>Cognitive symptoms</td><td>Brain fog, slowed thinking, poor concentration</td>'
        '<td>Trouble focusing, processing information, or remembering details</td>'
        '<td>During symptom flares, high stress, prolonged standing/visual exposure</td>'
        '<td>Can be misinterpreted as a primary cognitive disorder; linked to dizziness and fatigue</td></tr>'
        '<tr><td>Perceptual detachment</td><td>Derealisation or depersonalisation</td>'
        '<td>Feeling the world is unreal or watching yourself from outside your body</td>'
        '<td>Busy visual environments, high anxiety, strong dizziness episodes</td>'
        '<td>Common in chronic dizziness and anxiety; not psychosis but highly distressing</td></tr>'
        '<tr><td>Anxiety &amp; hypervigilance</td><td>Over\u2011monitoring of body sensations and balance</td>'
        '<td>Constantly checking for dizziness, scanning for instability, worrying about collapsing</td>'
        '<td>Before or during challenging situations (crowds, travel, work)</td>'
        '<td>Anxiety both follows and maintains symptoms; many patients have secondary anxiety</td></tr>'
        '<tr><td>Fatigue</td><td>Physical and mental exhaustion</td>'
        '<td>Feeling \u201cwiped out,\u201d needing to lie down, reduced stamina</td>'
        '<td>After bad symptom days, prolonged standing, stressful events</td>'
        '<td>Fatigue and dizziness reinforce each other, reducing activity and promoting deconditioning</td></tr>'
        '<tr><td>Sleep disturbance</td><td>Poor or unrefreshing sleep</td>'
        '<td>Difficulty falling asleep, or waking still tired and foggy</td>'
        '<td>Nights after stressful or symptom\u2011intense days</td>'
        '<td>Poor sleep amplifies dizziness, cognitive issues, and irritability</td></tr>'
        '<tr><td>Autonomic\u2011like sensations</td><td>Lightheadedness, internal \u201cvibrations,\u201d wooziness</td>'
        '<td>Feeling faint, weak, \u201cshaky inside,\u201d or heavy\u2011headed</td>'
        '<td>Standing still, heat, after exertion, during stress</td>'
        '<td>May overlap with dysautonomia; exclude true cardiovascular causes when indicated</td></tr>'
        '<tr><td>Emotional impact</td><td>Frustration, low mood, feeling misunderstood</td>'
        '<td>Feeling others \u201ccan\u2019t see\u201d the problem or think it is \u201cjust anxiety\u201d</td>'
        '<td>After unhelpful medical encounters, loss of work/roles, social isolation</td>'
        '<td>Drives social withdrawal and reduced quality of life if not addressed</td></tr>'
        '<tr><td>Functional impact</td><td>Reduced participation in daily activities</td>'
        '<td>Avoiding supermarkets, transport, exercise, or social events</td>'
        '<td>Any situation perceived as risky, busy, or unpredictable</td>'
        '<td>Over\u2011avoidance prevents sensory recalibration and reinforces fear, maintaining PPPD</td></tr>'
        '</tbody></table>'
        '</div></div></div>'
    )

    # ── Causes modal HTML ──
    causes_modal = (
        '<div id="causes-modal" class="sym-modal" onclick="if(event.target===this)this.style.display=\'none\'">'
        '<div class="sym-modal-content">'
        '<span class="sym-modal-close" onclick="document.getElementById(\'causes-modal\').style.display=\'none\'">&times;</span>'
        '<h2 class="sym-modal-title">Full Breakdown of PPPD Causes</h2>'
        '<div class="sym-modal-scroll">'
        '<table class="sym-table">'
        '<thead><tr>'
        '<th>Trigger / Category</th><th>Example Trigger</th><th>What Happens in the Acute Phase</th><th>How It Transitions into PPPD</th><th>Notes</th>'
        '</tr></thead><tbody>'
        '<tr><td>Peripheral vestibular disorders</td>'
        '<td>BPPV, vestibular neuritis, M\u00e9ni\u00e8re\u2019s disease, other inner\u2011ear problems</td>'
        '<td>Sudden spinning vertigo, imbalance, sometimes nausea and falls of confidence in walking or standing</td>'
        '<td>Even after the ear condition stabilises, the brain remains \u201con high alert\u201d for motion and posture, so non\u2011spinning dizziness, swaying, and visual motion sensitivity become persistent</td>'
        '<td>Ear\u2011related tests may normalise, but symptoms do not fully settle, leading to a PPPD phenotype</td></tr>'
        '<tr><td>Central vestibular / neurological events</td>'
        '<td>Stroke, transient ischaemic attack, or other brain\u2011stem or cerebellar events affecting balance</td>'
        '<td>Episodes of vertigo, gait ataxia, or sudden instability, often with objective neurological signs on examination</td>'
        '<td>After the acute phase resolves, patients continue to feel unsteady or \u201coff\u2011balance,\u201d with heightened attention to posture and visual cues</td>'
        '<td>If over\u2011vigilance and avoidance persist, this can evolve into a chronic PPPD\u2011like pattern</td></tr>'
        '<tr><td>Mild traumatic brain injury / whiplash</td>'
        '<td>Concussion, sports\u2011related head injury, or whiplash\u2011type neck injury</td>'
        '<td>Dizziness, brain fog, and imbalance after the trauma, sometimes with only mild or no clear abnormality on imaging</td>'
        '<td>When the person develops chronic hypervigilance about dizziness and avoids movement or visually busy environments, the brain\u2019s postural control system becomes maladaptively tuned</td>'
        '<td>This can meet formal criteria for PPPD even without classic \u201cspinning\u201d vertigo</td></tr>'
        '<tr><td>Vestibular migraine</td>'
        '<td>Migraine\u2011associated vertigo or dizziness, often with headache, aura, light/sound sensitivity</td>'
        '<td>Recurrent episodes of vertigo or profound dizziness, sometimes with nausea and motion sensitivity during attacks</td>'
        '<td>Between attacks, the brain may remain sensitised to motion and visual stimuli, and patients may develop persistent non\u2011spinning dizziness and avoidance behaviours</td>'
        '<td>This can overlap with PPPD and may be labelled as \u201cPPPD\u2011related\u201d or comorbid</td></tr>'
        '<tr><td>Panic attacks / psychological stress</td>'
        '<td>Acute panic attack, severe anxiety episode, or major life stressor</td>'
        '<td>Dizziness, palpitations, shortness of breath, and fear of fainting or losing control; vestibular tests often normal</td>'
        '<td>After the panic episode, the person may remain fearful of bodily sensations, scanning for dizziness or \u201csigns of collapse,\u201d which sustains the symptom pattern</td>'
        '<td>Over time, this can match PPPD criteria, especially if symptoms are chronic and postural/visual motion\u2011dependent</td></tr>'
        '<tr><td>Dysautonomia / autonomic dysfunction</td>'
        '<td>Postural orthostatic tachycardia syndrome (POTS) or other autonomic issues</td>'
        '<td>Lightheadedness, floatiness, or \u201cfoggy\u201d sensations, often worse when upright, with or without clear heart\u2011rate or blood\u2011pressure changes</td>'
        '<td>Persistent fatigue and postural lightheadedness may lead to cautious, stiff postures and avoidance of standing or movement</td>'
        '<td>This can feed into a PPPD\u2011like phenotype, particularly if over\u2011reliance on visual cues develops</td></tr>'
        '<tr><td>Gradual or \u201cno\u2011clear\u201d onset</td>'
        '<td>Insidious worsening of unsteadiness without a clear vertigo episode</td>'
        '<td>Symptoms creep in over weeks or months, sometimes misattributed to ageing, fatigue, or \u201cjust stress\u201d</td>'
        '<td>Later, reassessment often reveals that subtle vestibular or psychological triggers were present, but the onset is labelled as \u201cinsidious\u201d</td>'
        '<td>Important to differentiate from progressive neurological disorders while still validating the patient\u2019s experience</td></tr>'
        '</tbody></table>'
        '</div></div></div>'
    )

    html = (
        "<!DOCTYPE html><html><head>"
        '<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400'
        '&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">'
        "<style>" + css + "</style></head><body>"
        '<div class="container">'
        '<div class="nav-tabs">' + nav_tabs_html + '</div>'
        '<div class="counter" id="counter">01 / '
        + str(n).zfill(2)
        + "</div>"
        + sections_html
        + '<div class="sec-dots">'
        + sec_dots_html
        + "</div>"
        "</div>"
        + symptoms_modal
        + causes_modal
        + "<script>" + js + "</script>"
        "</body></html>"
    )
    components.html(html, height=height, scrolling=False)


# ── MAIN TABS ──
tab1, tab2, tab3 = st.tabs(
    ["📖 What is PPPD", "📊 Analytics & Treatment Prediction", "📚 Resources & References"]
)


# ── TAB 1 ──
with tab1:
    # ── All sections as full-screen layered scroll ──
    layered_scroll(
        [
            {
                "title": "What is PPPD?",
                "subtitle": "Understanding Persistent Postural-Perceptual Dizziness",
                "html": '<div class="intro-content">'
                '<p class="intro-desc"><strong>Persistent Postural Perceptual Dizziness (PPPD)</strong> '
                "is a chronic functional vestibular (inner\u2011ear) disorder and one of the most common causes of prolonged "
                "dizziness and unsteadiness. Rather than classic room\u2011spinning vertigo, it typically causes "
                "a non\u2011spinning dizziness\u2014often described as rocking, swaying, or a \u201cfloaty\u201d "
                "sensation\u2014along with a persistent feeling of being off\u2011balance.</p>"
                '<p class="intro-desc">PPPD is thought to arise from a '
                "subtle mismatch in how the brain combines input from the visual system, the inner ear "
                "(vestibular system), and body position sensors. Each system may work normally on its own, "
                "but the coordination between them becomes disrupted.</p>"
                '<p class="intro-desc">PPPD is formally recognised in the <strong>International Classification '
                "of Vestibular Disorders</strong>, helping clinicians distinguish it from structural inner\u2011ear "
                "or neurological conditions.</p>"
                '<div class="intro-cards">'
                '<div class="iflip"><div class="iflip-inner">'
                '<div class="iflip-f">Persistent (P)</div>'
                '<div class="iflip-b">Symptoms present most days for ≥3 months, often lasting hours with fluctuating intensity rather than occurring briefly or episodically.</div>'
                "</div></div>"
                '<div class="iflip"><div class="iflip-inner">'
                '<div class="iflip-f">Postural (P)</div>'
                '<div class="iflip-b">Symptoms typically worsen when upright, walking, or changing posture, and may ease when lying down or in low-stimulus environments.</div>'
                "</div></div>"
                '<div class="iflip"><div class="iflip-inner">'
                '<div class="iflip-f">Perceptual (P)</div>'
                '<div class="iflip-b">Patients feel unsteady, swaying, or moving despite normal or near-normal objective balance and gait testing.</div>'
                "</div></div>"
                '<div class="iflip"><div class="iflip-inner">'
                '<div class="iflip-f">Dizziness (D)</div>'
                '<div class="iflip-b">The core sensation is non-spinning dizziness—rocking, bobbing, floating, or "brain fog"—rather than true rotational vertigo.</div>'
                "</div></div>"
                "</div>"
                '<p class="intro-desc" style="margin-top:1.2rem;">This name emphasises that PPPD is driven by altered <strong>perception</strong> and <strong>postural processing</strong>, rather than a structural &ldquo;lesion&rdquo; in the inner ear or brain.</p>'
                "</div>",
            },
            {
                "title": "Typical Symptoms",
                "subtitle": "Common presentations in PPPD patients",
                "panels": [
                    {
                        "title": "Internal Motion",
                        "tag": "SYMPTOM",
                        "img": "app/static/img/1.png",
                        "body": "<p>A constant or near\u2011constant feeling of rocking, swaying, or floating, which is often more noticeable when standing or walking and when fatigued or anxious.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'symptoms-modal\').style.display=\'flex\';return false;">See full PPPD symptoms breakdown &rarr;</a></p>',
                    },
                    {
                        "title": "Unsteadiness",
                        "tag": "SYMPTOM",
                        "img": "app/static/img/2.jpg",
                        "body": "<p>A sense of unsteadiness or veering when walking, as if the ground is moving or the body may drift sideways, even though falls are rare and performance on clinical tests may appear relatively normal.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'symptoms-modal\').style.display=\'flex\';return false;">See full PPPD symptoms breakdown &rarr;</a></p>',
                    },
                    {
                        "title": "Visual Provocation",
                        "tag": "SYMPTOM",
                        "img": "app/static/img/3.jpg",
                        "body": "<p>Marked worsening in visually busy environments\u2014supermarkets, crowds, scrolling screens, traffic, patterned floors\u2014due to over-reliance on visual motion cues for balance.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'symptoms-modal\').style.display=\'flex\';return false;">See full PPPD symptoms breakdown &rarr;</a></p>',
                    },
                    {
                        "title": "Cognitive Symptoms",
                        "tag": "SYMPTOM",
                        "img": "app/static/img/4.jpg",
                        "body": "<p>Cognitive symptoms including brain fog, poor concentration, slowed thinking, and detachment from surroundings, especially during flare-ups or high stress.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'symptoms-modal\').style.display=\'flex\';return false;">See full PPPD symptoms breakdown &rarr;</a></p>',
                    },
                    {
                        "title": "The Hidden Burden",
                        "tag": "",
                        "hide_meta": True,
                        "img": "app/static/img/5.jpg",
                        "body": "<p>These symptoms are real and often disabling, yet patients typically look well on the outside, and routine vestibular tests and imaging frequently come back normal or near\u2011normal\u2014which can lead to great frustration and concerns about not being taken seriously.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'symptoms-modal\').style.display=\'flex\';return false;">See full PPPD symptoms breakdown &rarr;</a></p>',
                    },
                ],
            },
            {
                "title": "How it Starts",
                "subtitle": "The path from trigger to chronic disorder",
                "panels": [
                    {
                        "title": "Common Triggers",
                        "tag": "ONSET",
                        "img": "app/static/img/6.jpg",
                        "body": "<p>PPPD typically develops after an acute or episodic vestibular, neurological, or psychological event that causes sudden vertigo, dizziness, or imbalance, such as:</p>"
                        "<ul>"
                        "<li>A vestibular disorder (for example, <a href='https://www.webmd.com/brain/benign-paroxysmal-positional-vertigo' target='_blank'>benign paroxysmal positional vertigo</a>, <a href='https://www.webmd.com/brain/vestibular-neuritis' target='_blank'>vestibular neuritis</a>, or <a href='https://www.webmd.com/brain/menieres-disease' target='_blank'>M\u00e9ni\u00e8re\u2019s disease</a>).</li>"
                        "<li><a href='https://www.webmd.com/migraines-headaches/vestibular-migraine' target='_blank'>Vestibular migraine</a>, concussion, or mild traumatic brain injury.</li>"
                        "<li>A panic attack, severe anxiety episode, or a period of intense psychological stress that triggers acute dizziness or fear of falling.</li>"
                        "</ul>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'causes-modal\').style.display=\'flex\';return false;">See full breakdown of PPPD causes &rarr;</a></p>',
                    },
                    {
                        "title": "The Threat Response",
                        "tag": "ONSET",
                        "img": "app/static/img/7.jpg",
                        "body": "<p>During the initial episode, the body\u2019s \u201cthreat\u201d system (often described as fight\u2011or\u2011flight) becomes highly activated, leading to increased vigilance toward posture, movement, and visual cues.</p>"
                        "<p>The brain shifts into a protective mode, constantly monitoring balance and spatial orientation as though danger is still present.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'causes-modal\').style.display=\'flex\';return false;">See full breakdown of PPPD causes &rarr;</a></p>',
                    },
                    {
                        "title": "Why It Persists",
                        "tag": "ONSET",
                        "img": "app/static/img/8.jpg",
                        "body": "<p>In many people, balance recalibrates over time and the threat response fades. In PPPD, however, the nervous system does not fully reset.</p>"
                        "<p>This over\u2011sensitive, threat\u2011oriented pattern of processing becomes persistent, maintaining the dizziness and unsteadiness over months to years.</p>"
                        '<p class="symptom-link"><a href="#" onclick="document.getElementById(\'causes-modal\').style.display=\'flex\';return false;">See full breakdown of PPPD causes &rarr;</a></p>',
                    },
                ],
            },
            {
                "title": "Diagnostic Criteria",
                "subtitle": "The 2017 B\u00e1r\u00e1ny Society consensus",
                "panels": [
                    {
                        "title": "Persistent Symptoms",
                        "tag": "DIAGNOSIS",
                        "img": "app/static/img/9.jpg",
                        "body": "<p>International diagnostic criteria (<a href='https://www.jvr-web.org/ICVD.html' target='_blank'>B\u00e1r\u00e1ny Society</a> / <a href='https://www.jvr-web.org/ICVD.html' target='_blank'>International Classification of Vestibular Disorders</a>) define PPPD by the following criteria:</p>"
                        "<p>One or more symptoms of dizziness, unsteadiness, or non\u2011spinning vertigo present on most days for \u22653\u00a0months, usually for prolonged (hours\u2011long) periods, even though severity may fluctuate.</p>",
                    },
                    {
                        "title": "What Makes It Worse",
                        "tag": "DIAGNOSIS",
                        "img": "app/static/img/10.jpg",
                        "body": "<p>Symptoms that are consistently aggravated by:</p>"
                        "<ul>"
                        "<li>upright posture,</li>"
                        "<li>active or passive motion (regardless of direction), and</li>"
                        "<li>exposure to complex or moving visual stimuli.</li>"
                        "</ul>"
                        "<p>Symptoms are initiated by an acute or episodic vestibular, neurological, or psychological event that caused vertigo or imbalance.</p>",
                    },
                    {
                        "title": "Confirming the Diagnosis",
                        "tag": "DIAGNOSIS",
                        "img": "app/static/img/11.jpg",
                        "body": "<p>Symptoms must cause significant distress or functional impairment and are not better explained by another disease (such as progressive neurological disorders or severe structural vestibular disease).</p>"
                        "<p>Diagnosis is primarily clinical, based on a detailed history and examination, often after structural or neurodegenerative causes of dizziness have been ruled out by ENT, neurology, or vestibular specialists.</p>",
                    },
                ],
            },
            {
                "title": "Treatment Approaches",
                "subtitle": "Evidence\u2011based therapeutic strategies",
                "panels": [
                    {
                        "title": "Vestibular Rehab (VRT)",
                        "tag": "TREATMENT",
                        "img": "app/static/img/12.jpg",
                        "body": "<p>PPPD is treatable, although recovery is often gradual and may be incomplete in some patients. Evidence\u2011informed management typically includes:</p>"
                        "<p>A structured programme of balance and gaze\u2011stabilisation exercises tailored to the individual, aimed at recalibrating the vestibular system and reducing visual motion sensitivity and avoidance behaviours.</p>",
                    },
                    {
                        "title": "Psychological Therapies",
                        "tag": "TREATMENT",
                        "img": "app/static/img/13.png",
                        "body": "<p>Cognitive\u2011behavioural or psychologically informed therapies to address hypervigilance to bodily sensations, anxiety, fear\u2011of\u2011falling, and maladaptive avoidance patterns that maintain symptoms and limit engagement in daily activities.</p>",
                    },
                    {
                        "title": "Medication",
                        "tag": "TREATMENT",
                        "img": "app/static/img/14.jpg",
                        "body": "<p>In selected cases, SSRIs, SNRIs, or other neuromodulating agents may be used to help regulate central vestibular and emotional processing and to treat co\u2011existing anxiety, depression, or vestibular migraine.</p>",
                    },
                ],
            },
        ]
    )


# ── TAB 2 ──
DHI_QUESTIONS = [
    "Does looking up increase your dizziness?",
    "Because of your problem, do you feel frustrated?",
    "Because of your problem, do you restrict your travel for business or recreation?",
    "Does walking down the aisle of a supermarket increase your dizziness?",
    "Because of your problem, do you have difficulty getting into or out of bed?",
    "Does your problem significantly restrict your participation in social activities?",
    "Because of your problem, do you have difficulty reading?",
    "Does performing ambitious activities (e.g., dancing) increase your dizziness?",
    "Because of your problem, are you afraid to leave your home without someone accompanying you?",
    "Because of your problem, have you been embarrassed in front of others?",
    "Do quick movements of your head increase your dizziness?",
    "Because of your problem, do you avoid heights?",
    "Does turning over in bed increase your dizziness?",
    "Because of your problem, is it difficult for you to do strenuous housework?",
    "Because of your problem, are you afraid people might think you are intoxicated?",
    "Because of your problem, is it difficult for you to go for a walk by yourself?",
    "Does walking down a sidewalk increase your dizziness?",
    "Because of your problem, is it difficult for you to concentrate?",
    "Because of your problem, is it difficult for you to walk around the house in the dark?",
    "Because of your problem, are you afraid to stay home alone?",
    "Because of your problem, do you feel handicapped?",
    "Has your problem placed stress on your relationships with members of your family or friends?",
    "Because of your problem, are you depressed?",
    "Does your problem interfere with your job or household responsibilities?",
    "Does bending over increase your dizziness?",
]

with tab2:
    st.markdown(
        '<h2 class="tab-heading" style="margin-bottom:2.5rem">Analytics &amp; Treatment Prediction</h2>',
        unsafe_allow_html=True,
    )

    # ── Step tracker ──
    if "tab2_step" not in st.session_state:
        st.session_state.tab2_step = 1

    step_labels = [
        "Clinical Info",
        "DHI Questionnaire",
        "Prediction Results",
    ]
    current = st.session_state.tab2_step

    # Build labels row
    labels_row = '<div class="step-labels">'
    for i, label in enumerate(step_labels):
        step_num = i + 1
        cls = "active" if step_num == current else ("done" if step_num < current else "")
        labels_row += f'<span class="sl {cls}">{label}</span>'
    labels_row += '</div>'

    # Build dots row
    dots_row = '<div class="step-dots">'
    for i in range(len(step_labels)):
        step_num = i + 1
        cls = "active" if step_num == current else ("done" if step_num < current else "")
        dots_row += f'<span class="sd {cls}"></span>'
    dots_row += '</div>'

    st.markdown(
        """
        <style>
        .step-labels{display:flex;justify-content:space-between;padding:0 0.5rem;
                     margin-bottom:0.4rem;}
        .sl{flex:1;text-align:center;font-family:'Playfair Display',serif;
            font-size:0.82rem;font-weight:600;color:#bbb;line-height:1.25;}
        .sl.active{color:#111;font-weight:800;}
        .sl.done{color:#888;}
        .step-dots{display:flex;align-items:center;padding:0 0.5rem;
                   position:relative;height:18px;margin-bottom:0.8rem;}
        .step-dots::before{content:'';position:absolute;left:calc(12.5% + 7px);
                           right:calc(12.5% + 7px);top:50%;height:2px;
                           background:#ccc;transform:translateY(-50%);}
        .sd{flex:1;display:flex;justify-content:center;position:relative;z-index:1;}
        .sd::after{content:'';width:14px;height:14px;border-radius:50%;
                   border:2px solid #ccc;background:#fff;display:block;}
        .sd.active::after{background:#111;border-color:#111;}
        .sd.done::after{background:#111;border-color:#111;}
        </style>
        """
        + labels_row + dots_row,
        unsafe_allow_html=True,
    )

    # ── DHI setup (needed for score computation in all steps) ──
    physical_qs = DHI_PHYSICAL_QS
    emotional_qs = DHI_EMOTIONAL_QS
    functional_qs = DHI_FUNCTIONAL_QS

    answers = [0] * len(DHI_QUESTIONS)
    options = ["No (0)", "Sometimes (2)", "Yes (4)"]
    score_map = {"No (0)": 0, "Sometimes (2)": 2, "Yes (4)": 4}

    # Always collect DHI answers (hidden widgets persist in session_state)
    for idx in range(len(DHI_QUESTIONS)):
        key = f"dhi_{idx}"
        if key in st.session_state:
            answers[idx] = score_map[st.session_state[key]]

    # ═══════════════════════════════════════════════════════
    # STEP 1a — Demographics & Clinical Inputs
    # ═══════════════════════════════════════════════════════
    if st.session_state.tab2_step == 1:
        _pad_l, form_col, _pad_r = st.columns([1, 2, 1])
        with form_col:
            st.markdown(
                '<h3 style="text-align:center;margin-bottom:0.5rem">🩺 Patient Clinical Information</h3>',
                unsafe_allow_html=True,
            )
            st.caption("Enter the patient's demographics and clinical details below.")
            st.markdown('<div style="margin-bottom:1.8rem"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                age = st.number_input("Age", 18, 85, 38)
                anx_level = st.select_slider("Anxiety/Distress", options=range(11), value=5)
                vis_level = st.select_slider("Visual Sensitivity", options=range(11), value=6)
                symptom_dur = st.number_input(
                    "Symptom Duration (months)", 1, 72, 14,
                    help="How many months the patient has experienced PPPD symptoms"
                )
                trigger_count = st.selectbox(
                    "Number of Triggers", options=[1, 2, 3, 4, 5], index=1,
                    help="Known triggers: vestibular event, visual motion, head movement, etc."
                )
                migraine = st.selectbox(
                    "Migraine Comorbidity", options=["No", "Yes"], index=0,
                    help="Does the patient have comorbid migraine? (~35% of PPPD patients)"
                )
                anxiety_disorder = st.selectbox(
                    "Anxiety Disorder", options=["No", "Yes"], index=0,
                    help="Diagnosed anxiety disorder? (~40% of PPPD patients)"
                )

            # ── Input validation ──
            _validation_warnings = []
            if age < 18 or age > 85:
                _validation_warnings.append("Age is outside the typical adult PPPD range (18–85).")
            if symptom_dur < 3:
                _validation_warnings.append(
                    "PPPD diagnosis requires symptoms persisting ≥ 3 months (Staab et al., 2017)."
                )
            if trigger_count == 0:
                _validation_warnings.append(
                    "PPPD typically has at least one identifiable trigger."
                )
            for w in _validation_warnings:
                st.warning(w)

            st.markdown("")
            _bl, btn_col, _br = st.columns([1, 2, 1])
            with btn_col:
                if st.button("Continue to DHI Questionnaire →", type="primary", use_container_width=True):
                    st.session_state.tab2_step = 2
                    st.rerun()

    # ── Persist demographics values for steps > 1 (hidden inputs) ──
    if st.session_state.tab2_step > 1:
        age = st.session_state.get("Age", 38)
        anx_level = st.session_state.get("Anxiety/Distress", 5)
        vis_level = st.session_state.get("Visual Sensitivity", 6)
        symptom_dur = st.session_state.get("Symptom Duration (months)", 14)
        trigger_count = st.session_state.get("Number of Triggers", 2)
        migraine = st.session_state.get("Migraine Comorbidity", "No")
        anxiety_disorder = st.session_state.get("Anxiety Disorder", "No")

    # ═══════════════════════════════════════════════════════
    # STEP 1b — DHI Questionnaire
    # ═══════════════════════════════════════════════════════
    if st.session_state.tab2_step == 2:
        _dl, dhi_col, _dr = st.columns([0.5, 4, 0.5])
        with dhi_col:
            st.markdown("<h3 style='text-align:center;white-space:nowrap'>📝 DHI (Dizziness Handicap Inventory) Questionnaire</h3>", unsafe_allow_html=True)
            st.caption(
                "The **Dizziness Handicap Inventory (DHI)** measures the self-perceived impact of dizziness. "
                "Answer each question: **Yes (4)**, **Sometimes (2)**, or **No (0)**. "
                "Your total DHI score is calculated automatically."
            )
            st.markdown('<div style="margin-bottom:1.5rem"></div>', unsafe_allow_html=True)

            st.markdown("##### Physical (10 questions)")
            for idx in physical_qs:
                col_q, col_a = st.columns([3, 1])
                with col_q:
                    st.markdown(f"**{idx + 1}.** {DHI_QUESTIONS[idx]}")
                with col_a:
                    ans = st.selectbox(
                        f"Q{idx + 1}", options, key=f"dhi_{idx}",
                        label_visibility="collapsed",
                    )
                answers[idx] = score_map[ans]

            st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
            st.divider()
            st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)
            st.markdown("##### Emotional (9 questions)")
            for idx in emotional_qs:
                col_q, col_a = st.columns([3, 1])
                with col_q:
                    st.markdown(f"**{idx + 1}.** {DHI_QUESTIONS[idx]}")
                with col_a:
                    ans = st.selectbox(
                        f"Q{idx + 1}", options, key=f"dhi_{idx}",
                        label_visibility="collapsed",
                    )
                answers[idx] = score_map[ans]

            st.markdown('<div style="margin-top:2rem"></div>', unsafe_allow_html=True)
            st.divider()
            st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)
            st.markdown("##### Functional (6 questions)")
            for idx in functional_qs:
                col_q, col_a = st.columns([3, 1])
                with col_q:
                    st.markdown(f"**{idx + 1}.** {DHI_QUESTIONS[idx]}")
                with col_a:
                    ans = st.selectbox(
                        f"Q{idx + 1}", options, key=f"dhi_{idx}",
                        label_visibility="collapsed",
                    )
                answers[idx] = score_map[ans]

            st.divider()

            base_dhi = sum(answers)
            phys_score = sum(answers[i] for i in physical_qs)
            emot_score = sum(answers[i] for i in emotional_qs)
            func_score = sum(answers[i] for i in functional_qs)

            if base_dhi <= DHI_THRESHOLD_MILD:
                severity = "Mild"
            elif base_dhi <= DHI_THRESHOLD_MODERATE:
                severity = "Moderate"
            else:
                severity = "Severe"

            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Calculated DHI Score", f"{base_dhi} / 100")
            with col_m2:
                st.metric("Severity Classification", severity)
                st.caption("Jacobson & Newman (1990): Mild 0-30 | Moderate 31-60 | Severe 61-100")

            st.markdown("")
            col_back1b, col_next1b = st.columns(2)
            with col_back1b:
                if st.button("← Back to Clinical Info", use_container_width=True):
                    st.session_state.tab2_step = 1
                    st.rerun()
            with col_next1b:
                if st.button("Continue to Prediction Results →", type="primary", use_container_width=True):
                    st.session_state.tab2_step = 3
                    st.rerun()

    # Compute scores for use in later steps
    base_dhi = sum(answers)
    phys_score = sum(answers[i] for i in physical_qs)
    emot_score = sum(answers[i] for i in emotional_qs)
    func_score = sum(answers[i] for i in functional_qs)
    if base_dhi <= DHI_THRESHOLD_MILD:
        severity = "Mild"
    elif base_dhi <= DHI_THRESHOLD_MODERATE:
        severity = "Moderate"
    else:
        severity = "Severe"

    # ═══════════════════════════════════════════════════════
    # STEP 2 (3) — Prediction Results
    # ═══════════════════════════════════════════════════════
    if st.session_state.tab2_step == 3:
        _pl, pred_col, _pr = st.columns([0.5, 4, 0.5])
        with pred_col:
            st.markdown("<h3 style='text-align:center;margin-bottom:0.8rem'>📊 Prediction Results</h3>", unsafe_allow_html=True)
            st.warning(
                "⚠️ **Proof-of-concept:** These predictions are based on simulated clinical data "
                "derived from published PPPD literature, not real patient outcomes. "
                "They should not be used for actual clinical decision-making."
            )
            st.caption(
                "The predictions below show estimated DHI scores after **rehabilitation-based** "
                "interventions (VRT and VR-enhanced VRT). The clinical recommendations further "
                "below consider the **full treatment picture**, including CBT, medication, and "
                "migraine management where indicated."
            )

            if model is not None:
                mig_val = 1 if migraine == "Yes" else 0
                anx_dis_val = 1 if anxiety_disorder == "Yes" else 0
                input_features = np.array([[age, base_dhi, anx_level, vis_level,
                                             symptom_dur, trigger_count, mig_val, anx_dis_val]])
                preds = model.predict(input_features)[0]
                vrt_rate, vr_rate = float(preds[0]), float(preds[1])
                vrt_final = round(base_dhi * (1 - np.clip(vrt_rate, 0, 1)), 1)
                vr_final = round(base_dhi * (1 - np.clip(vr_rate, 0, 1)), 1)
                vrt_drop = round(base_dhi - vrt_final, 1)
                vr_drop = round(base_dhi - vr_final, 1)

                feat_names = ["Age", "Baseline DHI", "Anxiety", "Visual Sens.",
                              "Symptom Duration", "Trigger Count", "Migraine", "Anxiety Disorder"]

                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                col_k1, col_k2, col_k3 = st.columns(3)
                with col_k1:
                    st.metric("Baseline DHI", base_dhi)
                with col_k2:
                    st.metric(
                        "After Standard Rehabilitation (VRT)",
                        vrt_final,
                        delta=f"-{vrt_drop} pts" if vrt_drop > 0 else f"+{abs(vrt_drop)} pts",
                        delta_color="inverse",
                    )
                with col_k3:
                    st.metric(
                        "After VR-Enhanced Rehabilitation",
                        vr_final,
                        delta=f"-{vr_drop} pts" if vr_drop > 0 else f"+{abs(vr_drop)} pts",
                        delta_color="inverse",
                    )

                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                st.divider()

                st.markdown(
                    '<p style="text-align:center;margin-top:0.5rem;margin-bottom:0.5rem;font-size:1.1rem;font-weight:700">'
                    '<a href="#clinical-recommendations" style="color:#111;text-decoration:none" '
                    'onmouseover="this.style.textDecoration=\'underline\'" '
                    'onmouseout="this.style.textDecoration=\'none\'">'
                    'Jump to Clinical Recommendations ↓</a></p>',
                    unsafe_allow_html=True,
                )

                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                st.divider()
                st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)

                # ── Section: Predicted DHI After Rehabilitation ──
                st.markdown("#### Predicted DHI After Rehabilitation")
                fig = go.Figure()
                fig.add_trace(
                    go.Bar(
                        x=["Baseline", "Standard Rehab (VRT)", "VR-Enhanced Rehab"],
                        y=[base_dhi, vrt_final, vr_final],
                        marker_color=["#94a3b8", "#3b82f6", "#10b981"],
                        text=[base_dhi, vrt_final, vr_final],
                        textposition="outside",
                    )
                )
                fig.update_layout(
                    yaxis_title="DHI Score (lower = better)",
                    yaxis=dict(range=[0, 105]),
                )
                _chart_cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines","toImage"]}
                st.plotly_chart(fig, use_container_width=True, config=_chart_cfg)
                st.caption(
                    "Lower scores mean less impact from dizziness. "
                    "**Mild** = 0–30 | **Moderate** = 31–60 | **Severe** = 61–100."
                )

                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                st.divider()
                st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)

                # ── Section: Where Dizziness Affects You Most ──
                st.markdown("#### Where Dizziness Affects You Most")
                radar_fig = go.Figure()
                max_phys = len(physical_qs) * 4
                max_emot = len(emotional_qs) * 4
                max_func = len(functional_qs) * 4
                radar_fig.add_trace(
                    go.Scatterpolar(
                        r=[
                            phys_score / max_phys * 100,
                            emot_score / max_emot * 100,
                            func_score / max_func * 100,
                            phys_score / max_phys * 100,
                        ],
                        theta=["Physical", "Emotional", "Functional", "Physical"],
                        fill="toself",
                        fillcolor="rgba(59,130,246,0.15)",
                        line=dict(color="#3b82f6", width=2),
                        name="Patient Profile",
                    )
                )
                radar_fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 100],
                               ticksuffix="%")),
                    height=400,
                )
                st.plotly_chart(radar_fig, use_container_width=True, config=_chart_cfg)
                st.caption(
                    "**Physical** = dizziness triggered by movement or posture. "
                    "**Emotional** = frustration, embarrassment, or anxiety from dizziness. "
                    "**Functional** = difficulty with daily activities. "
                    "A larger shape means greater impact."
                )

                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                st.divider()
                st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)

                # ── Section: What's Driving This Prediction? ──
                st.markdown("#### What's Driving This Prediction?")
                st.caption(
                    "This chart shows which patient factors most increased (red) or "
                    "decreased (blue) the expected treatment benefit compared to an "
                    "average patient."
                )
                with st.spinner("Generating SHAP explanation..."):
                    explainer = shap.TreeExplainer(model)
                    shap_vals = explainer(input_features)
                    shap_vrt = shap_vals[:, :, 0]
                    shap_vrt.feature_names = feat_names

                plt.rcParams.update({
                    "figure.facecolor": "#f5f0eb",
                    "axes.facecolor": "#f5f0eb",
                    "font.family": "sans-serif",
                    "font.size": 11,
                })
                fig_shap, ax_shap = plt.subplots(figsize=(7, 4))
                shap.plots.waterfall(shap_vrt[0], max_display=8, show=False)
                plt.tight_layout()
                st.pyplot(fig_shap, clear_figure=True)
                plt.rcParams.update(plt.rcParamsDefault)

                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                st.divider()
                st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)

                # ── Section: Clinical Recommendations ──
                st.markdown(
                    '<h4 id="clinical-recommendations">Clinical Recommendations</h4>',
                    unsafe_allow_html=True,
                )
                _has_rec = False

                # Anxiety-predominant: CBT / medication as primary
                if anx_level > 7 or anxiety_disorder == "Yes":
                    _has_rec = True
                    if anx_level > 7 and anxiety_disorder == "Yes":
                        st.error(
                            "**Anxiety-predominant presentation.** "
                            "Consider **CBT** as first-line treatment to address hypervigilance "
                            "and avoidance behaviours, with **SSRI/SNRI** pharmacotherapy "
                            "for co-existing anxiety disorder "
                            "(Popkirov et al., 2018; Staab et al., 2017). "
                            "VRT may be introduced once anxiety is better managed."
                        )
                    elif anxiety_disorder == "Yes":
                        st.warning(
                            "**Anxiety disorder comorbidity.** "
                            "Consider **SSRI/SNRI** pharmacotherapy alongside psychological "
                            "therapy (CBT) before or in parallel with vestibular rehabilitation "
                            "(Popkirov et al., 2018)."
                        )
                    else:
                        st.warning(
                            "**Elevated anxiety/distress.** "
                            "Consider **CBT** or psychologically informed therapy to reduce "
                            "hypervigilance and fear-avoidance before intensive VRT "
                            "(Popkirov et al., 2018)."
                        )

                # Migraine comorbidity: migraine management first
                if migraine == "Yes":
                    _has_rec = True
                    st.warning(
                        "**Migraine comorbidity detected.** "
                        "Prioritise **migraine prophylaxis** (lifestyle modification, "
                        "medication review) as uncontrolled migraine can sustain PPPD symptoms. "
                        "VR-based exposure should be introduced gradually and may need to be "
                        "limited during migraine flares (Micarelli et al., 2019)."
                    )

                # High visual sensitivity: graded visual desensitisation
                if vis_level > 7:
                    _has_rec = True
                    st.info(
                        "**High visual sensitivity.** "
                        "Consider **graded optokinetic / visual desensitisation** exercises "
                        "as a primary intervention. VR-based vestibular rehabilitation may be "
                        "a useful adjunct once baseline tolerance improves "
                        "(Micarelli et al., 2019)."
                    )

                # Chronic / long-duration PPPD
                if symptom_dur > 24:
                    _has_rec = True
                    st.info(
                        "**Long symptom duration (>24 months).** "
                        "Chronic PPPD often requires a **multimodal approach**: "
                        "combined vestibular rehabilitation, psychological therapy, "
                        "and potentially pharmacotherapy. Set realistic expectations — "
                        "recovery may be slower and partial "
                        "(Bittar & von Söhsten Lins, 2015; Staab et al., 2017)."
                    )

                # Prediction-driven: if model predicts poor VRT response
                if vrt_rate < 0.15:
                    _has_rec = True
                    st.warning(
                        "**Low predicted VRT response.** "
                        "The model predicts limited improvement from VRT alone. "
                        "Consider **pharmacotherapy** (SSRI/SNRI) and/or **CBT** as primary "
                        "or adjunct interventions before committing to intensive VRT "
                        "(Staab et al., 2017)."
                    )

                # Default: balanced recommendation (not just VRT)
                if not _has_rec:
                    st.success(
                        "**Favourable profile for vestibular rehabilitation.** "
                        "Standard VRT with customised balance and gaze-stabilisation "
                        "exercises is recommended as first-line treatment "
                        "(Whitney et al., 2016). Monitor progress and consider adding "
                        "psychological support or medication if response plateaus."
                    )

                # ── Download Report ──
                st.markdown('<div style="margin-top:1.5rem"></div>', unsafe_allow_html=True)
                st.divider()

                # Generate PDF report
                pdf = FPDF()
                pdf.add_page()
                pdf.set_auto_page_break(auto=True, margin=20)

                # Title
                pdf.set_font("Helvetica", "B", 18)
                pdf.cell(0, 12, "PPPD Clinical Decision Support", new_x="LMARGIN", new_y="NEXT", align="C")
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 8, "Patient Summary Report", new_x="LMARGIN", new_y="NEXT", align="C")
                pdf.ln(6)

                # Divider line
                pdf.set_draw_color(0)
                pdf.set_line_width(0.5)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(6)

                # Demographics
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 8, "Patient Demographics", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(95, 7, f"Age: {age}")
                pdf.cell(95, 7, f"Symptom Duration: {symptom_dur} months", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(95, 7, f"Anxiety Level: {anx_level}/10")
                pdf.cell(95, 7, f"Visual Sensitivity: {vis_level}/10", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(95, 7, f"Triggers: {trigger_count}")
                pdf.cell(95, 7, f"Migraine: {migraine}", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 7, f"Anxiety Disorder: {anxiety_disorder}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

                # DHI Assessment
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 8, "DHI Assessment", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(95, 7, f"Baseline DHI: {base_dhi} / 100")
                pdf.cell(95, 7, f"Severity: {severity}", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 7, f"Physical: {phys_score}   |   Emotional: {emot_score}   |   Functional: {func_score}", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(4)

                # Predictions
                pdf.set_font("Helvetica", "B", 13)
                pdf.cell(0, 8, "Predicted Outcomes", new_x="LMARGIN", new_y="NEXT")
                pdf.set_font("Helvetica", "", 10)
                pdf.cell(0, 7, f"After Standard VRT:       DHI {vrt_final}  (drop: {vrt_drop} pts)", new_x="LMARGIN", new_y="NEXT")
                pdf.cell(0, 7, f"After VR-enhanced VRT:  DHI {vr_final}  (drop: {vr_drop} pts)", new_x="LMARGIN", new_y="NEXT")
                pdf.ln(6)

                # Disclaimer
                pdf.set_draw_color(0)
                pdf.line(10, pdf.get_y(), 200, pdf.get_y())
                pdf.ln(4)
                pdf.set_font("Helvetica", "I", 8)
                pdf.multi_cell(0, 5,
                    "DISCLAIMER: Proof-of-concept only. Predictions are based on simulated "
                    "clinical data derived from published PPPD literature and should NOT be "
                    "used for real clinical decision-making."
                )

                pdf_bytes = pdf.output()
                st.download_button(
                    "📄 Download Patient Summary (PDF)",
                    data=bytes(pdf_bytes),
                    file_name="pppd_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            else:
                st.error("Model not loaded. Run train_model.py first.")

            st.markdown("")
            _bb_l, _bb_c, _bb_r = st.columns([1, 1, 1])
            with _bb_c:
                if st.button("← Back to DHI Questionnaire", type="primary", use_container_width=True):
                    st.session_state.tab2_step = 2
                    st.rerun()

            st.divider()
            with st.expander("🔬 Methodology & Data"):
                st.info(
                    "**Data Source:** Demographics from "
                    "[OpenNeuro ds004460](https://openneuro.org/datasets/ds004460/versions/1.1.0) "
                    "(Gramann et al., 2021 — 20 subjects). "
                    "Clinical scores (DHI, anxiety, visual sensitivity, symptom duration, "
                    "trigger count, comorbidities) are simulated from "
                    "published PPPD literature distributions "
                    "(Staab 2017; Steensnaes 2023; Popkirov 2018; Bittar 2015; Herdman 2020). "
                    "Model predicts **treatment response rates** from 8 clinical features, "
                    "trained on n=500 bootstrapped samples with 5-fold cross-validation."
                )

                # ── Model evaluation metrics ──
                _metrics_path = _DIR / "model_metrics.json"
                if _metrics_path.exists():
                    with open(_metrics_path) as _mf:
                        _metrics = json.load(_mf)

                    st.markdown("##### Model Comparison (5-fold CV)")
                    cv_data = _metrics.get("cv_comparison", {})
                    cv_rows = [
                        {"Model": name, "Mean R²": f"{v['mean_r2']:.4f}", "Std R²": f"± {v['std_r2']:.4f}"}
                        for name, v in cv_data.items()
                    ]
                    st.dataframe(pd.DataFrame(cv_rows), use_container_width=True, hide_index=True)

                    st.markdown("##### Test-Set Metrics")
                    test_data = _metrics.get("test_metrics", {})
                    test_rows = [
                        {"Target": t, "R²": f"{v['r2']:.4f}", "MAE": f"{v['mae']:.2f}", "RMSE": f"{v['rmse']:.2f}"}
                        for t, v in test_data.items()
                    ]
                    st.dataframe(pd.DataFrame(test_rows), use_container_width=True, hide_index=True)
                    st.caption(
                        f"Trained on {_metrics.get('n_train', '?')} samples · "
                        f"Tested on {_metrics.get('n_test', '?')} samples · "
                        f"Best CV model: {_metrics.get('best_model', '?')}"
                    )

                if participants_df is not None:
                    st.markdown("##### Source Cohort (OpenNeuro ds004460)")
                    col_tbl, col_demo = st.columns([1.2, 1])
                    with col_tbl:
                        st.dataframe(
                            participants_df[["participant_id", "age", "sex", "handedness"]],
                            use_container_width=True,
                            height=250,
                        )
                    with col_demo:
                        demo_fig = go.Figure()
                        demo_fig.add_trace(
                            go.Histogram(
                                x=participants_df["age"],
                                nbinsx=8,
                                marker_color="#3b82f6",
                                name="Age",
                            )
                        )
                        demo_fig.update_layout(
                            title="Cohort Age Distribution",
                            xaxis_title="Age (years)",
                            yaxis_title="Count",
                            height=250,
                            margin=dict(t=40, b=30, l=40, r=20),
                        )
                        _chart_cfg = {"displaylogo": False, "modeBarButtonsToRemove": ["zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines","toImage"]}
                        st.plotly_chart(demo_fig, use_container_width=True, config=_chart_cfg)

                        n_m = (participants_df["sex"] == "M").sum()
                        n_f = (participants_df["sex"] == "F").sum()
                        sex_fig = go.Figure(
                            data=[
                                go.Pie(
                                    labels=["Male", "Female"],
                                    values=[n_m, n_f],
                                    marker=dict(colors=["#3b82f6", "#ec4899"]),
                                    hole=0.45,
                                )
                            ]
                        )
                        sex_fig.update_layout(
                            title="Sex Distribution",
                            height=200,
                            margin=dict(t=40, b=10, l=10, r=10),
                        )
                        st.plotly_chart(sex_fig, use_container_width=True, config=_chart_cfg)

                if model is not None and hasattr(model, "feature_importances_"):
                    st.markdown("##### Model Feature Importance")
                    importances = model.feature_importances_
                    imp_fig = go.Figure()
                    imp_fig.add_trace(
                        go.Bar(
                            x=importances,
                            y=feat_names,
                            orientation="h",
                            marker_color="#334155",
                        )
                    )
                    imp_fig.update_layout(
                        title="What Drives the Prediction?",
                        xaxis_title="Importance",
                        height=250,
                        margin=dict(t=40, b=30, l=100, r=20),
                    )
                    _chart_cfg_m = {"displaylogo": False, "modeBarButtonsToRemove": ["zoom2d","pan2d","select2d","lasso2d","zoomIn2d","zoomOut2d","autoScale2d","resetScale2d","hoverClosestCartesian","hoverCompareCartesian","toggleSpikelines","toImage"]}
                    st.plotly_chart(imp_fig, use_container_width=True, config=_chart_cfg_m)

                st.markdown(
                    """
**Model:** Multi-output Random Forest Regressor (200 trees, max depth 8).
The model predicts **treatment response rates** (% DHI improvement) rather
than absolute post-treatment scores, ensuring it learns what drives treatment
success rather than just memorising baseline severity.

**Features (8):** Age, Baseline DHI, Anxiety Level, Visual Sensitivity,
Symptom Duration (months), Number of Triggers, Migraine Comorbidity,
Anxiety Disorder Comorbidity.

**Training data:** 500 bootstrapped samples from the OpenNeuro ds004460 cohort
(20 subjects). Clinical variables and treatment outcomes are simulated from
published PPPD literature:
- Baseline DHI distribution: Staab et al. (2017), Popkirov et al. (2018)
- Symptom duration & triggers: Bittar & von Söhsten Lins (2015), Staab et al. (2017)
- Comorbidity prevalence: Herdman et al. (2020)
- VRT effect sizes: Whitney et al. (2016), Steensnaes et al. (2023)
- VR-VRT effect sizes: Micarelli et al. (2019)

**Evaluation:** 5-fold cross-validation with Linear Regression, Random Forest,
and Gradient Boosting compared. Test-set metrics (R², MAE, RMSE) reported
during training. **Note:** Linear Regression achieved the highest single-target
CV R², but Random Forest was selected because the clinical task requires
**multi-output prediction** (both VRT and VR-VRT response rates simultaneously).
Scikit-learn's `RandomForestRegressor` natively supports multi-output regression,
whereas Linear Regression and Gradient Boosting would require a separate model
per target — increasing complexity and losing any cross-target structure.

**Limitations:**
- Clinical scores are simulated, not measured from real patient questionnaires.
- The source dataset contains healthy participants, not diagnosed PPPD patients.
- Predictions should not be used for real clinical decision-making.
- Sample size after bootstrap (n=500) is small by ML standards.
                """
                )

    # ── Spacer so content isn't clipped by fixed header ──
    st.markdown('<div style="height:10rem"></div>', unsafe_allow_html=True)


# ── TAB 3 ──
with tab3:
    st.markdown(
        '<h2 class="tab-heading">Resources &amp; References</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    **📚 References This Project Is Based On**

    * Staab et al. (2017) — Diagnostic criteria for PPPD — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9249299/)
    * Popkirov et al. (2018) — PPPD: a treatable cause of chronic dizziness — [BMJ](https://pn.bmj.com/content/18/1/5)
    * Steensnaes et al. (2023) — Vestibular rehabilitation for PPPD
    * Micarelli et al. (2019) — VR-enhanced vestibular rehabilitation
    * PPPD mechanisms & treatment review — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11272556/)
    * Conservative therapy for PPPD — [Frontiers](https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2025.1676218/full)
    * PPPD management & rehabilitation — [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11193666/)

    **📂 Dataset**

    * [OpenNeuro ds004460 v1.1.0](https://openneuro.org/datasets/ds004460/versions/1.1.0) — EEG + motion capture data
    * Gramann et al. (2021) — Cortical dynamics during heading changes

    ---

    **🏋️ Vestibular Rehabilitation Therapy (VRT)**

    * [VeDA – PPPD Guide](https://vestibular.org/article/diagnosis-treatment/types-of-vestibular-disorders/persistent-postural-perceptual-dizziness/) · Overall PPPD guide with VRT section
    * [VeDA – Provider Directory](https://vestibular.org/healthcare-directory/) · Find vestibular therapists
    * [Balance & Dizziness Canada](https://balanceanddizziness.org/what-is-the-treatment-for-pppd/) · Habituation & VRT exercises
    * [Cornerstone Physio – PPPD & VRT](https://cornerstonephysio.com/resources/pppd/) · Desensitization exercises
    * [VeDA – Article Collection (PDF)](https://vestibular.org/wp-content/uploads/2024/03/PPPD_Article-Collection.pdf) · Evidence for VRT

    **🧠 Cognitive Behavioral Therapy (CBT)**

    * [Manhattan CBT – PPPD](https://manhattancbt.com/pppd/) · CBT techniques for PPPD
    * [VeDA – Article Collection (PDF)](https://vestibular.org/wp-content/uploads/2024/03/PPPD_Article-Collection.pdf) · CBT evidence
    * [The Steady Coach](https://thesteadycoach.com/) · Somatic tracking & neuroplasticity
    * [The Steady Coach – Free Course](https://thesteadycoach.com/free-course/) · CBT-style recovery education

    **💊 SSRI / Medication Insights**

    * [University of Utah – PPPD Medication](https://medicine.utah.edu/neurology/education/dizzy-school/pppd/medication-management) · SSRI & SNRI dosing guide
    * [VeDA – Article Collection (PDF)](https://vestibular.org/wp-content/uploads/2024/03/PPPD_Article-Collection.pdf) · SSRI/SNRI trial summaries

    **🧘 Mind-Body & Meditation**

    * [The Steady Coach – YouTube](https://youtube.com/c/thesteadycoach) · Recovery tools & success stories
    * [Pain Free You – YouTube](https://youtube.com/@PainFreeYou) · TMS approach for PPPD
    * [Is This PDP?](https://isthispdp.com) · Mind-body self-assessment
    * [Guided Meditation for Dizziness](https://youtube.com/watch?v=5p2npA1p0LM) · Body scan for PPPD
    * [Parasympathetic Breathing](https://youtube.com/watch?v=AUoRvDUtC68) · Breathing for symptom flares
    * [Breathworks – Mindfulness](https://breathworks-mindfulness.org.uk/dizziness) · Vestibular mindfulness program
    * [Anxious Relief – PPPD Meditation](https://anxiousrelief.com/pppd-meditation/) · Calming the anxiety-dizziness cycle

    ---

    **🇭🇰 Relevance to Hong Kong**

    PPPD is increasingly recognised in Hong Kong's ageing population, with
    growing demand for vestibular rehabilitation services across the Hospital
    Authority. This project demonstrates how machine learning and clinical
    decision support tools can augment evidence-based care — aligning with
    Hong Kong's Smart Hospital initiatives and the HA's digital health strategy.
    Relevant local programmes include the MSc Health Analytics at HKU, CUHK's
    MSc in Health Data Science, and PolyU's MSc in Health Informatics.
    """
    )

    # ── Spacer so content isn't clipped by fixed header ──
    st.markdown('<div style="height:10rem"></div>', unsafe_allow_html=True)
