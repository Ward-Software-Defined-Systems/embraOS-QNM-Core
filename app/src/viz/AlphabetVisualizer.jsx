import React, { useEffect, useRef, useState } from "react";
import * as THREE from "three";

// ============================================================================
// Physics core — ported from sandbox/lie_poisson.py (so(3)* Lie–Poisson toy).
// State L on the sphere psi = |L|^2 = 1. Flow: Ldot = L x gradH.
// Integrator: RKMK2 (one rotation per step) => psi exact, never renormalized.
// ============================================================================
const SQ78 = Math.sqrt(78), SQ26 = Math.sqrt(26);
const D1 = [7 / SQ78, -2 / SQ78, -5 / SQ78];   // traceless part of diag(1/I), unit Frobenius
const D2 = [1 / SQ26, -4 / SQ26, 3 / SQ26];    // traceless diagonal, perp to D1
const INERTIA = [1, 2, 3];

const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const add3 = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale3 = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const norm3 = (a) => Math.sqrt(dot(a, a));

const gradH0 = (L) => [L[0] / INERTIA[0], L[1] / INERTIA[1], L[2] / INERTIA[2]];
const h0 = (L) =>
  0.5 * (L[0] * L[0] / INERTIA[0] + L[1] * L[1] / INERTIA[1] + L[2] * L[2] / INERTIA[2]);

// A blend over the base set S0 = {k1 k2 k3 | c12 c13 c23 | d1 d2}, times amplitude.
function buildSymbol(p) {
  const A = [
    [p.d1 * D1[0] + p.d2 * D2[0], p.c12, p.c13],
    [p.c12, p.d1 * D1[1] + p.d2 * D2[1], p.c23],
    [p.c13, p.c23, p.d1 * D1[2] + p.d2 * D2[2]],
  ];
  const a = [p.k1, p.k2, p.k3];
  const m = p.amp;
  return {
    grad: (L) => [
      m * (a[0] + A[0][0] * L[0] + A[0][1] * L[1] + A[0][2] * L[2]),
      m * (a[1] + A[1][0] * L[0] + A[1][1] * L[1] + A[1][2] * L[2]),
      m * (a[2] + A[2][0] * L[0] + A[2][1] * L[1] + A[2][2] * L[2]),
    ],
    h: (L) => {
      const AL = [
        A[0][0] * L[0] + A[0][1] * L[1] + A[0][2] * L[2],
        A[1][0] * L[0] + A[1][1] * L[1] + A[1][2] * L[2],
        A[2][0] * L[0] + A[2][1] * L[1] + A[2][2] * L[2],
      ];
      return m * (dot(a, L) + 0.5 * dot(L, AL));
    },
  };
}

// Rodrigues rotation of x by rotation-vector v (sinc form, small-angle guard) —
// exactly norm-preserving: this is the mechanism that makes psi structural.
function rotate(v, x) {
  const th = norm3(v);
  let sa, sb;
  if (th < 1e-8) {
    sa = 1 - (th * th) / 6;
    sb = 0.5 - (th * th) / 24;
  } else {
    sa = Math.sin(th) / th;
    sb = (1 - Math.cos(th)) / (th * th);
  }
  const c = Math.cos(th);
  const cv = cross(v, x);
  const vd = dot(v, x);
  return [
    x[0] * c + cv[0] * sa + v[0] * vd * sb,
    x[1] * c + cv[1] * sa + v[1] * vd * sb,
    x[2] * c + cv[2] * sa + v[2] * vd * sb,
  ];
}

// One explicit-midpoint step on the rotation group. Ldot = L x w = -w x L,
// so the rotation vector is -w*dt.
function rkmk2(gradFn, L, dt) {
  const Lh = rotate(scale3(gradFn(L), -0.5 * dt), L);
  return rotate(scale3(gradFn(Lh), -dt), L);
}

// Fibonacci sphere sampler (for the flow-arrow field).
function fibSphere(n) {
  const pts = [];
  const ga = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < n; i++) {
    const y = 1 - (2 * i + 1) / n;
    const r = Math.sqrt(Math.max(0, 1 - y * y));
    const th = ga * i;
    pts.push([Math.cos(th) * r, y, Math.sin(th) * r]);
  }
  return pts;
}

// Diverging colormap (cool -> parchment -> warm), posterized so the bands read
// as level sets — which on the sphere ARE the streamlines of the generator.
const N_BANDS = 14;
function heatColor(t) {
  const q = Math.min(N_BANDS - 1, Math.floor(t * N_BANDS)) / (N_BANDS - 1);
  const edge = Math.abs(t * N_BANDS - Math.round(t * N_BANDS));
  const line = edge < 0.07 ? 0.72 : 1.0; // etch thin contour lines between bands
  let r, g, b;
  if (q < 0.5) {
    const u = q / 0.5; // deep blue -> parchment
    r = 0.11 + u * (0.91 - 0.11);
    g = 0.23 + u * (0.89 - 0.23);
    b = 0.42 + u * (0.84 - 0.42);
  } else {
    const u = (q - 0.5) / 0.5; // parchment -> warm red
    r = 0.91 - u * (0.91 - 0.70);
    g = 0.89 - u * (0.89 - 0.23);
    b = 0.84 - u * (0.84 - 0.18);
  }
  return [r * line, g * line, b * line];
}

// The base set and the letter alphabet for the word console.
const ZERO = { k1: 0, k2: 0, k3: 0, c12: 0, c13: 0, c23: 0, d1: 0, d2: 0, amp: 1 };
const BASE_LETTERS = {
  x: { ...ZERO, k1: 1 }, y: { ...ZERO, k2: 1 }, z: { ...ZERO, k3: 1 },
  u: { ...ZERO, c12: 1 }, v: { ...ZERO, c13: 1 }, w: { ...ZERO, c23: 1 },
  d: { ...ZERO, d1: 1 }, e: { ...ZERO, d2: 1 },
};
const LETTER_LABEL = {
  x: "k\u2081", y: "k\u2082", z: "k\u2083",
  u: "c\u2081\u2082", v: "c\u2081\u2083", w: "c\u2082\u2083",
  d: "d\u2081", e: "d\u2082", q: "blend",
};
const PRESETS = [
  { name: "k\u2081", p: { ...ZERO, k1: 1 } },
  { name: "k\u2082", p: { ...ZERO, k2: 1 } },
  { name: "k\u2083", p: { ...ZERO, k3: 1 } },
  { name: "c\u2081\u2082", p: { ...ZERO, c12: 1 } },
  { name: "c\u2081\u2083", p: { ...ZERO, c13: 1 } },
  { name: "c\u2082\u2083", p: { ...ZERO, c23: 1 } },
  { name: "d\u2081", p: { ...ZERO, d1: 1 } },
  { name: "d\u2082", p: { ...ZERO, d2: 1 } },
  { name: "x (v1)", p: { ...ZERO, k1: 1, amp: 0.5 } },
  { name: "s (v1)", p: { ...ZERO, d1: 0.793, d2: 0.196 } },
  { name: "g (v1, trimmed)", p: { ...ZERO, d1: -0.245 } },
  { name: "clear", p: { ...ZERO } },
];

const DT = 0.01, TAU = 0.5, STEPS = Math.round(TAU / DT); // the pinned schedule

export default function AlphabetVisualizer() {
  const mountRef = useRef(null);
  const sparkRef = useRef(null);
  const three = useRef({});          // three.js objects
  const sim = useRef(null);          // word-playback state
  const paramsRef = useRef({ ...ZERO, amp: 1 });

  const [params, setParams] = useState({ ...ZERO, amp: 1 });
  const [idleAdditive, setIdleAdditive] = useState(true);
  const [showArrows, setShowArrows] = useState(true);
  const [showParticles, setShowParticles] = useState(
    !(typeof window !== "undefined" &&
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches)
  );
  const [word, setWord] = useState("xuzdw");
  const [playing, setPlaying] = useState(false);
  const [nowPlaying, setNowPlaying] = useState(null);
  const [silent, setSilent] = useState(false);
  const [psiDrift, setPsiDrift] = useState(null);
  const [h0Range, setH0Range] = useState(null);
  const [speed, setSpeed] = useState(3);

  const flagsRef = useRef({ idleAdditive, showArrows, showParticles, playing, speed });
  useEffect(() => {
    flagsRef.current = { idleAdditive, showArrows, showParticles, playing, speed };
  }, [idleAdditive, showArrows, showParticles, playing, speed]);

  // ---------- the displayed generator (idle vs playback phase) ----------
  const idleGenerator = () => {
    const sym = buildSymbol(paramsRef.current);
    if (flagsRef.current.idleAdditive) {
      return { grad: (L) => add3(gradH0(L), sym.grad(L)), h: (L) => h0(L) + sym.h(L) };
    }
    return sym;
  };

  const recolorSphere = (gen) => {
    const t = three.current;
    if (!t.sphereGeo) return;
    const pos = t.spherePos, col = t.sphereGeo.attributes.color.array;
    const n = pos.length / 3;
    let mn = Infinity, mx = -Infinity;
    const vals = t.hvals || (t.hvals = new Float64Array(n));
    for (let i = 0; i < n; i++) {
      const h = gen.h([pos[3 * i], pos[3 * i + 1], pos[3 * i + 2]]);
      vals[i] = h;
      if (h < mn) mn = h;
      if (h > mx) mx = h;
    }
    const flat = mx - mn < 1e-9;
    for (let i = 0; i < n; i++) {
      const tt = flat ? 0.5 : (vals[i] - mn) / (mx - mn);
      const [r, g, b] = heatColor(tt);
      col[3 * i] = r; col[3 * i + 1] = g; col[3 * i + 2] = b;
    }
    t.sphereGeo.attributes.color.needsUpdate = true;
  };

  const rearrow = (gen) => {
    const t = three.current;
    if (!t.arrowGeo) return;
    const pos = t.arrowGeo.attributes.position.array;
    let vmax = 0;
    for (let i = 0; i < t.arrowPts.length; i++) {
      const L = t.arrowPts[i];
      const v = cross(L, gen.grad(L));
      const s = norm3(v);
      if (s > vmax) vmax = s;
      const len = 0.085 * Math.tanh(1.4 * s);
      const tip = s > 1e-12 ? add3(L, scale3(v, len / s)) : L;
      pos[6 * i] = L[0] * 1.004; pos[6 * i + 1] = L[1] * 1.004; pos[6 * i + 2] = L[2] * 1.004;
      pos[6 * i + 3] = tip[0] * 1.004; pos[6 * i + 4] = tip[1] * 1.004; pos[6 * i + 5] = tip[2] * 1.004;
    }
    t.arrowGeo.attributes.position.needsUpdate = true;
    setSilent(vmax < 1e-6);
  };

  const refreshField = () => {
    if (sim.current && sim.current.active) return; // playback owns the field
    const gen = idleGenerator();
    recolorSphere(gen);
    rearrow(gen);
  };

  // ---------- word playback ----------
  const buildTimeline = (letters) => {
    const tl = [{ phase: "gap", steps: STEPS, letter: null }];
    for (const ch of letters) {
      const p = ch === "q" ? { ...paramsRef.current } : BASE_LETTERS[ch];
      tl.push({ phase: "event", steps: STEPS, letter: ch, sym: buildSymbol(p) });
      tl.push({ phase: "gap", steps: STEPS, letter: null });
    }
    return tl;
  };

  const startWord = () => {
    const letters = word.toLowerCase().split("").filter((c) => BASE_LETTERS[c] || c === "q");
    if (letters.length === 0) return;
    const t = three.current;
    const finished = sim.current && !sim.current.active;
    if (!sim.current || finished) {
      let g = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1];
      const nrm = norm3(g) || 1;
      g = scale3(g, 1 / nrm);
      sim.current = {
        active: true, genesis: g, L: g, timeline: buildTimeline(letters),
        seg: 0, stepInSeg: 0, trailN: 0, maxDrift: 0,
        h0hist: [], phasehist: [], h0min: Infinity, h0max: -Infinity,
      };
      t.trailGeo.setDrawRange(0, 0);
      if (t.genesisDot) {
        t.genesisDot.position.set(g[0], g[1], g[2]);
        t.genesisDot.visible = true;
      }
      applyPhaseField(sim.current.timeline[0]);
      setNowPlaying({ phase: "gap", letter: null });
    }
    setPlaying(true);
  };

  const applyPhaseField = (segd) => {
    const gen =
      segd.phase === "event"
        ? { grad: (L) => add3(gradH0(L), segd.sym.grad(L)), h: (L) => h0(L) + segd.sym.h(L) }
        : { grad: gradH0, h: h0 };
    recolorSphere(gen);
    rearrow(gen);
    return gen;
  };

  const resetWord = () => {
    setPlaying(false);
    setNowPlaying(null);
    setPsiDrift(null);
    setH0Range(null);
    if (sim.current) sim.current.active = false;
    sim.current = null;
    const t = three.current;
    if (t.trailGeo) t.trailGeo.setDrawRange(0, 0);
    if (t.genesisDot) t.genesisDot.visible = false;
    drawSpark();
    refreshField();
  };

  const drawSpark = () => {
    const cv = sparkRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const s = sim.current;
    if (!s || s.h0hist.length < 2) return;
    const n = s.h0hist.length;
    const lo = s.h0min, hi = s.h0max, span = Math.max(1e-12, hi - lo);
    ctx.fillStyle = "rgba(160,170,190,0.13)";
    let i0 = null;
    for (let i = 0; i < n; i++) {
      const ev = s.phasehist[i] === 1;
      if (ev && i0 === null) i0 = i;
      if ((!ev || i === n - 1) && i0 !== null) {
        ctx.fillRect((i0 / n) * W, 0, ((i - i0 + 1) / n) * W, H);
        i0 = null;
      }
    }
    ctx.strokeStyle = "#E2694A";
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    for (let i = 0; i < n; i++) {
      const xx = (i / (n - 1)) * W;
      const yy = H - 4 - ((s.h0hist[i] - lo) / span) * (H - 8);
      i === 0 ? ctx.moveTo(xx, yy) : ctx.lineTo(xx, yy);
    }
    ctx.stroke();
    ctx.strokeStyle = "#3FD68C";
    ctx.lineWidth = 1;
    ctx.setLineDash([2, 3]);
    ctx.beginPath();
    ctx.moveTo(0, H - 2.5);
    ctx.lineTo(W, H - 2.5);
    ctx.stroke();
    ctx.setLineDash([]);
  };

  // ---------- scene ----------
  useEffect(() => {
    const mount = mountRef.current;
    const W = mount.clientWidth, H = mount.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setSize(W, H);
    mount.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 100);
    const cam = { theta: 0.9, phi: 1.15, r: 3.1 };
    const placeCam = () => {
      camera.position.set(
        cam.r * Math.sin(cam.phi) * Math.cos(cam.theta),
        cam.r * Math.cos(cam.phi),
        cam.r * Math.sin(cam.phi) * Math.sin(cam.theta)
      );
      camera.lookAt(0, 0, 0);
    };
    placeCam();

    // the sphere (the Casimir leaf psi = 1)
    const sphereGeo = new THREE.SphereGeometry(1, 128, 96);
    const nV = sphereGeo.attributes.position.count;
    sphereGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(nV * 3), 3));
    const spherePos = sphereGeo.attributes.position.array.slice();
    const sphere = new THREE.Mesh(
      sphereGeo,
      new THREE.MeshBasicMaterial({ vertexColors: true })
    );
    scene.add(sphere);

    // faint great circles + axis dots for orientation
    const circleMat = new THREE.LineBasicMaterial({ color: 0x2a3547, transparent: true, opacity: 0.7 });
    for (let k = 0; k < 3; k++) {
      const pts = [];
      for (let i = 0; i <= 128; i++) {
        const a = (i / 128) * Math.PI * 2;
        const c = Math.cos(a) * 1.006, s = Math.sin(a) * 1.006;
        pts.push(k === 0 ? new THREE.Vector3(c, s, 0) : k === 1 ? new THREE.Vector3(c, 0, s) : new THREE.Vector3(0, c, s));
      }
      scene.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), circleMat));
    }
    const axisCols = [0x8fa8d8, 0x8fd8b0, 0xd8c58f];
    [[1.03, 0, 0], [0, 1.03, 0], [0, 0, 1.03]].forEach((p, i) => {
      const d = new THREE.Mesh(
        new THREE.SphereGeometry(0.017, 10, 10),
        new THREE.MeshBasicMaterial({ color: axisCols[i] })
      );
      d.position.set(p[0], p[1], p[2]);
      scene.add(d);
    });

    // flow arrows
    const arrowPts = fibSphere(260);
    const arrowGeo = new THREE.BufferGeometry();
    arrowGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(arrowPts.length * 6), 3));
    const arrowCol = new Float32Array(arrowPts.length * 6);
    for (let i = 0; i < arrowPts.length; i++) {
      arrowCol.set([0.16, 0.19, 0.25, 0.94, 0.96, 1.0], i * 6);
    }
    arrowGeo.setAttribute("color", new THREE.BufferAttribute(arrowCol, 3));
    const arrows = new THREE.LineSegments(
      arrowGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.9 })
    );
    scene.add(arrows);

    // ambient particles (advected by the displayed generator)
    const NP = 700;
    const pPos = new Float32Array(NP * 3);
    const pArr = [];
    for (let i = 0; i < NP; i++) {
      let v = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1];
      v = scale3(v, 1 / (norm3(v) || 1));
      pArr.push(v);
      pPos.set(v, i * 3);
    }
    const pGeo = new THREE.BufferGeometry();
    pGeo.setAttribute("position", new THREE.BufferAttribute(pPos, 3));
    const particles = new THREE.Points(
      pGeo,
      new THREE.PointsMaterial({ color: 0xa8bbd9, size: 0.015, transparent: true, opacity: 0.8, depthWrite: false })
    );
    scene.add(particles);

    // hero trail (the word being lived): blue in gaps, law-red in events
    const MAXT = 6000;
    const trailGeo = new THREE.BufferGeometry();
    trailGeo.setAttribute("position", new THREE.BufferAttribute(new Float32Array(MAXT * 3), 3));
    trailGeo.setAttribute("color", new THREE.BufferAttribute(new Float32Array(MAXT * 3), 3));
    trailGeo.setDrawRange(0, 0);
    const trail = new THREE.Line(
      trailGeo,
      new THREE.LineBasicMaterial({ vertexColors: true, linewidth: 2 })
    );
    scene.add(trail);
    const genesisDot = new THREE.Mesh(
      new THREE.SphereGeometry(0.02, 12, 12),
      new THREE.MeshBasicMaterial({ color: 0x3fd68c })
    );
    genesisDot.visible = false;
    scene.add(genesisDot);

    three.current = { renderer, scene, camera, sphereGeo, spherePos, arrowGeo, arrowPts, particles, pArr, pGeo, trailGeo, trail, genesisDot };

    // custom orbit — ~20 lines, no addon import. (npm's three DOES ship
    // addons/controls/OrbitControls; swap it in if touch pinch-zoom matters.)
    let drag = false, px = 0, py = 0;
    const el = renderer.domElement;
    const down = (e) => { drag = true; px = e.clientX; py = e.clientY; };
    const move = (e) => {
      if (!drag) return;
      cam.theta += (e.clientX - px) * 0.006;
      cam.phi = Math.min(2.9, Math.max(0.25, cam.phi + (e.clientY - py) * 0.006));
      px = e.clientX; py = e.clientY;
      placeCam();
    };
    const up = () => { drag = false; };
    const wheel = (e) => {
      e.preventDefault();
      cam.r = Math.min(7, Math.max(1.7, cam.r + e.deltaY * 0.0022));
      placeCam();
    };
    el.addEventListener("pointerdown", down);
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", up);
    el.addEventListener("wheel", wheel, { passive: false });

    const onResize = () => {
      const w = mount.clientWidth, h = mount.clientHeight;
      renderer.setSize(w, h);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    window.addEventListener("resize", onResize);

    refreshField();

    // ---------- animation loop ----------
    let raf;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const F = flagsRef.current;
      arrows.visible = F.showArrows;
      particles.visible = F.showParticles;

      // ambient particles follow whatever field is displayed
      const s = sim.current;
      const seg = s && s.active ? s.timeline[s.seg] : null;
      const genGrad = seg
        ? seg.phase === "event"
          ? (L) => add3(gradH0(L), seg.sym.grad(L))
          : gradH0
        : (() => { const g = idleGenerator(); return g.grad; })();
      if (F.showParticles) {
        for (let i = 0; i < pArr.length; i++) {
          pArr[i] = rkmk2(genGrad, pArr[i], DT * 1.6);
          pPos.set(pArr[i], i * 3);
        }
        pGeo.attributes.position.needsUpdate = true;
      }

      // word playback
      if (s && s.active && F.playing) {
        const tpos = trailGeo.attributes.position.array;
        const tcol = trailGeo.attributes.color.array;
        for (let k = 0; k < F.speed; k++) {
          const sg = s.timeline[s.seg];
          if (!sg) break;
          const gfun = sg.phase === "event" ? (L) => add3(gradH0(L), sg.sym.grad(L)) : gradH0;
          s.L = rkmk2(gfun, s.L, DT);
          const drift = Math.abs(dot(s.L, s.L) - 1);
          if (drift > s.maxDrift) s.maxDrift = drift;
          const hv = h0(s.L);
          s.h0hist.push(hv);
          s.phasehist.push(sg.phase === "event" ? 1 : 0);
          if (hv < s.h0min) s.h0min = hv;
          if (hv > s.h0max) s.h0max = hv;
          if (s.trailN < MAXT) {
            tpos.set([s.L[0] * 1.002, s.L[1] * 1.002, s.L[2] * 1.002], s.trailN * 3);
            tcol.set(sg.phase === "event" ? [0.89, 0.41, 0.29] : [0.42, 0.62, 0.92], s.trailN * 3);
            s.trailN++;
          }
          s.stepInSeg++;
          if (s.stepInSeg >= sg.steps) {
            s.stepInSeg = 0;
            s.seg++;
            const nx = s.timeline[s.seg];
            if (!nx) {
              s.active = false;
              setPlaying(false);
              setNowPlaying({ phase: "done", letter: null });
              break;
            }
            applyPhaseField(nx);
            setNowPlaying({ phase: nx.phase, letter: nx.letter });
          }
        }
        trailGeo.attributes.position.needsUpdate = true;
        trailGeo.attributes.color.needsUpdate = true;
        trailGeo.setDrawRange(0, s.trailN);
        genesisDot.visible = true;
        setPsiDrift(s.maxDrift);
        setH0Range(s.h0max - s.h0min);
        drawSpark();
      }
      renderer.render(scene, camera);
    };
    tick();
    requestAnimationFrame(onResize); // size sync once layout settles

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener("resize", onResize);
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", up);
      el.removeEventListener("pointerdown", down);
      el.removeEventListener("wheel", wheel);
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // recolor when the blend or the idle-view toggle changes (outside playback)
  useEffect(() => {
    paramsRef.current = params;
    refreshField();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, idleAdditive]);

  // ---------- UI ----------
  const setP = (key, val) => setParams((p) => ({ ...p, [key]: val }));
  const blendString = () => {
    const terms = [];
    const push = (v, name) => {
      if (Math.abs(v) > 1e-9)
        terms.push(`${terms.length && v > 0 ? "+ " : v < 0 ? "− " : ""}${Math.abs(v).toFixed(2)}·${name}`);
    };
    push(params.k1, "k₁"); push(params.k2, "k₂"); push(params.k3, "k₃");
    push(params.c12, "c₁₂"); push(params.c13, "c₁₃"); push(params.c23, "c₂₃");
    push(params.d1, "d₁"); push(params.d2, "d₂");
    return terms.length ? terms.join(" ") : "0 (no Hσ)";
  };

  const renderSlider = (id, label) => (
    <div className="flex items-center gap-2 py-0.5" key={id}>
      <span className="w-8 shrink-0 text-right mono dim2">{label}</span>
      <input
        type="range" min={-1.5} max={1.5} step={0.05} value={params[id]}
        onChange={(e) => setP(id, parseFloat(e.target.value))}
        className="flex-1" aria-label={label}
      />
      <span className="w-11 shrink-0 mono val">{params[id].toFixed(2)}</span>
    </div>
  );

  const chip = "px-2 py-1 rounded border text-xs mono transition-colors";

  return (
    <div className="app w-full h-screen flex flex-col overflow-hidden">
      <header className="flex items-center justify-between px-4 py-2 border-b hair shrink-0">
        <div className="flex items-baseline gap-3">
          <span className="font-semibold tracking-tight" style={{ fontSize: 15 }}>Σ₀ on the sphere</span>
          <span className="dim text-xs">embraOS · so(3)* toy · ψ = |L|²</span>
        </div>
        <div className="flex items-center gap-5 mono text-xs">
          <span>
            <span className="dim">ψ drift </span>
            <span className="g">{psiDrift === null ? "—" : psiDrift.toExponential(1)}</span>
          </span>
          <span>
            <span className="dim">H₀ range </span>
            <span className="r">{h0Range === null ? "—" : h0Range.toFixed(3)}</span>
          </span>
        </div>
      </header>

      <div className="flex flex-1 min-h-0">
        <div className="relative flex-1 min-w-0">
          <div ref={mountRef} className="absolute inset-0" />
          {silent && (
            <div className="absolute top-3 left-3 panel rounded px-2 py-1 text-xs mono" style={{ color: "#d8b45a" }}>
              SILENT — ∇H ≈ 0 (H is a function of ψ)
            </div>
          )}
          {nowPlaying && nowPlaying.phase !== "done" && (
            <div className="absolute top-3 right-3 panel rounded px-2 py-1 text-xs mono">
              {nowPlaying.phase === "event" ? (
                <span className="r">event · {nowPlaying.letter} ({LETTER_LABEL[nowPlaying.letter]})</span>
              ) : (
                <span className="b">free flow (gap)</span>
              )}
            </div>
          )}
          {nowPlaying && nowPlaying.phase === "done" && (
            <div className="absolute top-3 right-3 panel rounded px-2 py-1 text-xs mono g">
              word complete — ψ held
            </div>
          )}
          <div className="absolute bottom-3 left-3 right-3 dim text-xs leading-snug pointer-events-none">
            Color bands are level sets of the displayed generator — the streamlines of its autonomous
            flow. Events are additive: trajectories ride H₀ + Hσ (red trail), silence is H₀ alone
            (blue trail). Drag to orbit, scroll to zoom. Axis dots: +e₁ +e₂ +e₃.
          </div>
        </div>

        <aside className="w-[340px] shrink-0 border-l hair overflow-y-auto px-4 py-3 flex flex-col gap-4">
          <section>
            <div className="eyebrow mb-1.5">Presets</div>
            <div className="flex flex-wrap gap-1.5">
              {PRESETS.map((pr) => (
                <button
                  key={pr.name}
                  className={chip + " hair hover:border-[#3FD68C] hover:text-[#3FD68C]"}
                  onClick={() => setParams({ ...pr.p })}
                >
                  {pr.name}
                </button>
              ))}
            </div>
          </section>

          <section>
            <div className="eyebrow mb-1">Blend — Hσ over Σ₀</div>
            <div className="dim2 mb-1">ℓ=1 · reorient</div>
            {renderSlider("k1", "k₁")}{renderSlider("k2", "k₂")}{renderSlider("k3", "k₃")}
            <div className="dim2 mt-1.5 mb-1">ℓ=2 · cross-couple</div>
            {renderSlider("c12", "c₁₂")}{renderSlider("c13", "c₁₃")}{renderSlider("c23", "c₂₃")}
            <div className="dim2 mt-1.5 mb-1">ℓ=2 · reshape (d₁ = the self direction)</div>
            {renderSlider("d1", "d₁")}{renderSlider("d2", "d₂")}
            <div className="flex items-center gap-2 py-0.5 mt-1.5">
              <span className="w-8 shrink-0 text-right mono dim2">amp</span>
              <input type="range" min={0} max={2} step={0.05} value={params.amp}
                onChange={(e) => setP("amp", parseFloat(e.target.value))} className="flex-1" aria-label="amplitude" />
              <span className="w-11 shrink-0 mono val">{params.amp.toFixed(2)}</span>
            </div>
            <div className="mono text-xs mt-2 px-2 py-1.5 panel rounded leading-relaxed">
              <span className="dim">Hσ = </span>{params.amp.toFixed(2)}·({blendString()})
            </div>
          </section>

          <section>
            <div className="eyebrow mb-1.5">Display</div>
            <div className="flex flex-col gap-1 text-xs">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={idleAdditive} onChange={(e) => setIdleAdditive(e.target.checked)} />
                <span>Idle view includes H₀ (the event generator, as lived)</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showArrows} onChange={(e) => setShowArrows(e.target.checked)} />
                <span>Flow arrows</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={showParticles} onChange={(e) => setShowParticles(e.target.checked)} />
                <span>Ambient particles (advected live)</span>
              </label>
            </div>
          </section>

          <section>
            <div className="eyebrow mb-1.5">Word console</div>
            <div className="flex gap-1.5 mb-1.5">
              <input
                type="text" value={word} spellCheck={false}
                onChange={(e) => setWord(e.target.value.toLowerCase().replace(/[^xyzuvwdeq]/g, ""))}
                className="mono flex-1 px-2 py-1 rounded panel text-sm"
                aria-label="word over the alphabet"
              />
              {!playing ? (
                <button className={chip + " hair hover:border-[#3FD68C]"} onClick={startWord}>▶ play</button>
              ) : (
                <button className={chip + " hair hover:border-[#E2694A]"} onClick={() => setPlaying(false)}>⏸ pause</button>
              )}
              <button className={chip + " hair hover:border-[#6FA0E8]"} onClick={resetWord}>reset</button>
            </div>
            <div className="flex items-center gap-2 py-0.5">
              <span className="w-8 shrink-0 text-right mono dim2">spd</span>
              <input type="range" min={1} max={10} step={1} value={speed}
                onChange={(e) => setSpeed(parseInt(e.target.value))} className="flex-1" aria-label="playback speed" />
              <span className="w-11 shrink-0 mono val">{speed}×</span>
            </div>
            <div className="dim2 leading-relaxed mt-1">
              letters: x y z → k₁ k₂ k₃ · u v w → c₁₂ c₁₃ c₂₃ · d e → d₁ d₂ · q → your blend.
              Schedule: gap–event–gap, τ = 0.5, dt = 0.01. Genesis: <span className="g">green dot</span>.
              Reset, then play, for a fresh genesis.
            </div>
            <canvas ref={sparkRef} width={300} height={64} className="w-full mt-2 rounded panel" />
            <div className="dim2 mt-1">
              <span className="r">— H₀ along the word</span> · grey bands = events ·{" "}
              <span className="g">··· ψ (flat by construction)</span>
            </div>
          </section>

          <section className="pb-2">
            <div className="eyebrow mb-1">Notes</div>
            <div className="dim text-xs leading-relaxed">
              The integrator is one rotation per step (RKMK2), so ψ is exact and never renormalized —
              the drift counter above is the mechanism claim, live. Kicks alone close into so(3):
              rigid rotations, an authored ceiling. Any c or d content opens the composition tower.
              A flat sphere with the SILENT badge is the isotropy trap at symbol level.
            </div>
          </section>
        </aside>
      </div>
    </div>
  );
}
