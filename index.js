// metaverse-celestial — TSL port.
//
// The dome and the star field are now authored in Three.js Shading Language
// (TSL) as node graphs and rendered with NodeMaterial / PointsNodeMaterial.
// TSL materials require the unified renderer (WebGPURenderer from
// 'three/webgpu', which falls back to a WebGL backend automatically); the
// classic THREE.WebGLRenderer cannot run NodeMaterial.
//
// The public CelestialBodies API is unchanged from the GLSL version: same
// constructor options, same settings, same setters (setSun/setSkySun/
// setMoonPhase/setMoon/setPlanets/setStars/update/dispose). Only the
// renderer the host uses changes.

import * as THREE from 'three/webgpu';
import {
  Fn,
  uniform, uniformArray,
  vec2, vec3, vec4, float, int,
  abs, max, min, mix, clamp, smoothstep, step, saturate,
  dot, cross, normalize, length, pow, exp, log, sin, cos, sqrt,
  fract, floor, oneMinus,
  If, Loop, Discard,
  select, and, or,
  positionLocal,
  instancedBufferAttribute,
  uv,
} from 'three/tsl';

const DEFAULT_RADIUS = 430;

function directionFromAngles(elevationDeg, azimuthDeg) {
  const elevation = THREE.MathUtils.degToRad(elevationDeg);
  const azimuth = THREE.MathUtils.degToRad(azimuthDeg);
  const cosEl = Math.cos(elevation);
  return new THREE.Vector3(
    cosEl * Math.cos(azimuth),
    Math.sin(elevation),
    cosEl * Math.sin(azimuth),
  ).normalize();
}

// Build a sun direction that produces a given lunar phase as seen on the sky.
// phase: 0 = new moon, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter, 1 = new again.
// The sun is placed on a great circle around the moon; phase is the sun/moon elongation
// angle (0..2*pi). When phase=0.5 the sun sits opposite the moon -> fully lit disc.
function sunDirFromPhase(moonDir, phase) {
  const angle = phase * Math.PI * 2; // elongation from new
  // pick a stable "east" tangent on the sky so the terminator sweeps consistently
  const up = Math.abs(moonDir.y) > 0.92 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
  const east = new THREE.Vector3().crossVectors(up, moonDir).normalize();
  // sun = moonDir*cos(elongation) + east*sin(elongation); a great-circle sweep.
  const sun = moonDir.clone().multiplyScalar(Math.cos(angle))
    .addScaledVector(east, Math.sin(angle));
  return sun.normalize();
}

// ---------------------------------------------------------------------------
// CPU-side star generation (unchanged from the GLSL version: static points,
// seeded RNG, magnitude distribution pow(rand,4.5) -> "rare bright, common faint").
// Positions are baked at `radius` so the Points object (parented to the dome
// group, which follows the camera) projects correctly.
// ---------------------------------------------------------------------------
function mulberry32(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = Math.imul(s ^ (s >>> 15), s | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function generateStarAttributes(rng, count, sizeMul, brightMul, radius) {
  const positions = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const brights = new Float32Array(count);
  const temps = new Float32Array(count);
  for (let i = 0; i < count; i += 1) {
    const u = rng();
    const v = rng();
    const theta = u * Math.PI * 2;
    const z = v * 2 - 1;
    const r = Math.sqrt(Math.max(0.0, 1 - z * z));
    positions[i * 3] = r * Math.cos(theta) * radius;
    positions[i * 3 + 1] = z * radius;
    positions[i * 3 + 2] = r * Math.sin(theta) * radius;
    const mag = Math.pow(rng(), 4.5);
    sizes[i] = sizeMul * (0.6 + mag * 2.2);
    brights[i] = brightMul * (0.4 + mag * 0.9);
    temps[i] = rng();
  }
  return { positions, sizes, brights, temps };
}

// ===========================================================================
// TSL helper functions (GPU). These are 1:1 ports of the GLSL functions in the
// original dome shader, expressed as TSL Fn() node graphs. The NodeBuilder
// inlines Fn bodies and unrolls Loop() with literal-bounded counts, so the
// split-octave fbm3_3/4/5/6 trick from the GLSL (forced ANGLE unrolling) is no
// longer necessary — a single parameterized fbm called with a literal octave
// count unrolls the same way.
// ===========================================================================

// ---- hashing --------------------------------------------------------------
// vec3 -> vec3 in [0,1). Direct port of the GLSL hash33.
const hash33 = /*@__PURE__*/ Fn(([p]) => {
  const q = fract(p.mul(vec3(443.8975, 441.4231, 437.5125)));
  const d = dot(q, q.yzx.add(vec3(19.19, 19.19, 19.19)).add(7.31));
  return fract(q.xxy.add(q.yxx).mul(q.zyx).add(d));
  // NOTE: the GLSL is fract((p.xxy + p.yxx) * p.zyx) after mutating p with the
  // dot product; the .add(d) above is folded into the fract to match.
});

// ---- 3D simplex noise (Ashima / Stefan Gustavson) -------------------------
const mod289_3 = /*@__PURE__*/ Fn(([x]) => x.sub(floor(x.mul(1.0 / 289.0)).mul(289.0)));
const mod289_4 = /*@__PURE__*/ Fn(([x]) => x.sub(floor(x.mul(1.0 / 289.0)).mul(289.0)));
const permute4 = /*@__PURE__*/ Fn(([x]) => mod289_4(x.mul(34.0).add(1.0).mul(x)));
const taylorInvSqrt4 = /*@__PURE__*/ Fn(([r]) => float(1.79284291400159).sub(float(0.85373472095314).mul(r)));

const snoise = /*@__PURE__*/ Fn(([v]) => {
  const C = vec2(1.0 / 6.0, 1.0 / 3.0);
  const D = vec4(0.0, 0.5, 1.0, 2.0);
  const i = floor(v.add(dot(v, C.yyy)));
  const x0 = v.sub(i).add(dot(i, C.xxx));
  const g = step(x0.yzx, x0.xyz);
  const l = float(1.0).sub(g);
  const i1 = min(g.xyz, l.zxy);
  const i2 = max(g.xyz, l.zxy);
  const x1 = x0.sub(i1).add(C.xxx);
  const x2 = x0.sub(i2).add(C.yyy);
  const x3 = x0.sub(D.yyy);
  const iMod = mod289_3(i);
  const p = permute4(permute4(permute4(
    iMod.z.add(vec4(0.0, i1.z, i2.z, 1.0)))
    .add(iMod.y.add(vec4(0.0, i1.y, i2.y, 1.0))))
    .add(iMod.x.add(vec4(0.0, i1.x, i2.x, 1.0))));
  const n_ = float(0.142857142857);
  const ns = n_.mul(D.wyz).sub(D.xzx);
  const j = p.sub(floor(p.mul(ns.z).mul(ns.z)).mul(49.0));
  const x_ = floor(j.mul(ns.z));
  const y_ = floor(j.sub(x_.mul(7.0)));
  const x = x_.mul(ns.x).add(ns.yyyy);
  const y = y_.mul(ns.x).add(ns.yyyy);
  const h = float(1.0).sub(abs(x)).sub(abs(y));
  const b0 = vec4(x.xy, y.xy);
  const b1 = vec4(x.zw, y.zw);
  const s0 = floor(b0).mul(2.0).add(1.0);
  const s1 = floor(b1).mul(2.0).add(1.0);
  const sh = step(h, vec4(0.0)).negate();
  const a0 = b0.xzyw.add(s0.xzyw.mul(sh.xxyy));
  const a1 = b1.xzyw.add(s1.xzyw.mul(sh.zzww));
  const p0 = vec3(a0.xy, h.x);
  const p1 = vec3(a0.zw, h.y);
  const p2 = vec3(a1.xy, h.z);
  const p3 = vec3(a1.zw, h.w);
  const norm = taylorInvSqrt4(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  const pp0 = p0.mul(norm.x);
  const pp1 = p1.mul(norm.y);
  const pp2 = p2.mul(norm.z);
  const pp3 = p3.mul(norm.w);
  const m = max(float(0.6).sub(vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3))), 0.0);
  const mm = m.mul(m);
  return float(42.0).mul(dot(mm.mul(mm), vec4(dot(pp0, x0), dot(pp1, x1), dot(pp2, x2), dot(pp3, x3))));
});

// Fractal brownian motion with `octaves` octaves. `octaves` is a JS literal so
// the NodeBuilder unrolls the Loop. Each octave rotates the sample point by a
// fixed offset (matching the GLSL `+ vec3(11.7,3.1,7.9)` offset) so the octaves
// are decorrelated.
const fbm = /*@__PURE__*/ Fn(([p, octaves]) => {
  const v = float(0.0).toVar();
  const a = float(0.5).toVar();
  const pp = p.toVar();
  Loop({ start: int(0), end: int(octaves), type: 'int', name: 'o' }, () => {
    v.addAssign(a.mul(snoise(pp)));
    pp.assign(pp.mul(2.02).add(vec3(11.7, 3.1, 7.9)));
    a.mulAssign(0.5);
  });
  return v;
});

// ---- geometry helpers -----------------------------------------------------
// localOnDir: project `dir` onto the tangent plane of `center` -> 2D local coords.
const localOnDir = /*@__PURE__*/ Fn(([dir, center]) => {
  const up = abs(center.y).greaterThan(0.92).select(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 0.0));
  const right = normalize(cross(up, center));
  const top = normalize(cross(center, right));
  return vec2(dot(dir, right), dot(dir, top));
});

// angular distance on the sphere (chord-length form: sqrt(max(0, 2-2*a.b))).
const angularDist = /*@__PURE__*/ Fn(([a, b]) => sqrt(max(0.0, float(2.0).sub(float(2.0).mul(dot(a, b))))));

// ===========================================================================
// Dome material (MeshBasicNodeMaterial). The fragment node composites, far to
// near: Milky Way -> 4 planets -> moon, using source-over alpha compositing,
// exactly like the GLSL main(). The dome additively blends this accumulated
// coverage onto the scene.
// ===========================================================================

// Blackbody-ish stellar color for the Milky Way nebular tint (0..1 temperature).
const milkyWay = /*@__PURE__*/ Fn(([dir, uSkySunDir]) => {
  const gN = normalize(vec3(0.18, 0.42, 0.89));
  const band = dot(dir, gN);
  const w = exp(band.mul(band).negate().div(0.022));
  const n = fbm(dir.mul(3.1).add(41.0), 4).mul(0.5).add(0.5);
  const n2 = fbm(dir.mul(8.5).sub(13.0), 4).mul(0.5).add(0.5);
  const density = w.mul(float(0.45).add(float(0.95).mul(n))).mul(smoothstep(0.18, 0.62, n2));
  const col = mix(vec3(0.45, 0.55, 0.92), vec3(0.92, 0.82, 0.72), n);
  // faint magenta tint in dense knots
  const magentaTint = smoothstep(0.7, 0.95, n).mul(0.4);
  const col2 = mix(col, vec3(0.75, 0.55, 0.85), magentaTint);
  return col2.mul(density).mul(0.20);
});

const renderMilkyWay = /*@__PURE__*/ Fn(([dir, uSkySunDir]) => {
  // Night-gate: skip during daytime (sky sun above horizon).
  const daytime = uSkySunDir.y.greaterThan(0.0);
  const horizon = smoothstep(-0.04, 0.16, dir.y);
  const mw = milkyWay(dir, uSkySunDir).mul(horizon);
  const alpha = clamp(length(mw).mul(2.2), 0.0, 1.0).mul(horizon);
  const out = vec4(mw, alpha);
  // daytime or below-horizon -> zero contribution
  return daytime.select(vec4(0.0), horizon.lessThanEqual(0.0).select(vec4(0.0), out));
});

// ---- moon -----------------------------------------------------------------
const moonHeight = /*@__PURE__*/ Fn(([n]) => fbm(n.mul(2.2).add(31.0), 5).mul(0.9).add(fbm(n.mul(5.0).add(7.0), 3).mul(0.1)));

// crater mask: 1 in cratered highlands, ~0 in smooth maria.
const craterMask = /*@__PURE__*/ Fn(([n, terra]) => {
  const highland = smoothstep(0.0, 0.4, terra);
  const rough = smoothstep(0.2, 0.8, fbm(n.mul(7.0).add(53.0), 3).mul(0.5).add(0.5));
  return highland.mul(rough);
});

// cellular craters, density gated by the highland mask. TSL's Loop passes
// the loop index via the callback's argument object; we use that to derive
// the per-tier scale and probability.
const moonCratersImpl = /*@__PURE__*/ Fn(([n, gate]) => {
  const c = float(0.0).toVar();
  Loop({ start: int(0), end: int(2), type: 'int', name: 's' }, ({ s }) => {
    const sf = float(s);
    const scale = float(22.0).mul(pow(float(1.9), sf));
    const cb = floor(n.mul(scale));
    Loop({ start: int(0), end: int(3), type: 'int', name: 'ix' }, ({ ix }) => {
      Loop({ start: int(0), end: int(3), type: 'int', name: 'iy' }, ({ iy }) => {
        Loop({ start: int(0), end: int(3), type: 'int', name: 'iz' }, ({ iz }) => {
          const o = vec3(float(ix).sub(1.0), float(iy).sub(1.0), float(iz).sub(1.0));
          const cell = cb.add(o);
          const h = hash33(cell.add(sf.mul(23.7)).add(4.1));
          const prob = mix(float(0.07), float(0.035), sf);
          If(h.x.greaterThan(prob), () => { /* continue */ });
          // TSL has no per-iteration continue inside If without a stack Fn; we
          // instead gate the whole body with the inverse condition.
          const cp = normalize(cell.add(0.5).add(h.sub(0.5).mul(0.7)).div(scale));
          const dd = angularDist(n, cp);
          const sz = pow(h.y, 3.0);
          const r = float(0.12).add(sz.mul(0.55)).div(scale);
          const inside = dd.lessThanEqual(r.mul(1.15));
          const g = gate;
          const rim = smoothstep(r, r.mul(0.78), dd).mul(smoothstep(r.mul(0.4), r.mul(0.92), dd));
          const pit = smoothstep(r.mul(0.7), 0.0, dd);
          const contrib = rim.mul(0.14).sub(pit.mul(0.10)).mul(g);
          c.addAssign(h.x.greaterThan(prob).select(float(0.0), inside.select(contrib, float(0.0))));
        });
      });
    });
  });
  return c;
});

const renderMoon = /*@__PURE__*/ Fn(([dir, uMoonDir, uSunDir, uMoonSize, uMoonGlow, uMoonHorizonBoost]) => {
  const mDir = normalize(uMoonDir);
  const sunDir = normalize(uSunDir);
  // Moon illusion: horizon Moon perceived larger.
  const elev = clamp(mDir.y, 0.0, 1.0);
  const horizonFactor = pow(oneMinus(elev), 1.3);
  const sizeScale = float(1.0).add(uMoonHorizonBoost.mul(0.5).mul(horizonFactor));
  const angularSize = uMoonSize.mul(0.00125).mul(sizeScale);
  const dist = angularDist(dir, mDir);
  const disc = smoothstep(angularSize, angularSize.mul(0.93), dist);
  const glowR = angularSize.mul(2.4);
  const glow = exp(dist.mul(dist).negate().div(glowR.mul(glowR))).mul(uMoonGlow);

  const p = localOnDir(dir, mDir).div(angularSize);
  const d = length(p);
  const n = normalize(vec3(p, sqrt(max(0.0, oneMinus(d.mul(d))))));

  const terra = moonHeight(n);
  const maria = smoothstep(0.42, 0.66, fbm(n.mul(1.8).add(91.0), 4).mul(0.5).add(0.5));
  const gate = craterMask(n, terra).mul(oneMinus(maria.mul(0.9)));
  const craters = moonCratersImpl(n, gate);

  // orthonormal basis for relief shading
  const upChosen = abs(n.y).lessThan(0.94).select(vec3(0.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0));
  const tx = normalize(cross(upChosen, n));
  const ty = cross(n, tx);
  const e = float(0.06);
  const hx = moonHeight(n.add(tx.mul(e)));
  const hy = moonHeight(n.add(ty.mul(e)));
  const rn = normalize(n.sub(tx.mul(hx.sub(terra).div(e).mul(0.03))).sub(ty.mul(hy.sub(terra).div(e).mul(0.03))));

  // Lighting from the actual sun direction -> lunar phase emerges from geometry.
  // Full moon: sun opposite moon, lit normal is dot(rn, -sunDir).
  const lit = dot(rn, sunDir.negate());
  const day = smoothstep(-0.06, 0.16, lit);
  const lambert = max(0.0, lit);

  const mu = max(0.0, n.z);
  const limb = pow(mu, 0.55);

  const highland = mix(vec3(0.80, 0.77, 0.72), vec3(0.70, 0.68, 0.64), smoothstep(0.2, 0.7, terra.mul(0.5).add(0.5)));
  const mariaC = vec3(0.26, 0.26, 0.29);
  let base = mix(highland, mariaC, maria);
  base = base.add(craters);
  base = base.mul(float(0.92).add(float(0.08).mul(smoothstep(-0.2, 0.2, terra))));

  let color = base.mul(float(0.05).add(day.mul(1.9).mul(float(0.5).add(float(0.5).mul(lambert))))).mul(limb).mul(disc);
  const earthshine = oneMinus(day).mul(disc);
  color = color.add(vec3(0.05, 0.06, 0.085).mul(earthshine));

  // Atmospheric extinction near the horizon (Rayleigh).
  const airMass = float(1.0).div(max(elev, 0.04));
  const ext = clamp(float(0.18).mul(log(airMass)), 0.0, 0.85);
  color = vec3(
    color.r.mul(float(1.0).add(ext.mul(0.5))),
    color.g.mul(float(1.0).sub(ext.mul(0.25))),
    color.b.mul(float(1.0).sub(ext.mul(0.8))),
  );

  // Moon glow: soft halo OUTSIDE the disc.
  const glowColor = mix(vec3(0.62, 0.71, 0.9), vec3(0.85, 0.55, 0.35), ext);
  const halo = glow.mul(oneMinus(disc));
  color = color.add(glowColor.mul(halo));

  const alpha = max(disc, halo.mul(0.45));
  // Early-out equivalent: if disc<=0 and glow<=0.002, return zero.
  const dead = disc.lessThanEqual(0.0).and(glow.lessThanEqual(0.002));
  return dead.select(vec4(0.0), vec4(color, alpha));
});

// ---- planets --------------------------------------------------------------
const renderPlanet = /*@__PURE__*/ Fn(([dir, index, uPlanetsVisible, uPlanetScale, uPlanetGlow, uPlanetDir, uPlanetSize, uPlanetGlowBase, uPlanetColorA, uPlanetColorB, uPlanetBands, uTime]) => {
  const pDir = normalize(uPlanetDir.element(index));
  const angularSize = uPlanetSize.element(index).mul(uPlanetScale).mul(0.00068);
  const dist = angularDist(dir, pDir);
  const disc = smoothstep(angularSize, angularSize.mul(0.93), dist);
  const glowR = angularSize.mul(3.4);
  const glow = exp(dist.mul(dist).negate().div(glowR.mul(glowR))).mul(uPlanetGlowBase.element(index)).mul(uPlanetGlow);

  const lp = localOnDir(dir, pDir).div(angularSize);
  const dd = dot(lp, lp);
  const n = normalize(vec3(lp, sqrt(max(0.0, oneMinus(dd)))));
  const pid = float(index);

  const lightDir = normalize(vec3(-0.4, 0.32, 0.86));
  const lit = dot(n, lightDir);
  const day = smoothstep(-0.08, 0.18, lit);
  const lambert = max(0.0, lit);

  const terrain = fbm(n.mul(4.0).add(pid.mul(11.0)), 4).mul(0.5).add(0.5);
  const detail = fbm(n.mul(16.0).add(pid.mul(5.0)), 3).mul(0.5).add(0.5);
  const rocky = mix(uPlanetColorB.element(index), uPlanetColorA.element(index), smoothstep(0.32, 0.68, terrain.mul(0.8).add(detail.mul(0.2))));

  const warp = fbm(n.mul(3.0).add(pid.mul(17.0)), 3);
  const lat = n.y.add(warp.mul(0.35));
  const flow = fbm(n.mul(2.2).add(pid.mul(3.0)).add(vec3(0.0, uTime.mul(0.015), 0.0)), 3).mul(0.5).add(0.5);
  const bandPattern = sin(lat.add(flow.mul(0.25)).mul(22.0)).mul(0.5).add(0.5);
  const turb = fbm(n.mul(9.0).add(pid.mul(7.0)), 4).mul(0.5).add(0.5);
  let gas = mix(uPlanetColorB.element(index), uPlanetColorA.element(index), float(0.42).add(float(0.58).mul(bandPattern)));
  gas = gas.mul(float(0.82).add(float(0.32).mul(turb)));

  const body = mix(rocky, gas, uPlanetBands.element(index));

  const mu = max(0.0, n.z);
  const limb = pow(mu, 0.5);
  const fres = pow(oneMinus(mu), 3.0);

  let color = body.mul(float(0.06).add(day.mul(1.3).mul(float(0.55).add(float(0.45).mul(lambert))))).mul(limb).mul(disc);
  color = color.add(uPlanetColorA.element(index).mul(fres).mul(day).mul(0.55).mul(disc));
  color = color.add(uPlanetColorA.element(index).mul(glow).mul(0.85));

  const alpha = max(disc, glow.mul(0.55));
  const dead = disc.lessThanEqual(0.0).and(glow.lessThanEqual(0.002));
  const invisible = uPlanetsVisible.lessThan(0.5);
  return invisible.select(vec4(0.0), dead.select(vec4(0.0), vec4(color, alpha)));
});

// Source-over compositing (far-to-near), drawn into an inout vec4.
const composite = /*@__PURE__*/ Fn(([dst, src]) => {
  const a = clamp(src.w, 0.0, 1.0);
  const rgb = src.rgb.mul(a).add(dst.rgb.mul(oneMinus(a)));
  const aw = a.add(dst.w.mul(oneMinus(a)));
  return vec4(rgb, aw);
});

// ===========================================================================
// CelestialBodies — public API (unchanged surface).
// ===========================================================================
export class CelestialBodies {
  constructor({ scene, camera, renderer, radius = DEFAULT_RADIUS } = {}) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.radius = radius;
    this.group = new THREE.Group();
    this.group.name = 'celestial-bodies';
    this.settings = {
      visible: true,
      sunElevation: 30,
      sunAzimuth: 28,
      moonPhase: 0.62,
      moonElevation: 34,
      moonAzimuth: -62,
      moonSize: 18,
      moonGlow: 0.22,
      moonHorizonBoost: 1,
      planetsVisible: true,
      planetScale: 1,
      planetGlow: 1,
      starsVisible: true,
      starOpacity: 0.85,
      starSize: 1,
    };
  }

  init() {
    this.material = createCelestialMaterial(this.settings);
    this.dome = new THREE.Mesh(new THREE.SphereGeometry(this.radius, 128, 64), this.material);
    this.dome.name = 'shader-celestial-dome';
    this.dome.renderOrder = -6;
    this.group.add(this.dome);
    this._buildStarField();
    this.scene.add(this.group);
    return this;
  }

  _buildStarField() {
    const rng = mulberry32(1337);
    const bright = generateStarAttributes(rng, 9000, 1.0, 1.0, this.radius);
    const faint = generateStarAttributes(rng, 12000, 0.6, 0.5, this.radius);
    const count = bright.positions.length / 3 + faint.positions.length / 3;
    const positions = new Float32Array(count * 3);
    const sizes = new Float32Array(count);
    const brights = new Float32Array(count);
    const temps = new Float32Array(count);
    let o = 0;
    for (const src of [bright, faint]) {
      const n = src.positions.length / 3;
      for (let i = 0; i < n; i += 1) {
        positions[o * 3] = src.positions[i * 3];
        positions[o * 3 + 1] = src.positions[i * 3 + 1];
        positions[o * 3 + 2] = src.positions[i * 3 + 2];
        sizes[o] = src.sizes[i];
        brights[o] = src.brights[i];
        temps[o] = src.temps[i];
        o += 1;
      }
    }
    // Star field: instanced screen-space quads (sprites), one per star.
    // The original GLSL used THREE.Points with gl_PointSize; the TSL/node
    // renderer hard-codes point size to 1px on both WebGPU and the WebGL2
    // fallback, so sized points aren't possible through PointsNodeMaterial +
    // THREE.Points. Instanced sprites (InstancedMesh + PointsNodeMaterial with
    // positionNode/scaleNode) preserve the key property of the points
    // optimization — only pixels that contain a star do any work, no
    // full-screen star shader — while giving per-star pixel sizes.
    //
    // Base geometry: a unit quad (two triangles) centered at the origin in xy.
    // Per-instance attributes carry the star's sky position, pixel size,
    // brightness and temperature. instanceMatrix is set to identity (position
    // comes from positionNode, not instanceMatrix).
    const quad = new THREE.BufferGeometry();
    quad.setAttribute('position', new THREE.BufferAttribute(new Float32Array([
      -0.5, -0.5, 0.0,  0.5, -0.5, 0.0,  0.5, 0.5, 0.0,
      -0.5, -0.5, 0.0,  0.5, 0.5, 0.0,  -0.5, 0.5, 0.0,
    ]), 3));
    quad.setAttribute('uv', new THREE.BufferAttribute(new Float32Array([
      0.0, 0.0, 1.0, 0.0, 1.0, 1.0,
      0.0, 0.0, 1.0, 1.0, 0.0, 1.0,
    ]), 2));
    const iPositionAttr = new THREE.InstancedBufferAttribute(positions, 3);
    const iSizeAttr = new THREE.InstancedBufferAttribute(sizes, 1);
    const iBrightAttr = new THREE.InstancedBufferAttribute(brights, 1);
    const iTempAttr = new THREE.InstancedBufferAttribute(temps, 1);
    quad.setAttribute('iPosition', iPositionAttr);
    quad.setAttribute('iSize', iSizeAttr);
    quad.setAttribute('iBright', iBrightAttr);
    quad.setAttribute('iTemp', iTempAttr);
    quad.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), this.radius);

    this.starMaterial = createStarMaterial(this.settings, this.renderer, {
      iPosition: iPositionAttr, iSize: iSizeAttr, iBright: iBrightAttr, iTemp: iTempAttr,
    });
    this.stars = new THREE.InstancedMesh(quad, this.starMaterial, count);
    this.stars.name = 'celestial-stars';
    this.stars.frustumCulled = false;
    this.stars.renderOrder = -7;
    // Identity instance matrices: the per-instance sky position is driven by
    // positionNode (iPosition), not instanceMatrix.
    const identity = new THREE.Matrix4();
    for (let i = 0; i < count; i += 1) this.stars.setMatrixAt(i, identity);
    this.stars.instanceMatrix.needsUpdate = true;
    this.group.add(this.stars);
  }

  setVisible(visible) {
    this.settings.visible = Boolean(visible);
    this.group.visible = this.settings.visible;
  }

  setSun(options = {}) {
    Object.assign(this.settings, options);
    if (!this.material) return;
    this.material.userData.uniforms.uSunDir.value.copy(directionFromAngles(this.settings.sunElevation, this.settings.sunAzimuth));
  }

  setSkySun(dir) {
    if (this.material) this.material.userData.uniforms.uSkySunDir.value.copy(dir).normalize();
    if (this.starMaterial) this.starMaterial.userData.uniforms.uSkySunDir.value.copy(dir).normalize();
  }

  setMoonPhase(phase) {
    this.settings.moonPhase = phase;
    if (!this.material) return;
    const moonDir = directionFromAngles(this.settings.moonElevation, this.settings.moonAzimuth);
    this.material.userData.uniforms.uSunDir.value.copy(sunDirFromPhase(moonDir, phase));
  }

  setMoon(options = {}) {
    const moved = options.moonElevation !== undefined || options.moonAzimuth !== undefined;
    Object.assign(this.settings, options);
    if (!this.material) return;
    const moonDir = directionFromAngles(this.settings.moonElevation, this.settings.moonAzimuth);
    const u = this.material.userData.uniforms;
    u.uMoonDir.value.copy(moonDir);
    u.uMoonSize.value = this.settings.moonSize;
    u.uMoonGlow.value = this.settings.moonGlow;
    u.uMoonHorizonBoost.value = this.settings.moonHorizonBoost;
    if (moved) u.uSunDir.value.copy(sunDirFromPhase(moonDir, this.settings.moonPhase ?? 0));
  }

  setPlanets(options = {}) {
    Object.assign(this.settings, options);
    if (!this.material) return;
    const u = this.material.userData.uniforms;
    u.uPlanetsVisible.value = this.settings.planetsVisible ? 1 : 0;
    u.uPlanetScale.value = this.settings.planetScale;
    u.uPlanetGlow.value = this.settings.planetGlow;
  }

  setStars(options = {}) {
    Object.assign(this.settings, options);
    if (this.starMaterial) {
      const u = this.starMaterial.userData.uniforms;
      u.uStarsVisible.value = this.settings.starsVisible ? 1 : 0;
      u.uStarOpacity.value = this.settings.starOpacity;
      u.uStarSize.value = this.settings.starSize;
    }
  }

  update(time = 0) {
    if (this.camera) this.group.position.copy(this.camera.position);
    if (this.material) this.material.userData.uniforms.uTime.value = time;
    if (this.starMaterial && this.renderer) {
      const pr = this.renderer.getPixelRatio();
      if (this.starMaterial.userData.uniforms.uPixelRatio.value !== pr) {
        this.starMaterial.userData.uniforms.uPixelRatio.value = pr;
      }
    }
  }

  dispose() {
    this.scene?.remove(this.group);
    this.group.traverse((obj) => {
      obj.geometry?.dispose?.();
      obj.material?.dispose?.();
    });
  }
}

// ===========================================================================
// Material builders. Uniforms are created once and stored on
// material.userData.uniforms so the setters above can update their .value.
// ===========================================================================
function createCelestialMaterial(settings) {
  const uTime = uniform(0);
  const uSunDir = uniform(sunDirFromPhase(directionFromAngles(settings.moonElevation, settings.moonAzimuth), settings.moonPhase ?? 0.62));
  const uSkySunDir = uniform(new THREE.Vector3(0.45, 0.86, 0.24).normalize());
  const uMoonDir = uniform(directionFromAngles(settings.moonElevation, settings.moonAzimuth));
  const uMoonSize = uniform(settings.moonSize);
  const uMoonGlow = uniform(settings.moonGlow);
  const uMoonHorizonBoost = uniform(settings.moonHorizonBoost);
  const uPlanetsVisible = uniform(settings.planetsVisible ? 1 : 0);
  const uPlanetScale = uniform(settings.planetScale);
  const uPlanetGlow = uniform(settings.planetGlow);
  const uPlanetDir = uniformArray([
    directionFromAngles(18, 38),
    directionFromAngles(26, 92),
    directionFromAngles(39, 132),
    directionFromAngles(24, -128),
  ], 'vec3');
  const uPlanetSize = uniformArray([3.1, 2.15, 4.6, 3.7]);
  const uPlanetGlowBase = uniformArray([0.42, 0.34, 0.30, 0.28]);
  const uPlanetColorA = uniformArray([
    new THREE.Color('#c66a3a'),
    new THREE.Color('#e6cf94'),
    new THREE.Color('#d2b48a'),
    new THREE.Color('#e0caa0'),
  ], 'color');
  const uPlanetColorB = uniformArray([
    new THREE.Color('#5e2e18'),
    new THREE.Color('#8a7448'),
    new THREE.Color('#856a48'),
    new THREE.Color('#8a7548'),
  ], 'color');
  const uPlanetBands = uniformArray([0, 0, 1, 1]);

  const dir = positionLocal.normalize();

  // Composite far-to-near: Milky Way -> planets -> moon.
  const fragment = Fn(() => {
    let color = renderMilkyWay(dir, uSkySunDir);
    color = composite(color, renderPlanet(dir, 0, uPlanetsVisible, uPlanetScale, uPlanetGlow, uPlanetDir, uPlanetSize, uPlanetGlowBase, uPlanetColorA, uPlanetColorB, uPlanetBands, uTime));
    color = composite(color, renderPlanet(dir, 1, uPlanetsVisible, uPlanetScale, uPlanetGlow, uPlanetDir, uPlanetSize, uPlanetGlowBase, uPlanetColorA, uPlanetColorB, uPlanetBands, uTime));
    color = composite(color, renderPlanet(dir, 2, uPlanetsVisible, uPlanetScale, uPlanetGlow, uPlanetDir, uPlanetSize, uPlanetGlowBase, uPlanetColorA, uPlanetColorB, uPlanetBands, uTime));
    color = composite(color, renderPlanet(dir, 3, uPlanetsVisible, uPlanetScale, uPlanetGlow, uPlanetDir, uPlanetSize, uPlanetGlowBase, uPlanetColorA, uPlanetColorB, uPlanetBands, uTime));
    color = composite(color, renderMoon(dir, uMoonDir, uSunDir, uMoonSize, uMoonGlow, uMoonHorizonBoost));
    // discard fully transparent pixels (matches GLSL `if (color.a <= 0.001) discard;`)
    If(color.w.lessThanEqual(0.001), () => { Discard(); });
    return color;
  });

  const material = new THREE.MeshBasicNodeMaterial();
  material.transparent = true;
  material.depthWrite = false;
  material.depthTest = true;
  material.blending = THREE.AdditiveBlending;
  material.toneMapped = false;
  material.side = THREE.BackSide;
  material.fragmentNode = fragment();

  material.userData.uniforms = {
    uTime, uSunDir, uSkySunDir, uMoonDir, uMoonSize, uMoonGlow, uMoonHorizonBoost,
    uPlanetsVisible, uPlanetScale, uPlanetGlow,
    uPlanetDir, uPlanetSize, uPlanetGlowBase, uPlanetColorA, uPlanetColorB, uPlanetBands,
  };
  return material;
}

// Star field material (PointsNodeMaterial on an InstancedMesh of unit quads).
// The GLSL version wrote gl_Position/gl_PointSize directly on THREE.Points and
// culled daytime/below-horizon points. The TSL/node renderer hard-codes point
// size to 1px (both WebGPU and the WebGL2 fallback), so sized points aren't
// possible via THREE.Points. Instead each star is an instanced screen-space
// sprite: positionNode = per-instance sky position, scaleNode = per-star pixel
// size (0 when culled -> nothing renders), and the fragment does the soft disc
// + halo + blackbody color using the sprite quad UV. This keeps the points
// optimization (only star pixels do work; no full-screen star shader).
function starColor(t) {
  const c0 = vec3(1.00, 0.52, 0.32);
  const c1 = vec3(1.00, 0.84, 0.62);
  const c2 = vec3(1.00, 0.97, 0.92);
  const c3 = vec3(0.78, 0.86, 1.00);
  const lo = t.lessThan(0.34);
  const mid = t.lessThan(0.67);
  const colLow = mix(c0, c1, t.div(0.34));
  const colMid = mix(c1, c2, t.sub(0.34).div(0.33));
  const colHigh = mix(c2, c3, t.sub(0.67).div(0.33));
  return lo.select(colLow, mid.select(colMid, colHigh));
}

function createStarMaterial(settings, renderer, attrs) {
  const uStarSize = uniform(settings.starSize);
  const uStarOpacity = uniform(settings.starOpacity);
  const uStarsVisible = uniform(settings.starsVisible ? 1 : 0);
  const uBasePixels = uniform(4.4); // tuned so default starSize gives ~3px bright stars
  const uPixelRatio = uniform(renderer ? renderer.getPixelRatio() : (typeof window !== 'undefined' ? window.devicePixelRatio : 1));
  const uSkySunDir = uniform(new THREE.Vector3(0.45, 0.86, 0.24).normalize());

  // Per-instance star attributes (sky position, size, brightness, temperature).
  const iPosition = instancedBufferAttribute(attrs.iPosition, 'vec3');
  const iSize = instancedBufferAttribute(attrs.iSize, 'float');
  const iBright = instancedBufferAttribute(attrs.iBright, 'float');
  const iTemp = instancedBufferAttribute(attrs.iTemp, 'float');

  // The star's direction on the sky sphere (baked at radius; normalize -> dir).
  const skyDir = normalize(iPosition);
  // Daytime (sky sun above horizon) or hidden stars -> render nothing.
  const daytime = uStarsVisible.lessThan(0.5).or(uSkySunDir.y.greaterThan(0.0));
  const horizon = smoothstep(-0.04, 0.16, skyDir.y);
  const culled = daytime.or(horizon.lessThanEqual(0.0));
  // Per-star pixel size; 0 when culled (a zero-size sprite renders nothing).
  const size = uBasePixels.mul(uStarSize).mul(iSize).mul(uPixelRatio);

  const material = new THREE.PointsNodeMaterial();
  // positionNode = per-instance sky position; scaleNode = per-star pixel size
  // (sizeAttenuation=false keeps it in pixels, matching the original constant
  // gl_PointSize since all stars sit at the same radius from the camera).
  material.positionNode = iPosition;
  material.scaleNode = culled.select(float(0.0), size);
  material.sizeAttenuation = false;

  // Fragment: soft disc + halo + blackbody color, matching the GLSL star
  // fragment shader. Disc coords come from the sprite quad UV (0..1) -> [-1,1].
  material.fragmentNode = Fn(() => {
    const c = uv().mul(2.0).sub(1.0);
    const r2 = dot(c, c);
    If(r2.greaterThan(1.0), () => { Discard(); });
    const core = exp(r2.div(0.18).negate());
    const halo = exp(r2.div(0.9).negate()).mul(0.35);
    const intensity = core.add(halo).mul(iBright).mul(uStarOpacity).mul(horizon);
    If(intensity.lessThanEqual(0.002), () => { Discard(); });
    const col = starColor(iTemp).mul(intensity);
    return vec4(col, clamp(intensity, 0.0, 1.0));
  })();

  material.transparent = true;
  material.depthTest = false;
  material.depthWrite = false;
  material.blending = THREE.AdditiveBlending;
  material.toneMapped = false;

  material.userData.uniforms = {
    uStarSize, uStarOpacity, uStarsVisible, uBasePixels, uPixelRatio, uSkySunDir,
  };
  return material;
}
