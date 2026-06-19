import * as THREE from 'three';

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
  const north = new THREE.Vector3().crossVectors(moonDir, east).normalize();
  // sun = moonDir*cos(elongation) + east*sin(elongation); a great-circle sweep.
  const sun = moonDir.clone().multiplyScalar(Math.cos(angle))
    .addScaledVector(east, Math.sin(angle));
  return sun.normalize();
}

function createCelestialMaterial(settings) {
  return new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    depthTest: true,
    blending: THREE.AdditiveBlending,
    toneMapped: false,
    side: THREE.BackSide,
    uniforms: {
      uTime: { value: 0 },
      uSunDir: { value: sunDirFromPhase(directionFromAngles(settings.moonElevation, settings.moonAzimuth), settings.moonPhase ?? 0.62) },
      uMoonDir: { value: directionFromAngles(settings.moonElevation, settings.moonAzimuth) },
      uMoonSize: { value: settings.moonSize },
      uMoonGlow: { value: settings.moonGlow },
      uMoonHorizonBoost: { value: settings.moonHorizonBoost },
      uPlanetsVisible: { value: settings.planetsVisible ? 1 : 0 },
      uPlanetScale: { value: settings.planetScale },
      uPlanetGlow: { value: settings.planetGlow },
      uStarsVisible: { value: settings.starsVisible ? 1 : 0 },
      uStarOpacity: { value: settings.starOpacity },
      uStarSize: { value: settings.starSize },
      uPlanetDir: { value: [
        directionFromAngles(18, 38),
        directionFromAngles(26, 92),
        directionFromAngles(39, 132),
        directionFromAngles(24, -128),
      ] },
      uPlanetSize: { value: [3.1, 2.15, 4.6, 3.7] },
      uPlanetGlowBase: { value: [0.42, 0.34, 0.30, 0.28] },
      uPlanetColorA: { value: [
        new THREE.Color('#c66a3a'),  // Mars (rocky)      - lit rust
        new THREE.Color('#e6cf94'),  // Venus (rocky)     - lit golden
        new THREE.Color('#d2b48a'),  // Jupiter (gas)     - lit tan
        new THREE.Color('#e0caa0'),  // Saturn (gas)      - lit pale yellow
      ] },
      uPlanetColorB: { value: [
        new THREE.Color('#5e2e18'),  // Mars    - shadow rust
        new THREE.Color('#8a7448'),  // Venus   - shadow golden-brown
        new THREE.Color('#856a48'),  // Jupiter - shadow brown
        new THREE.Color('#8a7548'),  // Saturn  - shadow pale brown
      ] },
      uPlanetBands: { value: [0, 0, 1, 1] },
    },
    vertexShader: `
      varying vec3 vDir;
      void main() {
        vDir = normalize(position);
        gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
      }
    `,
    fragmentShader: `
      // Let the GPU negotiate fragment precision: force highp only where the
      // device actually has fp32 fragments (desktop/iOS Apple GPUs). Older
      // Android Adreno/Mali fragment units without native highp would otherwise
      // fall into slow emulation or clamping under an unconditional highp — one
      // reason this dome was disproportionately slow on phones but fine on iOS.
      #ifdef GL_FRAGMENT_PRECISION_HIGH
      precision highp float;
      #else
      precision mediump float;
      #endif
      varying vec3 vDir;
      uniform float uTime;
      uniform vec3 uSunDir;
      uniform vec3 uMoonDir;
      uniform float uMoonSize;
      uniform float uMoonGlow;
      uniform float uMoonHorizonBoost;
      uniform float uPlanetsVisible;
      uniform float uPlanetScale;
      uniform float uPlanetGlow;
      uniform float uStarsVisible;
      uniform float uStarOpacity;
      uniform float uStarSize;
      uniform vec3 uPlanetDir[4];
      uniform float uPlanetSize[4];
      uniform float uPlanetGlowBase[4];
      uniform vec3 uPlanetColorA[4];
      uniform vec3 uPlanetColorB[4];
      uniform float uPlanetBands[4];

      const float PI = 3.141592653589793;

      // ---- hashing ----------------------------------------------------------
      float hash11(float p) {
        p = fract(p * 0.1031);
        p *= p + 33.33;
        p *= p + p;
        return fract(p);
      }

      vec3 hash33(vec3 p) {
        p = fract(p * vec3(443.8975, 441.4231, 437.5125));
        p += dot(p, p.yzx + 19.19 + 7.31);
        return fract((p.xxy + p.yxx) * p.zyx);
      }

      // ---- 3D simplex noise (Ashima / Stefan Gustavson) --------------------
      vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
      vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
      vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

      float snoise(vec3 v) {
        const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
        const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
        vec3 i = floor(v + dot(v, C.yyy));
        vec3 x0 = v - i + dot(i, C.xxx);
        vec3 g = step(x0.yzx, x0.xyz);
        vec3 l = 1.0 - g;
        vec3 i1 = min(g.xyz, l.zxy);
        vec3 i2 = max(g.xyz, l.zxy);
        vec3 x1 = x0 - i1 + C.xxx;
        vec3 x2 = x0 - i2 + C.yyy;
        vec3 x3 = x0 - D.yyy;
        i = mod289(i);
        vec4 p = permute(permute(permute(
          i.z + vec4(0.0, i1.z, i2.z, 1.0))
          + i.y + vec4(0.0, i1.y, i2.y, 1.0))
          + i.x + vec4(0.0, i1.x, i2.x, 1.0));
        float n_ = 0.142857142857;
        vec3 ns = n_ * D.wyz - D.xzx;
        vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
        vec4 x_ = floor(j * ns.z);
        vec4 y_ = floor(j - 7.0 * x_);
        vec4 x = x_ * ns.x + ns.yyyy;
        vec4 y = y_ * ns.x + ns.yyyy;
        vec4 h = 1.0 - abs(x) - abs(y);
        vec4 b0 = vec4(x.xy, y.xy);
        vec4 b1 = vec4(x.zw, y.zw);
        vec4 s0 = floor(b0) * 2.0 + 1.0;
        vec4 s1 = floor(b1) * 2.0 + 1.0;
        vec4 sh = -step(h, vec4(0.0));
        vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
        vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
        vec3 p0 = vec3(a0.xy, h.x);
        vec3 p1 = vec3(a0.zw, h.y);
        vec3 p2 = vec3(a1.xy, h.z);
        vec3 p3 = vec3(a1.zw, h.w);
        vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
        p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
        vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
        m = m * m;
        return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
      }

      // Octave-specialized fractal brownian motion (smooth, seamless on spheres).
      // The old single fbm3(p, int oct) overload had a *dynamic* loop bound —
      // 'oct' was a function argument — which ANGLE/fxc refuses to unroll when
      // transcribed to HLSL, emitting a real dynamic loop whose per-iteration
      // snoise body (~40 ALU) can't be hoisted or scheduled across iterations.
      // That's a Windows-Chrome-specific cliff: Metal and Linux-native GL inline
      // the caller so 'oct' becomes a literal and the loop unrolls. Splitting
      // into compile-time-constant bounds (3/4/5/6 octaves) lets every backend
      // unroll, and the fully-unrolled form is also friendlier to mobile GL.
      float fbm3_3(vec3 p) {
        float v = 0.0; float a = 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p);
        return v;
      }
      float fbm3_4(vec3 p) {
        float v = 0.0; float a = 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p);
        return v;
      }
      float fbm3_5(vec3 p) {
        float v = 0.0; float a = 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p);
        return v;
      }
      float fbm3_6(vec3 p) {
        float v = 0.0; float a = 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p); p = p * 2.02 + vec3(11.7, 3.1, 7.9); a *= 0.5;
        v += a * snoise(p);
        return v;
      }

      // ---- geometry helpers -------------------------------------------------
      vec2 localOnDir(vec3 dir, vec3 center) {
        vec3 up = abs(center.y) > 0.92 ? vec3(0.0, 0.0, 1.0) : vec3(0.0, 1.0, 0.0);
        vec3 right = normalize(cross(up, center));
        vec3 top = normalize(cross(center, right));
        return vec2(dot(dir, right), dot(dir, top));
      }

      float angularDist(vec3 a, vec3 b) {
        return sqrt(max(0.0, 2.0 - 2.0 * dot(a, b)));
      }

      void orthonormalBasis(vec3 n, out vec3 u, out vec3 v) {
        vec3 up = abs(n.y) < 0.94 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        u = normalize(cross(up, n));
        v = cross(n, u);
      }

      // blackbody-ish stellar color from a 0..1 temperature parameter
      vec3 starColor(float t) {
        const vec3 c0 = vec3(1.00, 0.52, 0.32); // red / orange
        const vec3 c1 = vec3(1.00, 0.84, 0.62); // orange / yellow
        const vec3 c2 = vec3(1.00, 0.97, 0.92); // yellow-white
        const vec3 c3 = vec3(0.78, 0.86, 1.00); // blue-white / hot
        if (t < 0.34) return mix(c0, c1, t / 0.34);
        if (t < 0.67) return mix(c1, c2, (t - 0.34) / 0.33);
        return mix(c2, c3, (t - 0.67) / 0.33);
      }

      // ---- Milky Way band ---------------------------------------------------
      vec3 milkyWay(vec3 dir) {
        vec3 gN = normalize(vec3(0.18, 0.42, 0.89));
        float band = dot(dir, gN);
        float w = exp(-(band * band) / 0.022);
        float n = fbm3_4(dir * 3.1 + 41.0) * 0.5 + 0.5;
        float n2 = fbm3_4(dir * 8.5 - 13.0) * 0.5 + 0.5;
        float density = w * (0.45 + 0.95 * n) * smoothstep(0.18, 0.62, n2);
        vec3 col = mix(vec3(0.45, 0.55, 0.92), vec3(0.92, 0.82, 0.72), n);
        // faint magenta tint in dense knots
        col = mix(col, vec3(0.75, 0.55, 0.85), smoothstep(0.7, 0.95, n) * 0.4);
        return col * density * 0.20;
      }

      // ---- stars (3D cellular placement, no pole pinching) ------------------
      vec4 starLayer(vec3 dir, float grid, float densityThresh, float sizeMul, float brightMul) {
        vec3 base = floor(dir * grid);
        vec3 u, v;
        orthonormalBasis(dir, u, v);
        vec4 acc = vec4(0.0);
        for (int ix = 0; ix < 3; ix++) {
          for (int iy = 0; iy < 3; iy++) {
            for (int iz = 0; iz < 3; iz++) {
              vec3 o = vec3(float(ix) - 1.0, float(iy) - 1.0, float(iz) - 1.0);
              vec3 cell = base + o;
              vec3 h = hash33(cell * 1.37 + 7.1);
              if (h.x < densityThresh) continue;
              vec3 sp = (cell + 0.5 + (h - 0.5) * 0.78) / grid;
              vec3 sd = normalize(sp);
              float cs = dot(dir, sd);
              if (cs < 0.9985) continue; // cull far cells early
              vec3 delta = sd - dir;
              float pu = dot(delta, u);
              float pv = dot(delta, v);
              // magnitude: rare bright stars, common faint ones
              float mag = pow(h.y, 4.5);
              float r = (0.00045 + mag * 0.0022) * sizeMul * uStarSize;
              float r2 = r * r;
              float core = exp(-(pu * pu + pv * pv) / r2);
              // diffraction spikes for the brightest stars
              float spike = 0.0;
              if (mag > 0.6) {
                float k = smoothstep(0.6, 0.95, mag);
                float sx = r * 7.0;
                float sy = r * 0.55;
                spike = k * (exp(-pu * pu / (sx * sx)) * exp(-pv * pv / (sy * sy))
                          + exp(-pv * pv / (sx * sx)) * exp(-pu * pu / (sy * sy)));
              }
              // soft halo
              float halo = mag * exp(-(pu * pu + pv * pv) / (r2 * 12.0)) * 0.35;
              float tw = hash11(h.z * 53.1 + 1.3);
              float twinkle = 0.72 + 0.28 * sin(uTime * (0.5 + tw * 2.2) + tw * 31.0)
                                  * sin(uTime * (1.7 + tw * 1.3) + tw * 17.0);
              float intensity = (core + spike * 0.55 + halo) * (0.4 + mag * 0.9) * brightMul * twinkle;
              vec3 col = starColor(h.z);
              acc.rgb += col * intensity;
              acc.a = max(acc.a, clamp(intensity, 0.0, 1.0));
            }
          }
        }
        return acc;
      }

      vec4 renderStars(vec3 dir) {
        if (uStarsVisible < 0.5) return vec4(0.0);
        float horizon = smoothstep(-0.04, 0.16, dir.y);
        if (horizon <= 0.0) return vec4(0.0);
        vec3 mw = milkyWay(dir) * horizon;
        // bright / mid tier
        vec4 a = starLayer(dir, 165.0, 0.977, 1.0, 1.0);
        // faint dust tier
        vec4 b = starLayer(dir, 330.0, 0.992, 0.6, 0.5);
        vec3 rgb = (a.rgb + b.rgb) * uStarOpacity * horizon + mw;
        float alpha = max(a.a, b.a) * uStarOpacity * horizon;
        alpha = max(alpha, clamp(length(mw) * 2.2, 0.0, 1.0) * uStarOpacity * horizon);
        return vec4(rgb, alpha);
      }

      // ---- moon -------------------------------------------------------------
      // smooth, large-scale terrain height (3D noise on sphere normal -> seamless).
      // Kept low-frequency so the surface reads as soft, hazy plains rather than
      // sharp, cratered detail — the moon intentionally looks slightly out of focus.
      float moonHeight(vec3 n) {
        return fbm3_5(n * 2.2 + 31.0) * 0.9 + fbm3_3(n * 5.0 + 7.0) * 0.1;
      }

      // a smooth "crater field" mask: 1 in cratered highlands, ~0 in smooth maria.
      // craters are concentrated where the terrain is high and rough, mirroring the
      // real Moon (maria are young, smooth basalt plains; highlands are old & saturated).
      float craterMask(vec3 n, float terra) {
        float highland = smoothstep(0.0, 0.4, terra);          // only in elevated terrain
        float rough = smoothstep(0.2, 0.8, fbm3_3(n * 7.0 + 53.0) * 0.5 + 0.5);
        return highland * rough;
      }

      // cellular craters, density gated by the highland mask (gate).
      float moonCraters(vec3 n, float gate) {
        if (gate < 0.02) return 0.0;
        float c = 0.0;
        for (int s = 0; s < 2; s++) {
          float scale = 22.0 * pow(1.9, float(s));
          vec3 cb = floor(n * scale);
          for (int ix = 0; ix < 3; ix++) {
            for (int iy = 0; iy < 3; iy++) {
              for (int iz = 0; iz < 3; iz++) {
                vec3 o = vec3(float(ix) - 1.0, float(iy) - 1.0, float(iz) - 1.0);
                vec3 cell = cb + o;
                vec3 h = hash33(cell + float(s) * 23.7 + 4.1);
                // rare host cells (most of the surface stays smooth)
                float prob = mix(0.07, 0.035, float(s));
                if (h.x > prob) continue;
                vec3 cp = normalize((cell + 0.5 + (h - 0.5) * 0.7) / scale);
                float dd = angularDist(n, cp);
                // a few large basins, mostly small pits; rims kept soft/low for a hazy look
                float sz = pow(h.y, 3.0);
                float r = (0.12 + sz * 0.55) / scale;
                if (dd > r * 1.15) continue;
                // soften craters near maria so they fade, not pop
                float g = gate;
                float rim = smoothstep(r, r * 0.78, dd) * smoothstep(r * 0.4, r * 0.92, dd);
                float pit = smoothstep(r * 0.7, 0.0, dd);
                c += (rim * 0.14 - pit * 0.10) * g;
              }
            }
          }
        }
        return c;
      }

      vec4 renderMoon(vec3 dir) {
        vec3 mDir = normalize(uMoonDir);
        vec3 sunDir = normalize(uSunDir);
        // Moon illusion emulation: the horizon Moon is perceived ~50% larger.
        // This is a perceptual effect (not physical), exposed as a tunable strength.
        float elev = clamp(mDir.y, 0.0, 1.0);
        float horizonFactor = pow(1.0 - elev, 1.3);            // 0 at zenith, 1 at horizon
        float sizeScale = 1.0 + uMoonHorizonBoost * 0.5 * horizonFactor;
        float angularSize = uMoonSize * 0.00125 * sizeScale;
        float dist = angularDist(dir, mDir);
        // softer disc edge: wider anti-aliased transition for a slightly hazy limb
        float disc = smoothstep(angularSize, angularSize * 0.93, dist);
        float glowR = angularSize * 2.4;
        float glow = exp(-(dist * dist) / (glowR * glowR)) * uMoonGlow;
        if (disc <= 0.0 && glow <= 0.002) return vec4(0.0);

        vec2 p = localOnDir(dir, mDir) / angularSize;
        float d = length(p);
        vec3 n = normalize(vec3(p, sqrt(max(0.0, 1.0 - d * d))));

        // terrain + maria (3D noise on the sphere normal: no seam)
        float terra = moonHeight(n);
        // large, smooth maria basins — the defining dark features of the near side
        float maria = smoothstep(0.42, 0.66, fbm3_4(n * 1.8 + 91.0) * 0.5 + 0.5);
        float gate = craterMask(n, terra) * (1.0 - maria * 0.9);
        float craters = moonCraters(n, gate);

        // gentle relief shading from the large-scale height gradient only; the
        // larger sample radius (e) blurs the perturbed normal for a soft-focus look
        vec3 tx, ty;
        orthonormalBasis(n, tx, ty);
        float e = 0.06;
        float hx = moonHeight(n + tx * e);
        float hy = moonHeight(n + ty * e);
        vec3 rn = normalize(n - tx * ((hx - terra) / e) * 0.03 - ty * ((hy - terra) / e) * 0.03);

        // Lighting from the actual sun direction -> the lunar phase (crescent,
        // quarter, gibbous, full, new) emerges from the sun/moon geometry on the sky.
        // Note the sign: a full moon has the sun opposite the moon on the sky
        // (sunDir ~= -mDir), so the lit normal is dot(rn, -sunDir).
        float lit = dot(rn, -sunDir);
        float day = smoothstep(-0.06, 0.16, lit);           // soft terminator
        float lambert = max(0.0, lit);

        // physical-ish limb darkening (mu = cos of angle from disc center)
        float mu = max(0.0, n.z);
        float limb = pow(mu, 0.55);

        // highlands are brighter & warmer; maria are dark, cool basalt plains
        // Real lunar regolith is a dark brownish gray (albedo ~0.12). Highlands are
        // brighter/warmer; maria are darker, cooler basalt plains (~0.07 albedo).
        vec3 highland = mix(vec3(0.80, 0.77, 0.72), vec3(0.70, 0.68, 0.64), smoothstep(0.2, 0.7, terra * 0.5 + 0.5));
        vec3 mariaC = vec3(0.26, 0.26, 0.29);
        vec3 base = mix(highland, mariaC, maria);
        base += craters;                 // craters supply the localized light/dark relief
        base *= (0.92 + 0.08 * smoothstep(-0.2, 0.2, terra)); // very subtle albedo variation

        // Lighting term + a faint earthshine on the shadowed side. Earthshine is
        // real (sunlight reflected off Earth onto the dark lunar face) and keeps a
        // crescent's dark side just barely visible instead of pure black. It is
        // kept subtle and independent of uMoonGlow so high glow never turns the
        // dark side into a glowing disc.
        vec3 color = base * (0.05 + day * 1.9 * (0.5 + 0.5 * lambert)) * limb * disc;
        float earthshine = (1.0 - day) * disc;
        color += vec3(0.05, 0.06, 0.085) * earthshine;

        // Atmospheric extinction near the horizon: longer air path scatters out
        // blue light, leaving the low Moon yellow -> orange -> red (Rayleigh).
        float airMass = 1.0 / max(elev, 0.04);              // rough relative air mass
        float ext = clamp(0.18 * log(airMass), 0.0, 0.85);
        color.r *= 1.0 + ext * 0.5;
        color.g *= 1.0 - ext * 0.25;
        color.b *= 1.0 - ext * 0.8;

        // Moon glow: a soft halo that lives only OUTSIDE the disc (a true halo),
        // so it never washes over the shadowed side of a crescent.
        vec3 glowColor = mix(vec3(0.62, 0.71, 0.9), vec3(0.85, 0.55, 0.35), ext);
        float halo = glow * (1.0 - disc);
        color += glowColor * halo;
        return vec4(color, max(disc, halo * 0.45));
      }

      // ---- planets ----------------------------------------------------------
      vec4 renderPlanet(vec3 dir, int index) {
        if (uPlanetsVisible < 0.5) return vec4(0.0);
        vec3 pDir = normalize(uPlanetDir[index]);
        float angularSize = uPlanetSize[index] * uPlanetScale * 0.00068;
        float dist = angularDist(dir, pDir);
        float disc = smoothstep(angularSize, angularSize * 0.93, dist);
        float glowR = angularSize * 3.4;
        float glow = exp(-(dist * dist) / (glowR * glowR)) * uPlanetGlowBase[index] * uPlanetGlow;
        if (disc <= 0.0 && glow <= 0.002) return vec4(0.0);

        vec2 lp = localOnDir(dir, pDir) / angularSize;
        float dd = dot(lp, lp);
        vec3 n = normalize(vec3(lp, sqrt(max(0.0, 1.0 - dd))));
        float pid = float(index);

        // lighting with soft terminator
        vec3 lightDir = normalize(vec3(-0.4, 0.32, 0.86));
        float lit = dot(n, lightDir);
        float day = smoothstep(-0.08, 0.18, lit);
        float lambert = max(0.0, lit);

        // surface features (3D noise -> no seam)
        float terrain = fbm3_4(n * 4.0 + pid * 11.0) * 0.5 + 0.5;
        float detail = fbm3_3(n * 16.0 + pid * 5.0) * 0.5 + 0.5;
        vec3 rocky = mix(uPlanetColorB[index], uPlanetColorA[index], smoothstep(0.32, 0.68, terrain * 0.8 + detail * 0.2));

        // domain-warped gas-giant banding
        float warp = fbm3_3(n * 3.0 + pid * 17.0);
        float lat = n.y + warp * 0.35;
        float flow = fbm3_3(n * 2.2 + pid * 3.0 + vec3(0.0, uTime * 0.015, 0.0)) * 0.5 + 0.5;
        float bandPattern = sin((lat + flow * 0.25) * 22.0) * 0.5 + 0.5;
        float turb = fbm3_4(n * 9.0 + pid * 7.0) * 0.5 + 0.5;
        vec3 gas = mix(uPlanetColorB[index], uPlanetColorA[index], 0.42 + 0.58 * bandPattern);
        gas *= (0.82 + 0.32 * turb);

        vec3 body = mix(rocky, gas, uPlanetBands[index]);

        // limb darkening + fresnel atmospheric rim
        float mu = max(0.0, n.z);
        float limb = pow(mu, 0.5);
        float fres = pow(1.0 - mu, 3.0);

        vec3 color = body * (0.06 + day * 1.3 * (0.55 + 0.45 * lambert)) * limb * disc;
        // atmospheric rim scattering on the lit limb
        color += uPlanetColorA[index] * fres * day * 0.55 * disc;
        // outer atmosphere glow
        color += uPlanetColorA[index] * glow * 0.85;
        return vec4(color, max(disc, glow * 0.55));
      }

      // Source-over compositing, drawn far-to-near: stars first, then planets,
      // then the moon last. Because each layer covers what's behind it, the moon
      // always occludes planets and stars, and planets always occlude stars.
      // (The final dome still additively blends onto the scene via AdditiveBlending,
      // using this accumulated coverage as its alpha.)
      void composite(inout vec4 dst, vec4 src) {
        float a = clamp(src.a, 0.0, 1.0);
        dst.rgb = src.rgb * a + dst.rgb * (1.0 - a);
        dst.a = a + dst.a * (1.0 - a);
      }

      void main() {
        vec3 dir = normalize(vDir);
        vec4 color = renderStars(dir);
        composite(color, renderPlanet(dir, 0));
        composite(color, renderPlanet(dir, 1));
        composite(color, renderPlanet(dir, 2));
        composite(color, renderPlanet(dir, 3));
        composite(color, renderMoon(dir));
        if (color.a <= 0.001) discard;
        gl_FragColor = color;
      }
    `,
  });
}

export class CelestialBodies {
  constructor({ scene, camera, radius = DEFAULT_RADIUS } = {}) {
    this.scene = scene;
    this.camera = camera;
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
    this.scene.add(this.group);
    return this;
  }

  setVisible(visible) {
    this.settings.visible = Boolean(visible);
    this.group.visible = this.settings.visible;
  }

  setSun(options = {}) {
    Object.assign(this.settings, options);
    if (!this.material) return;
    this.material.uniforms.uSunDir.value.copy(directionFromAngles(this.settings.sunElevation, this.settings.sunAzimuth));
  }

  // Drive the lunar phase directly with a 0..1 value:
  // 0 = new, 0.25 = first quarter, 0.5 = full, 0.75 = last quarter, 1 = new.
  // This positions the sun on a great circle around the moon so the terminator
  // geometry stays physically consistent. Moving the moon afterwards keeps the phase.
  setMoonPhase(phase) {
    this.settings.moonPhase = phase;
    if (!this.material) return;
    const moonDir = directionFromAngles(this.settings.moonElevation, this.settings.moonAzimuth);
    this.material.uniforms.uSunDir.value.copy(sunDirFromPhase(moonDir, phase));
  }

  setMoon(options = {}) {
    const moved = options.moonElevation !== undefined || options.moonAzimuth !== undefined;
    Object.assign(this.settings, options);
    if (!this.material) return;
    const moonDir = directionFromAngles(this.settings.moonElevation, this.settings.moonAzimuth);
    this.material.uniforms.uMoonDir.value.copy(moonDir);
    this.material.uniforms.uMoonSize.value = this.settings.moonSize;
    this.material.uniforms.uMoonGlow.value = this.settings.moonGlow;
    this.material.uniforms.uMoonHorizonBoost.value = this.settings.moonHorizonBoost;
    // keep the sun consistent with the current phase if the moon moved on the sky
    if (moved) this.material.uniforms.uSunDir.value.copy(sunDirFromPhase(moonDir, this.settings.moonPhase ?? 0));
  }

  setPlanets(options = {}) {
    Object.assign(this.settings, options);
    if (!this.material) return;
    this.material.uniforms.uPlanetsVisible.value = this.settings.planetsVisible ? 1 : 0;
    this.material.uniforms.uPlanetScale.value = this.settings.planetScale;
    this.material.uniforms.uPlanetGlow.value = this.settings.planetGlow;
  }

  setStars(options = {}) {
    Object.assign(this.settings, options);
    if (!this.material) return;
    this.material.uniforms.uStarsVisible.value = this.settings.starsVisible ? 1 : 0;
    this.material.uniforms.uStarOpacity.value = this.settings.starOpacity;
    this.material.uniforms.uStarSize.value = this.settings.starSize;
  }

  update(time = 0) {
    if (this.camera) this.group.position.copy(this.camera.position);
    if (this.material) this.material.uniforms.uTime.value = time;
  }

  dispose() {
    this.scene?.remove(this.group);
    this.group.traverse((obj) => {
      obj.geometry?.dispose?.();
      obj.material?.dispose?.();
    });
  }
}
