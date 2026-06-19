import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CelestialBodies } from 'metaverse-celestial';

const canvas = document.querySelector('#scene');
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050912);

const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 900);
camera.position.set(0, 18, 70);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 0.8;

const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 16, 0);
controls.enableDamping = true;
controls.update();

const ground = new THREE.Mesh(
  new THREE.CircleGeometry(75, 96),
  new THREE.MeshBasicMaterial({ color: 0x080d16 }),
);
ground.rotation.x = -Math.PI / 2;
scene.add(ground);

const celestials = new CelestialBodies({ scene, camera }).init();

bindPanel();

const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  controls.update();
  celestials.update(clock.elapsedTime);
  renderer.render(scene, camera);
});

function bindPanel() {
  bindCheckbox('#visible', (value) => celestials.setVisible(value));
  bindRange('#sun-elevation', '#sun-elevation-value', (value) => {
    celestials.setSun({ sunElevation: value });
    return `${Math.round(value)}°`;
  });
  bindRange('#sun-azimuth', '#sun-azimuth-value', (value) => {
    celestials.setSun({ sunAzimuth: value });
    return `${Math.round(value)}°`;
  });
  bindRange('#moon-phase', '#moon-phase-value', (value) => {
    celestials.setMoonPhase(value);
    return phaseLabel(value);
  });
  bindRange('#moon-elevation', '#moon-elevation-value', (value) => {
    celestials.setMoon({ moonElevation: value });
    return `${Math.round(value)}°`;
  });
  bindRange('#moon-azimuth', '#moon-azimuth-value', (value) => {
    celestials.setMoon({ moonAzimuth: value });
    return `${Math.round(value)}°`;
  });
  bindRange('#moon-size', '#moon-size-value', (value) => {
    celestials.setMoon({ moonSize: value });
    return value.toFixed(1);
  });
  bindRange('#moon-glow', '#moon-glow-value', (value) => {
    celestials.setMoon({ moonGlow: value });
    return `${Math.round(value * 100)}%`;
  });
  bindRange('#moon-horizon-boost', '#moon-horizon-boost-value', (value) => {
    celestials.setMoon({ moonHorizonBoost: value });
    return `${Math.round(value * 100)}%`;
  });
  bindCheckbox('#planets-visible', (value) => celestials.setPlanets({ planetsVisible: value }));
  bindRange('#planet-scale', '#planet-scale-value', (value) => {
    celestials.setPlanets({ planetScale: value });
    return value.toFixed(2);
  });
  bindRange('#planet-glow', '#planet-glow-value', (value) => {
    celestials.setPlanets({ planetGlow: value });
    return `${Math.round(value * 100)}%`;
  });
  bindCheckbox('#stars-visible', (value) => celestials.setStars({ starsVisible: value }));
  bindRange('#star-opacity', '#star-opacity-value', (value) => {
    celestials.setStars({ starOpacity: value });
    return `${Math.round(value * 100)}%`;
  });
  bindRange('#star-size', '#star-size-value', (value) => {
    celestials.setStars({ starSize: value });
    return value.toFixed(2);
  });
  window.addEventListener('resize', resize);
  resize();
}

function bindCheckbox(selector, onChange) {
  const input = document.querySelector(selector);
  input.addEventListener('change', () => onChange(input.checked));
}

function bindRange(inputSelector, valueSelector, onInput) {
  const input = document.querySelector(inputSelector);
  const label = document.querySelector(valueSelector);
  input.addEventListener('input', () => {
    label.textContent = onInput(Number(input.value));
  });
}

function phaseLabel(phase) {
  const names = [
    'New', 'Waxing Crescent', 'First Quarter', 'Waxing Gibbous',
    'Full', 'Waning Gibbous', 'Last Quarter', 'Waning Crescent', 'New',
  ];
  const i = Math.round(phase * 8);
  return names[Math.max(0, Math.min(8, i))];
}

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}
