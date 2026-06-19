import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CelestialBodies } from 'metaverse-celestial';

const canvas = document.querySelector('canvas');
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

const celestials = new CelestialBodies({ scene, camera, renderer }).init();
// Tell the celestial bodies it's nighttime: a sky-sun below the horizon opens
// the night-gate so the star field + Milky Way render. (In a real app the host
// derives this from its Preetham sky and calls setSkySun each frame.)
celestials.setSkySun(new THREE.Vector3(0.3, -0.25, 0.4));
const clock = new THREE.Clock();

function resize() {
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height, false);
}

function animate() {
  controls.update();
  celestials.update(clock.elapsedTime);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

window.addEventListener('resize', resize);
resize();
animate();
