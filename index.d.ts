import type * as THREE from 'three';

export interface CelestialSettings {
  visible: boolean;
  sunElevation: number;
  sunAzimuth: number;
  moonPhase: number;
  moonElevation: number;
  moonAzimuth: number;
  moonSize: number;
  moonGlow: number;
  moonHorizonBoost: number;
  planetsVisible: boolean;
  planetScale: number;
  planetGlow: number;
  starsVisible: boolean;
  starOpacity: number;
  starSize: number;
}

export interface CelestialBodiesOptions {
  scene?: THREE.Scene;
  camera?: THREE.Camera;
  radius?: number;
}

export class CelestialBodies {
  scene?: THREE.Scene;
  camera?: THREE.Camera;
  radius: number;
  group: THREE.Group;
  dome?: THREE.Mesh;
  material?: THREE.ShaderMaterial;
  settings: CelestialSettings;

  constructor(options?: CelestialBodiesOptions);
  init(): this;
  setVisible(visible: boolean): void;
  setSun(options?: Partial<Pick<CelestialSettings, 'sunElevation' | 'sunAzimuth'>>): void;
  /** Feed the SKY's real sun direction so stars/Milky Way are skipped at daytime. Distinct from setSun (moon-phase sun). */
  setSkySun(dir: THREE.Vector3 | { x: number; y: number; z: number }): void;
  setMoonPhase(phase: number): void;
  setMoon(options?: Partial<Pick<CelestialSettings, 'moonElevation' | 'moonAzimuth' | 'moonSize' | 'moonGlow' | 'moonHorizonBoost'>>): void;
  setPlanets(options?: Partial<Pick<CelestialSettings, 'planetsVisible' | 'planetScale' | 'planetGlow'>>): void;
  setStars(options?: Partial<Pick<CelestialSettings, 'starsVisible' | 'starOpacity' | 'starSize'>>): void;
  update(time?: number): void;
  dispose(): void;
}
