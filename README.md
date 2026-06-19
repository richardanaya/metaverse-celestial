# metaverse-celestial

Shader-based moon, planets, and stars for Three.js metaverse skies.

```js
import { CelestialBodies } from 'metaverse-celestial';

const celestials = new CelestialBodies({ scene, camera }).init();

renderer.setAnimationLoop(() => {
  celestials.update(clock.elapsedTime);
  renderer.render(scene, camera);
});
```

The entire celestial field renders through one procedural shader on a sky dome:

- moon with procedural craters/maria/glow
- stars with shader twinkle
- visible planet bodies/glow/banding

## Examples

- `example/simple`
- `example/editor`
