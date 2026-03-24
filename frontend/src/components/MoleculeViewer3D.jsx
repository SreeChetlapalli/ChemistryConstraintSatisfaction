import { useRef, useMemo, useState } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import * as THREE from "three";
import { ELEMENT_MAP } from "../data/elements";

function getElem(sym) {
  const e = ELEMENT_MAP[sym];
  if (e) return { color: e.color, radius: e.radius };
  return { color: "#888888", radius: 0.77 };
}

function layout3D(atoms, adjacency) {
  const n = atoms.length;
  if (n === 0) return [];
  if (n === 1) return [new THREE.Vector3(0, 0, 0)];

  const pos = atoms.map(() =>
    new THREE.Vector3(
      (Math.random() - 0.5) * 4,
      (Math.random() - 0.5) * 4,
      (Math.random() - 0.5) * 4,
    ),
  );

  const adj = adjacency || buildAdj(atoms);
  const idealDist = 2.2;
  const repulsion = 3.0;

  for (let iter = 0; iter < 200; iter++) {
    const forces = pos.map(() => new THREE.Vector3());
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const d = pos[i].clone().sub(pos[j]);
        const dist = d.length() || 0.01;
        const rep = (repulsion / (dist * dist)) * 0.05;
        const repF = d.normalize().multiplyScalar(rep);
        forces[i].add(repF);
        forces[j].sub(repF);

        if (adj[i]?.[j] > 0) {
          const attract = (dist - idealDist) * 0.1;
          const attF = d.normalize().multiplyScalar(-attract);
          forces[i].add(attF);
          forces[j].sub(attF);
        }
      }
    }
    const damping = Math.max(0.01, 1 - iter / 200);
    for (let i = 0; i < n; i++) {
      pos[i].add(forces[i].multiplyScalar(damping));
    }
  }

  const center = new THREE.Vector3();
  pos.forEach((p) => center.add(p));
  center.divideScalar(n);
  pos.forEach((p) => p.sub(center));

  let maxR = 0;
  pos.forEach((p) => {
    maxR = Math.max(maxR, p.length());
  });
  if (maxR > 0) {
    const scale = 4 / maxR;
    pos.forEach((p) => p.multiplyScalar(scale));
  }

  return pos;
}

function buildAdj(atoms) {
  const n = atoms.length;
  const adj = Array.from({ length: n }, () => Array(n).fill(0));
  const rem = atoms.map((a) => a.bonds || 0);
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      if (rem[i] <= 0 || rem[j] <= 0) continue;
      const o = Math.min(rem[i], rem[j]);
      adj[i][j] = o;
      adj[j][i] = o;
      rem[i] -= o;
      rem[j] -= o;
    }
  }
  return adj;
}

function AtomSphere({ position, element, index, hovered, onHover }) {
  const ref = useRef();
  const { color, radius } = getElem(element);
  const scale = Math.max(0.45, radius * 1.0);
  const isHov = hovered === index;

  return (
    <group position={position}>
      <mesh
        ref={ref}
        onPointerOver={() => onHover(index)}
        onPointerOut={() => onHover(null)}
      >
        <sphereGeometry args={[scale, 32, 32]} />
        <meshStandardMaterial
          color={color}
          roughness={0.3}
          metalness={0.1}
          emissive={isHov ? color : "#000000"}
          emissiveIntensity={isHov ? 0.4 : 0}
        />
      </mesh>
      <Text
        position={[0, scale + 0.3, 0]}
        fontSize={0.28}
        color="white"
        anchorX="center"
        anchorY="bottom"
        font={undefined}
      >
        {element}
      </Text>
    </group>
  );
}

function Bond({ start, end, order }) {
  const mid = start.clone().add(end).multiplyScalar(0.5);
  const dir = end.clone().sub(start);
  const len = dir.length();
  const orientation = new THREE.Quaternion();
  orientation.setFromUnitVectors(
    new THREE.Vector3(0, 1, 0),
    dir.clone().normalize(),
  );

  const perp = new THREE.Vector3(1, 0, 0);
  if (Math.abs(dir.clone().normalize().dot(perp)) > 0.9) {
    perp.set(0, 0, 1);
  }
  const cross = dir.clone().normalize().cross(perp).normalize();

  const offsets =
    order === 1
      ? [0]
      : order === 2
        ? [-0.08, 0.08]
        : [-0.12, 0, 0.12];

  return (
    <>
      {offsets.map((off, i) => {
        const offset = cross.clone().multiplyScalar(off);
        const p = mid.clone().add(offset);
        return (
          <mesh key={i} position={p} quaternion={orientation}>
            <cylinderGeometry args={[0.06, 0.06, len, 8]} />
            <meshStandardMaterial
              color="#4a5568"
              roughness={0.5}
              metalness={0.2}
            />
          </mesh>
        );
      })}
    </>
  );
}

function MoleculeScene({ atoms, adjacency, autoRotate }) {
  const [hovered, setHovered] = useState(null);
  const groupRef = useRef();

  const adj = useMemo(
    () => adjacency || buildAdj(atoms),
    [atoms, adjacency],
  );
  const positions = useMemo(
    () => layout3D(atoms, adj),
    [atoms, adj],
  );

  useFrame((_, delta) => {
    if (autoRotate && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.3;
    }
  });

  const bonds = useMemo(() => {
    const b = [];
    for (let i = 0; i < adj.length; i++) {
      for (let j = i + 1; j < adj.length; j++) {
        if (adj[i][j] > 0 && positions[i] && positions[j]) {
          b.push({ i, j, order: Math.round(adj[i][j]) });
        }
      }
    }
    return b;
  }, [adj, positions]);

  return (
    <group ref={groupRef}>
      {bonds.map(({ i, j, order }, idx) => (
        <Bond
          key={`b-${idx}`}
          start={positions[i]}
          end={positions[j]}
          order={order}
        />
      ))}
      {atoms.map((a, i) =>
        positions[i] ? (
          <AtomSphere
            key={i}
            position={positions[i]}
            element={a.element}
            index={i}
            hovered={hovered}
            onHover={setHovered}
          />
        ) : null,
      )}
    </group>
  );
}

export default function MoleculeViewer3D({
  atoms,
  adjacency,
  height = 400,
  autoRotate = true,
}) {
  if (!atoms || atoms.length === 0) {
    return (
      <div
        style={{ height }}
        className="flex items-center justify-center text-slate-600 text-sm"
      >
        No atoms to display
      </div>
    );
  }

  return (
    <div style={{ height }} className="w-full rounded-lg overflow-hidden">
      <Canvas
        camera={{ position: [0, 0, atoms.length <= 3 ? 6 : 9], fov: 55 }}
        gl={{ antialias: true, alpha: true }}
        style={{ background: "transparent" }}
      >
        <color attach="background" args={["#171717"]} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[5, 5, 5]} intensity={0.9} />
        <directionalLight position={[-3, -3, 2]} intensity={0.4} />
        <MoleculeScene
          atoms={atoms}
          adjacency={adjacency}
          autoRotate={autoRotate}
        />
        <OrbitControls
          enablePan={false}
          minDistance={2}
          maxDistance={20}
          autoRotate={false}
        />
      </Canvas>
    </div>
  );
}
