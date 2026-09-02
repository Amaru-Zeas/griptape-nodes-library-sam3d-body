var THREE_CDN = "https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js";

function loadThree() {
  return new Promise(function (res, rej) {
    if (window.THREE) return res();
    if (window.__gtnThreeLoading) {
      window.__gtnThreeLoading.then(res, rej);
      return;
    }
    window.__gtnThreeLoading = new Promise(function (res2, rej2) {
      var s = document.createElement("script");
      s.src = THREE_CDN;
      s.onload = res2;
      s.onerror = rej2;
      document.head.appendChild(s);
    });
    window.__gtnThreeLoading.then(res, rej);
  });
}

function parseValue(raw) {
  var v = raw && typeof raw === "object" ? raw : {};
  var ft = v.faceTrack && typeof v.faceTrack === "object" ? v.faceTrack : null;
  return {
    jointsUrl: String(v.jointsUrl || ""),
    meshUrl: String(v.meshUrl || ""),
    fps: Number(v.fps) > 0 ? Number(v.fps) : 24,
    faceCam: !!v.faceCam,
    faceTrack: ft && Array.isArray(ft.headIdx) && ft.headIdx.length && Number(ft.noseIdx) >= 0
      ? { headIdx: ft.headIdx, noseIdx: Number(ft.noseIdx) }
      : null
  };
}

export default function View3DBodyWidget(container, props) {
  var state = parseValue(props && props.value);

  // The widget is width-driven (16:9 overall) so it never overflows the
  // host's allotted area vertically by surprise. The controls live in their
  // OWN strip below the viewport - nothing overlaps the 3D view.
  container.innerHTML =
    '<div class="v3d nodrag nowheel" style="display:flex;flex-direction:column;width:100%;aspect-ratio:16/9;' +
    "background:#232323;border:1px solid #1a1a1a;border-radius:8px;font-family:monospace;" +
    'font-size:11px;color:#d8d8d8;box-sizing:border-box;padding:6px;">' +
    '<div class="v3d-wrap" style="position:relative;flex:1;min-height:0;width:100%;background:#3b3b3b;border:1px solid #1a1a1a;border-radius:6px;overflow:hidden;">' +
    '<div class="v3d-msg" style="color:#9a9a9a;padding:60px 0;text-align:center;">Loading three.js...</div>' +
    "</div>" +
    '<div class="v3d-bar" style="display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-top:6px;' +
    'background:#1e1e1e;border:1px solid #1a1a1a;border-radius:4px;padding:4px 6px;">' +
    '<button class="v3d-play" style="width:52px;background:#3d3d3d;color:#e0e0e0;border:1px solid #4a4a4a;border-radius:4px;padding:3px 0;cursor:pointer;">Play</button>' +
    '<input class="v3d-frame" type="range" min="0" max="0" step="1" value="0" style="flex:1;accent-color:#4772b3;min-width:40px;" />' +
    '<span class="v3d-label" style="text-align:right;color:#b8b8b8;white-space:nowrap;">0 / 0</span>' +
    '<span class="v3d-status" style="color:#d0905a;"></span>' +
    '<select class="v3d-speed" style="background:#3d3d3d;color:#e0e0e0;border:1px solid #4a4a4a;border-radius:4px;padding:2px;cursor:pointer;">' +
    '<option value="0.25">0.25x</option><option value="0.5">0.5x</option><option value="1" selected>1x</option>' +
    '<option value="1.5">1.5x</option><option value="2">2x</option><option value="4">4x</option></select>' +
    '<button class="v3d-face" title="Face mocap camera: follow the head" style="background:#3d3d3d;color:#e0e0e0;border:1px solid #4a4a4a;border-radius:4px;padding:3px 6px;cursor:pointer;white-space:nowrap;">Face</button>' +
    '<button class="v3d-full" title="Fullscreen" style="background:#3d3d3d;color:#e0e0e0;border:1px solid #4a4a4a;border-radius:4px;padding:3px 6px;cursor:pointer;">\u26f6</button>' +
    '<span class="v3d-ver" style="color:#7a7a7a;white-space:nowrap;">v7</span>' +
    '<label style="display:flex;gap:3px;align-items:center;color:#b0b0b0;cursor:pointer;">' +
    '<input class="v3d-mesh" type="checkbox" checked />mesh</label>' +
    '<label class="v3d-body-label" style="display:flex;gap:3px;align-items:center;color:#b0b0b0;cursor:pointer;">' +
    '<input class="v3d-body" type="checkbox" checked />body</label>' +
    '<label style="display:flex;gap:3px;align-items:center;color:#b0b0b0;cursor:pointer;">' +
    '<input class="v3d-joints" type="checkbox" />joints</label>' +
    "</div></div>";

  var root = container.querySelector(".v3d");
  var wrap = container.querySelector(".v3d-wrap");
  var msg = container.querySelector(".v3d-msg");
  var playBtn = container.querySelector(".v3d-play");
  var frameSlider = container.querySelector(".v3d-frame");
  var frameLabel = container.querySelector(".v3d-label");
  var meshToggle = container.querySelector(".v3d-mesh");
  var bodyToggle = container.querySelector(".v3d-body");
  var bodyLabel = container.querySelector(".v3d-body-label");
  var jointsToggle = container.querySelector(".v3d-joints");
  var statusEl = container.querySelector(".v3d-status");
  var verEl = container.querySelector(".v3d-ver");
  var speedSel = container.querySelector(".v3d-speed");
  var faceBtn = container.querySelector(".v3d-face");
  var fullBtn = container.querySelector(".v3d-full");

  ["pointerdown", "keydown", "keyup", "wheel"].forEach(function (evt) {
    [playBtn, frameSlider, meshToggle, bodyToggle, jointsToggle, speedSel, faceBtn, fullBtn].forEach(function (el) {
      el.addEventListener(evt, function (e) {
        e.stopPropagation();
      });
    });
  });

  var destroyed = false;
  var ctx = null;
  var anim = { frame: 0, playing: false, lastTick: 0, nFrames: 0 };
  var jointData = { frames: [] };
  var meshData = null; // {nFrames, nVerts, verts, geometry, mesh}
  var loadToken = 0;
  var loadedJointsUrl = "";
  var loadedMeshUrl = "";
  var resizeObserver = null;
  var perf = { t0: 0, count: 0 };
  // Face mocap cam: rigidly attached to the head like a helmet rig. sc/sf/su
  // are the smoothed (locked) center / forward / up anchors so per-frame
  // estimation noise never shakes the camera; crownIdx is a top-of-skull
  // vertex used to make the camera roll with the head.
  var face = { on: false, dist: 0.55, sc: null, sf: null, su: null, crownIdx: -1, lastFrame: -99 };

  function setMsg(text) {
    if (msg) msg.textContent = text;
  }

  function setFaceButton() {
    faceBtn.style.background = face.on ? "#4772b3" : "#3d3d3d";
    faceBtn.style.color = face.on ? "#ffffff" : "#e0e0e0";
  }

  function buildScene() {
    var R = window.THREE;
    var renderer = new R.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio || 1);
    renderer.setSize(640, 360, false);
    renderer.setClearColor(0x3b3b3b); // Blender viewport gray
    renderer.outputEncoding = R.sRGBEncoding;
    var scene = new R.Scene();
    var camera = new R.PerspectiveCamera(40, 16 / 9, 0.01, 1000);

    scene.add(new R.AmbientLight(0xffffff, 0.5));
    var key = new R.DirectionalLight(0xffffff, 0.9);
    key.position.set(2, 4, 3);
    scene.add(key);
    var rim = new R.DirectionalLight(0xdddddd, 0.35);
    rim.position.set(-3, 2, -2);
    scene.add(rim);

    var grid = new R.GridHelper(4, 16, 0x565656, 0x484848);
    scene.add(grid);

    var orbit = { theta: Math.PI, phi: 1.25, dist: 3.2, cx: 0, cy: 0.9, cz: 0 };
    function applyCamera() {
      if (face.on) return; // face cam owns the camera per-frame
      var sp = Math.sin(orbit.phi);
      camera.position.set(
        orbit.cx + orbit.dist * sp * Math.sin(orbit.theta),
        orbit.cy + orbit.dist * Math.cos(orbit.phi),
        orbit.cz + orbit.dist * sp * Math.cos(orbit.theta)
      );
      camera.lookAt(orbit.cx, orbit.cy, orbit.cz);
    }
    applyCamera();

    var dom = renderer.domElement;
    var drag = null;
    dom.addEventListener("pointerdown", function (e) {
      e.stopPropagation();
      e.preventDefault();
      if (face.on) return; // face cam: camera is locked to the head, no user control
      // Middle or right button (or shift+left) pans, plain left rotates.
      var pan = e.button === 1 || e.button === 2 || e.shiftKey;
      drag = { x: e.clientX, y: e.clientY, pan: pan };
      dom.setPointerCapture(e.pointerId);
    });
    dom.addEventListener("contextmenu", function (e) {
      e.preventDefault();
      e.stopPropagation();
    });
    dom.addEventListener("pointermove", function (e) {
      if (!drag || face.on) return;
      var dx = e.clientX - drag.x;
      var dy = e.clientY - drag.y;
      if (drag.pan) {
        // Slide the orbit target along the camera's right/up axes so the
        // scene follows the cursor (grab-style pan).
        var scale = orbit.dist * 0.0016;
        var sinT = Math.sin(orbit.theta);
        var cosT = Math.cos(orbit.theta);
        var sinP = Math.sin(orbit.phi);
        var cosP = Math.cos(orbit.phi);
        // right = (cosT, 0, -sinT); up = (-cosP*sinT, sinP, -cosP*cosT)
        orbit.cx += (-cosT * dx - cosP * sinT * dy) * scale;
        orbit.cy += sinP * dy * scale;
        orbit.cz += (sinT * dx - cosP * cosT * dy) * scale;
      } else {
        orbit.theta -= dx * 0.008;
        orbit.phi = Math.max(0.15, Math.min(Math.PI - 0.15, orbit.phi - dy * 0.008));
      }
      drag = { x: e.clientX, y: e.clientY, pan: drag.pan };
      applyCamera();
    });
    dom.addEventListener("pointerup", function () {
      drag = null;
    });
    dom.addEventListener(
      "wheel",
      function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (face.on) return; // face cam: locked, no zoom
        orbit.dist = Math.max(0.4, Math.min(30, orbit.dist * (e.deltaY > 0 ? 1.1 : 0.9)));
        applyCamera();
      },
      { passive: false }
    );

    return { renderer: renderer, scene: scene, camera: camera, orbit: orbit, applyCamera: applyCamera, jointGroup: null };
  }

  function updateFaceCam(idx) {
    if (!ctx || !face.on) return;
    var ft = state.faceTrack;
    if (!ft || !meshData) return;
    var f = Math.max(0, Math.min(meshData.nFrames - 1, idx | 0));
    var base = f * meshData.nVerts * 3;
    var verts = meshData.verts;
    var n = ft.headIdx.length;
    var cx = 0, cy = 0, cz = 0;
    for (var i = 0; i < n; i++) {
      var o = base + ft.headIdx[i] * 3;
      cx += verts[o];
      cy += verts[o + 1];
      cz += verts[o + 2];
    }
    cx /= n; cy /= n; cz /= n;
    // Rigid head frame, like a helmet-mounted mocap cam:
    // forward = center->nose (full 3D, so the camera pitches with the head),
    // up = center->crown (so the camera rolls with the head).
    var no = base + ft.noseIdx * 3;
    var fx = verts[no] - cx, fy = verts[no + 1] - cy, fz = verts[no + 2] - cz;
    var len = Math.sqrt(fx * fx + fy * fy + fz * fz) || 1;
    fx /= len; fy /= len; fz /= len;

    if (face.crownIdx < 0) {
      // One-time pick: the head vertex highest above the head center. A fixed
      // anatomical point (top of skull) that then tracks roll on every frame.
      var bestIdx = ft.headIdx[0], bestY = -Infinity;
      for (var c = 0; c < n; c++) {
        var yv = verts[base + ft.headIdx[c] * 3 + 1];
        if (yv > bestY) { bestY = yv; bestIdx = ft.headIdx[c]; }
      }
      face.crownIdx = bestIdx;
    }
    var co = base + face.crownIdx * 3;
    var ux = verts[co] - cx, uy = verts[co + 1] - cy, uz = verts[co + 2] - cz;
    // Orthogonalize up against forward so the camera basis stays rigid.
    var dp = ux * fx + uy * fy + uz * fz;
    ux -= dp * fx; uy -= dp * fy; uz -= dp * fz;
    var ul = Math.sqrt(ux * ux + uy * uy + uz * uz);
    if (ul < 1e-6) { ux = 0; uy = 1; uz = 0; } else { ux /= ul; uy /= ul; uz /= ul; }

    // Locked framing: exponentially smooth the anchors so per-frame estimation
    // noise never shakes the camera. A scrub (big frame jump) resets the
    // smoother so the camera snaps straight to the new pose.
    var jump = Math.abs(f - face.lastFrame) > 3;
    face.lastFrame = f;
    if (!face.sc || jump) {
      face.sc = [cx, cy, cz];
      face.sf = [fx, fy, fz];
      face.su = [ux, uy, uz];
    } else {
      var a = 0.5; // tight follow: damps jitter without lagging head turns
      face.sc[0] += a * (cx - face.sc[0]);
      face.sc[1] += a * (cy - face.sc[1]);
      face.sc[2] += a * (cz - face.sc[2]);
      face.sf[0] += a * (fx - face.sf[0]);
      face.sf[1] += a * (fy - face.sf[1]);
      face.sf[2] += a * (fz - face.sf[2]);
      var sl = Math.sqrt(face.sf[0] * face.sf[0] + face.sf[1] * face.sf[1] + face.sf[2] * face.sf[2]) || 1;
      face.sf[0] /= sl; face.sf[1] /= sl; face.sf[2] /= sl;
      face.su[0] += a * (ux - face.su[0]);
      face.su[1] += a * (uy - face.su[1]);
      face.su[2] += a * (uz - face.su[2]);
      // Re-orthogonalize the smoothed up against the smoothed forward.
      var du = face.su[0] * face.sf[0] + face.su[1] * face.sf[1] + face.su[2] * face.sf[2];
      face.su[0] -= du * face.sf[0]; face.su[1] -= du * face.sf[1]; face.su[2] -= du * face.sf[2];
      var usl = Math.sqrt(face.su[0] * face.su[0] + face.su[1] * face.su[1] + face.su[2] * face.su[2]) || 1;
      face.su[0] /= usl; face.su[1] /= usl; face.su[2] /= usl;
    }
    var scx = face.sc[0], scy = face.sc[1], scz = face.sc[2];
    // Camera sits straight out along the face normal and inherits head roll.
    ctx.camera.up.set(face.su[0], face.su[1], face.su[2]);
    ctx.camera.position.set(
      scx + face.dist * face.sf[0],
      scy + face.dist * face.sf[1],
      scz + face.dist * face.sf[2]
    );
    ctx.camera.lookAt(scx, scy, scz);
  }

  function applyBodyVisibility() {
    if (!meshData || !meshData.mesh) return;
    var headOnly = !bodyToggle.checked && meshData.headIndex;
    meshData.geometry.setIndex(headOnly ? meshData.headIndex : meshData.fullIndex);
    meshData.geometry.index.needsUpdate = true;
  }

  function setFaceMode(on) {
    if (on && !(state.faceTrack && state.meshUrl)) {
      statusEl.textContent = "no face-track data - re-run the pipeline";
      return;
    }
    face.on = on;
    face.sc = null; // re-lock the smoothed anchors from the current pose
    face.su = null;
    setFaceButton();
    if (ctx) {
      ctx.camera.fov = on ? 28 : 40; // longer lens for the face close-up
      ctx.camera.updateProjectionMatrix();
      if (on) {
        updateFaceCam(anim.frame);
      } else {
        ctx.camera.up.set(0, 1, 0); // undo any head roll before orbiting again
        ctx.applyCamera();
      }
    }
  }

  function setFrame(idx) {
    if (!ctx || !anim.nFrames) return;
    idx = Math.max(0, Math.min(anim.nFrames - 1, idx | 0));
    anim.frame = idx;

    if (meshData && meshData.mesh) {
      var pos = meshData.geometry.attributes.position;
      var f = Math.min(idx, meshData.nFrames - 1);
      pos.array.set(meshData.verts.subarray(f * meshData.nVerts * 3, (f + 1) * meshData.nVerts * 3));
      pos.needsUpdate = true;
      meshData.geometry.computeVertexNormals();
    }

    if (ctx.jointGroup) {
      var people = (jointData.frames[Math.min(idx, jointData.frames.length - 1)] || []);
      var spheres = ctx.jointGroup.children;
      var s = 0;
      for (var p = 0; p < people.length; p++) {
        var flat = people[p];
        for (var j = 0; j + 2 < flat.length; j += 3) {
          if (s >= spheres.length) break;
          var m = spheres[s++];
          m.visible = jointsToggle.checked;
          m.position.set(flat[j], flat[j + 1], flat[j + 2]);
        }
      }
      for (; s < spheres.length; s++) spheres[s].visible = false;
    }

    if (face.on) updateFaceCam(idx);

    frameSlider.value = String(idx);
    frameLabel.textContent = idx + 1 + " / " + anim.nFrames;
  }

  function loop(ts) {
    if (destroyed) return;
    requestAnimationFrame(loop);
    if (!ctx) return;
    if (anim.playing && anim.nFrames > 1) {
      if (!anim.lastTick) anim.lastTick = ts;
      var step = 1000 / (state.fps * (Number(speedSel.value) || 1));
      // Advance by elapsed time (may skip frames) so playback holds true fps
      // even when a mesh update takes longer than one display refresh.
      var advance = Math.floor((ts - anim.lastTick) / step);
      if (advance > 0) {
        anim.lastTick += advance * step;
        setFrame((anim.frame + advance) % anim.nFrames);
        // Measured playback rate: data-frames advanced per wall-clock second.
        if (!perf.t0) perf.t0 = ts;
        perf.count += advance;
        if (ts - perf.t0 >= 1000) {
          verEl.textContent = "v7 \u00b7 " + (perf.count * 1000 / (ts - perf.t0)).toFixed(1) + "fps";
          perf.t0 = ts;
          perf.count = 0;
        }
      }
    } else {
      perf.t0 = 0;
      perf.count = 0;
    }
    ctx.renderer.render(ctx.scene, ctx.camera);
  }

  function frameCamera() {
    if (!ctx || !jointData.frames.length || !jointData.frames[0].length) return;
    var flat = jointData.frames[0][0];
    var minY = Infinity;
    var maxY = -Infinity;
    var n = Math.floor(flat.length / 3);
    var cx = 0;
    var cy = 0;
    var cz = 0;
    for (var j = 0; j < flat.length; j += 3) {
      cx += flat[j];
      cy += flat[j + 1];
      cz += flat[j + 2];
      minY = Math.min(minY, flat[j + 1]);
      maxY = Math.max(maxY, flat[j + 1]);
    }
    ctx.orbit.cx = cx / n;
    ctx.orbit.cy = cy / n;
    ctx.orbit.cz = cz / n;
    ctx.orbit.dist = Math.max(1.5, (maxY - minY) * 2.2);
    ctx.applyCamera();
  }

  function clearJoints() {
    if (ctx && ctx.jointGroup) {
      ctx.scene.remove(ctx.jointGroup);
      ctx.jointGroup = null;
    }
    jointData.frames = [];
  }

  function clearMesh() {
    if (meshData && ctx) {
      ctx.scene.remove(meshData.mesh);
      meshData.geometry.dispose();
    }
    meshData = null;
    face.crownIdx = -1; // new mesh = re-pick the crown anchor
    face.sc = null;
    face.su = null;
  }

  function buildJoints(data) {
    var R = window.THREE;
    clearJoints();
    jointData.frames = data.frames || [];
    state.fps = Number(data.fps) > 0 ? Number(data.fps) : state.fps;
    anim.nFrames = Math.max(meshData ? meshData.nFrames : 0, jointData.frames.length);
    frameSlider.max = String(Math.max(0, anim.nFrames - 1));

    var maxJoints = 0;
    if (jointData.frames.length) {
      var first = jointData.frames[0];
      for (var p = 0; p < first.length; p++) maxJoints += Math.floor(first[p].length / 3);
    }
    var group = new R.Group();
    var geo = new R.SphereGeometry(0.012, 8, 8);
    var mat = new R.MeshStandardMaterial({ color: 0x53f2a7, roughness: 0.4 });
    for (var i = 0; i < maxJoints; i++) {
      var mesh = new R.Mesh(geo, mat);
      mesh.visible = false;
      group.add(mesh);
    }
    ctx.jointGroup = group;
    ctx.scene.add(group);
    frameCamera();
    setFrame(Math.min(anim.frame, anim.nFrames - 1));
  }

  function buildMesh(buffer) {
    var R = window.THREE;
    clearMesh();
    // v3 header: nFrames, nVerts, nFaces (u32), fps (f32), hasColors (u32),
    // hasHeadMask (u32), then faces, optional per-vertex RGB, optional head
    // mask (u8/vertex), then per-frame vertex blocks.
    var head = new DataView(buffer, 0, 24);
    var nFrames = head.getUint32(0, true);
    var nVerts = head.getUint32(4, true);
    var nFaces = head.getUint32(8, true);
    var fps = head.getFloat32(12, true);
    var hasColors = head.getUint32(16, true);
    var hasHeadMask = head.getUint32(20, true);
    if (fps > 0) state.fps = fps;
    var offset = 24;
    var faces = new Uint32Array(buffer, offset, nFaces * 3);
    offset += nFaces * 12;
    var colors = null;
    if (hasColors) {
      colors = new Float32Array(buffer, offset, nVerts * 3);
      offset += nVerts * 12;
    }
    var headMask = null;
    if (hasHeadMask) {
      headMask = new Uint8Array(buffer, offset, nVerts);
      // Mask block is zero-padded to 4-byte alignment for the float view below.
      offset += nVerts + ((4 - (nVerts % 4)) % 4);
    }
    var verts = new Float32Array(buffer, offset, nFrames * nVerts * 3);

    var geometry = new R.BufferGeometry();
    var pos = new Float32Array(nVerts * 3);
    pos.set(verts.subarray(0, nVerts * 3));
    geometry.setAttribute("position", new R.BufferAttribute(pos, 3));
    geometry.setIndex(new R.BufferAttribute(faces, 1));
    geometry.computeVertexNormals();
    var material;
    if (colors) {
      geometry.setAttribute("color", new R.BufferAttribute(new Float32Array(colors), 3));
      material = new R.MeshStandardMaterial({ vertexColors: true, roughness: 0.55, metalness: 0.05, side: R.DoubleSide });
    } else {
      material = new R.MeshStandardMaterial({ color: 0x8fa7ff, roughness: 0.55, metalness: 0.05, side: R.DoubleSide });
    }
    var mesh = new R.Mesh(geometry, material);
    mesh.visible = meshToggle.checked;
    ctx.scene.add(mesh);

    // Head-only index buffer: triangles whose three corners are all in the
    // head mask. Used by the "body" toggle to hide everything below the neck.
    var headIndex = null;
    if (headMask) {
      var kept = [];
      for (var t = 0; t < nFaces; t++) {
        var i0 = faces[t * 3], i1 = faces[t * 3 + 1], i2 = faces[t * 3 + 2];
        if (headMask[i0] && headMask[i1] && headMask[i2]) kept.push(i0, i1, i2);
      }
      if (kept.length) headIndex = new R.BufferAttribute(new Uint32Array(kept), 1);
    }
    bodyLabel.style.display = headIndex ? "flex" : "none";

    meshData = {
      nFrames: nFrames,
      nVerts: nVerts,
      verts: verts,
      geometry: geometry,
      mesh: mesh,
      fullIndex: geometry.index,
      headIndex: headIndex
    };
    anim.nFrames = Math.max(anim.nFrames, nFrames);
    frameSlider.max = String(Math.max(0, anim.nFrames - 1));
    // Nodes spawned as a dedicated face viewer start in face-cam mode with the
    // body hidden (head close-up only).
    if (state.faceCam && state.faceTrack && !face.on) {
      setFaceMode(true);
      if (headIndex) bodyToggle.checked = false;
    }
    applyBodyVisibility();
    setFrame(anim.frame);
  }

  function loadData() {
    if (!ctx || destroyed) return;
    var token = ++loadToken;

    if (!state.jointsUrl && !state.meshUrl) {
      clearJoints();
      clearMesh();
      anim.nFrames = 0;
      frameLabel.textContent = "run node";
      statusEl.textContent = "";
      return;
    }

    if (state.jointsUrl && state.jointsUrl !== loadedJointsUrl) {
      fetch(state.jointsUrl)
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (data) {
          if (destroyed || token !== loadToken) return;
          loadedJointsUrl = state.jointsUrl;
          buildJoints(data);
        })
        .catch(function (err) {
          if (token === loadToken) statusEl.textContent = "joints failed: " + (err && err.message ? err.message : err);
        });
    }

    if (state.meshUrl && state.meshUrl !== loadedMeshUrl) {
      statusEl.textContent = "mesh: loading...";
      fetch(state.meshUrl)
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.arrayBuffer();
        })
        .then(function (buffer) {
          if (destroyed || token !== loadToken) return;
          loadedMeshUrl = state.meshUrl;
          buildMesh(buffer);
          statusEl.textContent = "";
        })
        .catch(function (err) {
          if (token === loadToken) statusEl.textContent = "mesh failed: " + (err && err.message ? err.message : err);
        });
    } else if (!state.meshUrl && state.jointsUrl) {
      statusEl.textContent = "no mesh in state - re-run node";
    }
  }

  playBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    anim.playing = !anim.playing;
    anim.lastTick = 0;
    playBtn.textContent = anim.playing ? "Pause" : "Play";
  });
  frameSlider.addEventListener("input", function (e) {
    e.stopPropagation();
    anim.playing = false;
    playBtn.textContent = "Play";
    setFrame(Number(frameSlider.value));
  });
  meshToggle.addEventListener("change", function (e) {
    e.stopPropagation();
    if (meshData && meshData.mesh) meshData.mesh.visible = meshToggle.checked;
  });
  bodyToggle.addEventListener("change", function (e) {
    e.stopPropagation();
    applyBodyVisibility();
  });
  jointsToggle.addEventListener("change", function (e) {
    e.stopPropagation();
    setFrame(anim.frame);
  });
  faceBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    setFaceMode(!face.on);
  });
  fullBtn.addEventListener("click", function (e) {
    e.stopPropagation();
    // Fullscreen the whole widget (viewport + control strip) so the play bar
    // stays available in fullscreen.
    if (document.fullscreenElement === root) {
      if (document.exitFullscreen) document.exitFullscreen();
    } else if (root.requestFullscreen) {
      root.requestFullscreen().catch(function (err) {
        statusEl.textContent = "fullscreen blocked: " + (err && err.message ? err.message : err);
      });
    }
  });
  document.addEventListener("fullscreenchange", onFullscreenChange);
  function onFullscreenChange() {
    // The browser resizes the widget in/out of fullscreen; re-fit the canvas.
    fullBtn.textContent = document.fullscreenElement === root ? "\u2715" : "\u26f6";
    resizeRenderer();
  }

  function resizeRenderer() {
    if (!ctx || destroyed) return;
    var w = wrap.clientWidth || 640;
    var h = wrap.clientHeight || 360;
    if (w < 8 || h < 8) return;
    // Render at the actual displayed size (x devicePixelRatio) so scaling the
    // node scales the resolution instead of stretching a fixed-size canvas.
    ctx.renderer.setPixelRatio(window.devicePixelRatio || 1);
    ctx.renderer.setSize(w, h, false);
    ctx.camera.aspect = w / h;
    ctx.camera.updateProjectionMatrix();
  }

  loadThree()
    .then(function () {
      if (destroyed) return;
      ctx = buildScene();
      ctx.renderer.domElement.style.width = "100%";
      ctx.renderer.domElement.style.height = "100%";
      ctx.renderer.domElement.style.display = "block";
      ctx.renderer.domElement.style.position = "absolute";
      ctx.renderer.domElement.style.inset = "0";
      if (msg && msg.parentNode) msg.parentNode.removeChild(msg);
      msg = null;
      wrap.insertBefore(ctx.renderer.domElement, wrap.firstChild);
      resizeRenderer();
      if (window.ResizeObserver) {
        resizeObserver = new ResizeObserver(resizeRenderer);
        resizeObserver.observe(wrap);
      }
      setFaceButton();
      requestAnimationFrame(loop);
      loadData();
    })
    .catch(function () {
      setMsg("Failed to load three.js (offline?).");
    });

  function externalUpdate(newProps) {
    if (destroyed) return;
    var next = parseValue(newProps && newProps.value);
    var changed = next.jointsUrl !== state.jointsUrl || next.meshUrl !== state.meshUrl;
    state = next;
    if (changed) loadData();
  }

  return {
    update: externalUpdate,
    cleanup: function cleanup() {
      destroyed = true;
      document.removeEventListener("fullscreenchange", onFullscreenChange);
      if (resizeObserver) {
        resizeObserver.disconnect();
        resizeObserver = null;
      }
      if (ctx && ctx.renderer && ctx.renderer.dispose) ctx.renderer.dispose();
      ctx = null;
      container.innerHTML = "";
    }
  };
}
