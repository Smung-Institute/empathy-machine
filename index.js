/**
 * @license
 * Copyright 2020 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

import * as facemesh from '@tensorflow-models/facemesh';
import Stats from 'stats.js';
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

import { TRIANGULATION } from './triangulation';
import { version } from 'os';
import { defaultCoreCipherList } from 'constants';
import { UV_COORDS } from '@tensorflow-models/facemesh/dist/uv_coords';

tfjsWasm.setWasmPath(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`);

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model, ctx, videoWidth, videoHeight, video, canvas,
  scatterGLHasInitialized = false, scatterGL, scene, camera, renderer, points = [], verts = [], geom, face, material_dolph;


const VIDEO_SIZE = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.
const renderPointcloud = mobile === false;
const stats = new Stats();
const state = {
  backend: 'wasm',
  maxFaces: 1,
  triangulateMesh: true
};

if (renderPointcloud) {
  state.renderPointcloud = true;
}



function setupDatGui() {
  const gui = new dat.GUI();
  gui.add(state, 'backend', ['wasm', 'webgl', 'cpu'])
    .onChange(async backend => {
      await tf.setBackend(backend);
    });

  gui.add(state, 'maxFaces', 1, 20, 1).onChange(async val => {
    model = await facemesh.load({ maxFaces: val });
  });

  gui.add(state, 'triangulateMesh');

  if (renderPointcloud) {
    gui.add(state, 'renderPointcloud').onChange(render => {
      document.querySelector('#scatter-gl-container').style.display =
        render ? 'inline-block' : 'none';
    });
  }
}

async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      // Only setting the video to a specified size in order to accommodate a
      // point cloud, so on mobile devices accept the default size.
      width: mobile ? undefined : VIDEO_SIZE,
      height: mobile ? undefined : VIDEO_SIZE
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function renderPrediction() {
  stats.begin();

  const predictions = await model.estimateFaces(video);

  //  ctx.drawImage(
  //    video, 0, 0, videoWidth, videoHeight, 0, 0, canvas.width, canvas.height);
  //renderer.render(scene, camera);

  if (predictions.length > 0) {
    predictions.forEach(prediction => {
      const keypoints = prediction.scaledMesh;

      if (state.triangulateMesh) {
        if (verts.length == 0) {
          geom = new THREE.Geometry();
          while (verts.length < keypoints.length) {
            var new_point = new THREE.Vector3(0, 0, 0);
            verts.push(new_point)
            geom.vertices.push(new_point);
          }
          geom.faceVertexUvs[0] = [];
          for (let i = 0; i < TRIANGULATION.length / 3; i++) {
            var vert_1 = TRIANGULATION[i * 3]
            var vert_2 = TRIANGULATION[i * 3 + 1]
            var vert_3 = TRIANGULATION[i * 3 + 2]
            geom.faces.push(new THREE.Face3(vert_1, vert_2, vert_3)
            )
            geom.faceVertexUvs[0].push([
              new THREE.Vector2(UV_COORDS[vert_1][0], 1 - UV_COORDS[vert_1][1]),
              new THREE.Vector2(UV_COORDS[vert_2][0], 1 - UV_COORDS[vert_2][1]),
              new THREE.Vector2(UV_COORDS[vert_3][0], 1 - UV_COORDS[vert_3][1])
            ])
          }
          console.log(UV_COORDS)
          // geom.faceVertexUvs[0] = UV_COORDS.map(v => THREE.Vector2(v[0], v[1]));
          geom.uvsNeedUpdate = true;
          var object = new THREE.Mesh(geom, material_dolph);
          scene.add(object);
        }
        for (let i = 0; i < keypoints.length; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];
          const z = keypoints[i][2];
          var vert = verts[i]
          vert.x = x - videoWidth / 2;
          vert.y = -y + videoHeight / 2;
          vert.z = 1;
          geom.verticesNeedUpdate = true;
        }

      } else {
        while (points.length < keypoints.length) {
          var geometry = new THREE.SphereGeometry(2, 4, 4);
          var material = new THREE.MeshBasicMaterial({ color: 0xffff00 });
          var sphere = new THREE.Mesh(geometry, material);
          scene.add(sphere);
          points.push(sphere)
        }
        for (let i = 0; i < keypoints.length; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];
          var sphere = points[i]
          sphere.position.x = x - videoWidth / 2;
          sphere.position.y = -y + videoHeight / 2;
        }
      }
    });

    if (renderPointcloud && state.renderPointcloud && scatterGL != null) {
      const pointsData = predictions.map(prediction => {
        let scaledMesh = prediction.scaledMesh;
        return scaledMesh.map(point => ([-point[0], -point[1], -point[2]]));
      });

      let flattenedPointsData = [];
      for (let i = 0; i < pointsData.length; i++) {
        flattenedPointsData = flattenedPointsData.concat(pointsData[i]);
      }
      const dataset = new ScatterGL.Dataset(flattenedPointsData);

      if (!scatterGLHasInitialized) {
        scatterGL.render(dataset);
      } else {
        scatterGL.updateDataset(dataset);
      }
      scatterGLHasInitialized = true;
    }
  }

  stats.end();
  requestAnimationFrame(renderPrediction);
  renderer.render(scene, camera);

};

async function main() {
  await tf.setBackend(state.backend);
  setupDatGui();



  stats.showPanel(0);  // 0: fps, 1: ms, 2: mb, 3+: custom
  document.getElementById('main').appendChild(stats.dom);
  var dolph = document.getElementById('dolph')

  var texture_dolph = new THREE.Texture(dolph);
  texture_dolph.needsUpdate = true;

  material_dolph = new THREE.MeshBasicMaterial({ map: texture_dolph });

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;


  canvas = document.getElementById('output');
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector('.canvas-wrapper');
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;


  renderer = new THREE.WebGLRenderer({ canvas: canvas });

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(65, videoWidth / videoHeight, 0.1, 1000);
  var light = new THREE.AmbientLight(0xffffff);
  scene.add(light);

  const loader = new THREE.TextureLoader();

  const boxWidth = 100;
  const boxHeight = 100;
  const boxDepth = 100;
  const geometry = new THREE.BoxGeometry(boxWidth, boxHeight, boxDepth);
  const material_test = new THREE.MeshBasicMaterial({
    map: loader.load("https://d2gg9evh47fn9z.cloudfront.net/800px_COLOURBOX4404617.jpg"),
  });
  //scene.add(cube);

  //renderer.setSize( window.innerWidth, window.innerHeight );

  var texture = new THREE.VideoTexture(video);
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  texture.format = THREE.RGBFormat;

  var imageObject = new THREE.Mesh(
    new THREE.PlaneGeometry(videoWidth, videoHeight),
    new THREE.MeshBasicMaterial({ map: texture }));

  scene.add(imageObject);
  scene.add(camera)
  camera.position.z = 500;


  /*
  ctx = canvas.getContext('webgl');
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = '#32EEDB';
  ctx.strokeStyle = '#32EEDB';
  ctx.lineWidth = 0.5;
*/


  model = await facemesh.load({ maxFaces: state.maxFaces });
  renderPrediction();

  if (renderPointcloud) {
    document.querySelector('#scatter-gl-container').style =
      `width: ${VIDEO_SIZE}px; height: ${VIDEO_SIZE}px;`;

    scatterGL = new ScatterGL(
      document.querySelector('#scatter-gl-container'),
      { 'rotateOnStart': false, 'selectEnabled': false });
  }
};

main();
