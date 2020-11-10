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
import * as tf from '@tensorflow/tfjs-core';
import '@tensorflow/tfjs-backend-webgl';

import * as tfjsWasm from '@tensorflow/tfjs-backend-wasm';

import { TRIANGULATION } from './triangulation';
import { UV_COORDS } from '@tensorflow-models/facemesh/dist/uv_coords';

tfjsWasm.setWasmPath(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/tfjs-backend-wasm.wasm`);

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

let model, videoWidth, videoHeight, video, canvas, scene, camera, renderer, faces, myUVCoords;
let N_KEYPOINTS = 468;
const N_FACES = 3;

const VIDEO_SIZE = 500;
const mobile = isMobile();
const state = {
  backend: 'webgl',
  maxFaces: N_FACES,
  triangulateMesh: true
};


async function setupCamera() {
  video = document.getElementById('video');

  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
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

function updateUVs() {
  faces.forEach(face => {
    var geom = face.geometry
    for (let i = 0; i < TRIANGULATION.length / 3; i++) {
      for (let j = 0; j < 3; j++) {
        var vert_1 = TRIANGULATION[3 * i + j]
        geom.faceVertexUvs[0][i][j].x = myUVCoords[vert_1][0];
        geom.faceVertexUvs[0][i][j].y = 1 - myUVCoords[vert_1][1];
      }
    }

    geom.uvsNeedUpdate = true;
  })

}

function initFaceMeshes() {
  faces = []
  for (var i = 0; i < N_FACES; i++) {
    var geom = new THREE.Geometry();

    // add the geometry's vertices
    for (let i = 0; i < N_KEYPOINTS; i++) {
      var new_point = new THREE.Vector3(0, 0, 0);
      geom.vertices.push(new_point);
    }

    // add the geometry's faces
    geom.faceVertexUvs[0] = [];
    for (let i = 0; i < TRIANGULATION.length / 3; i++) {
      var vert_1 = TRIANGULATION[i * 3]
      var vert_2 = TRIANGULATION[i * 3 + 1]
      var vert_3 = TRIANGULATION[i * 3 + 2]
      geom.faces.push(new THREE.Face3(vert_1, vert_2, vert_3)
      )
      geom.faceVertexUvs[0].push([
        new THREE.Vector2(myUVCoords[vert_1][0], 1 - myUVCoords[vert_1][1]),
        new THREE.Vector2(myUVCoords[vert_2][0], 1 - myUVCoords[vert_2][1]),
        new THREE.Vector2(myUVCoords[vert_3][0], 1 - myUVCoords[vert_3][1])
      ])
    }

    geom.uvsNeedUpdate = true;

    var face = new THREE.Mesh(geom, getMaterial());
    scene.add(face);
    faces.push(face)
  }
}

function getMaterial() {
  var dolph = document.getElementById('dolph')

  var texture_dolph = new THREE.Texture(dolph);
  texture_dolph.needsUpdate = true;

  return new THREE.MeshBasicMaterial({ map: texture_dolph });
}

async function takePic() {
  const myCanvas = document.createElement("canvas");
  // scale the canvas accordingly
  myCanvas.width = video.videoWidth;
  myCanvas.height = video.videoHeight;
  // draw the video at that frame
  myCanvas.getContext('2d')
    .drawImage(video, 0, 0, myCanvas.width, myCanvas.height);
  // convert it to a usable data URL
  var myDataURL = myCanvas.toDataURL();
  faces.forEach(face => {
    face.material.map = THREE.ImageUtils.loadTexture(myDataURL);
    face.material.needsUpdate = true;
  })

  const predictions = await model.estimateFaces(video);
  if (predictions.length > 0) {
    var prediction = predictions[0]
    const keypoints = prediction.scaledMesh;

    console.log(UV_COORDS)

    for (let i = 0; i < keypoints.length; i++) {
      const x = keypoints[i][0];
      const y = keypoints[i][1];
      const z = keypoints[i][2];
      console.log(x, y, z)
      console.log(videoWidth, videoHeight)
      myUVCoords[i] = [x / videoWidth, y / videoHeight]
    }

    updateUVs()

  }
}


async function renderPrediction() {
  const predictions = await model.estimateFaces(video);


  if (predictions.length > 0) {
    predictions.forEach((prediction, i) => {
      var geom = faces[i].geometry
      const keypoints = prediction.scaledMesh;

      for (let i = 0; i < keypoints.length; i++) {
        const x = keypoints[i][0];
        const y = keypoints[i][1];
        const z = keypoints[i][2];
        var vert = geom.vertices[i]
        vert.x = x - videoWidth / 2;
        vert.y = -y + videoHeight / 2;
        vert.z = 1;
        geom.verticesNeedUpdate = true;
      }


    });
  }

  requestAnimationFrame(renderPrediction);
  renderer.render(scene, camera);

};

async function main() {
  await tf.setBackend(state.backend);

  var capture_button = document.getElementById("take_pic")
  capture_button.onclick = takePic

  myUVCoords = UV_COORDS.map(arr => { return arr.slice() })

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


  var texture = new THREE.VideoTexture(video);
  var imageObject = new THREE.Mesh(
    new THREE.PlaneGeometry(videoWidth, videoHeight),
    new THREE.MeshBasicMaterial({ map: texture }));

  scene.add(imageObject);
  scene.add(camera)
  camera.position.z = 500;

  initFaceMeshes()
  model = await facemesh.load({ maxFaces: state.maxFaces });
  renderPrediction();

};

main();
