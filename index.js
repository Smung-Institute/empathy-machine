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

let model, videoWidth, videoHeight, video, scene, camera, renderer, faces, myUVCoords, renderer_canvas, output_canvas, video_valid, stream, particleSystem, t0;
let N_KEYPOINTS = 468;
const N_FACES = 20;

const VIDEO_SIZE = 500;
const mobile = isMobile();
const state = {
  backend: 'webgl',
  maxFaces: N_FACES,
  triangulateMesh: true
};


async function setupCamera(facing_mode) {
  if (video_valid) {
    stream.getTracks().forEach(function (track) {
      track.stop();
    });
    //video.srcObject.stop();
  }
  video_valid = false;
  video = document.getElementById('video');



  if (!isMobile()) {
    facing_mode = "user";
  }

  stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: facing_mode,
      width: mobile ? undefined : VIDEO_SIZE,
      height: mobile ? undefined : VIDEO_SIZE
    },
  });
  video.srcObject = stream;
  video.play();
  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video_valid = true;
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


    for (let i = 0; i < keypoints.length; i++) {
      const x = keypoints[i][0];
      const y = keypoints[i][1];
      const z = keypoints[i][2];
      myUVCoords[i] = [x / videoWidth, y / videoHeight]
    }

    updateUVs()

  }

  setupCamera("environment");


}


async function renderPrediction() {
  if (video_valid) {
    const predictions = await model.estimateFaces(video);

    document.getElementById("splash").style.display = "none";
    document.getElementById("loading").style.display = "none";

    if (predictions.length > 0) {
      predictions.forEach((prediction, i) => {
        faces[i].visible = true;
        var geom = faces[i].geometry
        const keypoints = prediction.scaledMesh;

        var ann = prediction.annotations
        var upper_lip = new THREE.Vector3(...ann.lipsUpperInner[6])
        var lower_lip = new THREE.Vector3(...ann.lipsLowerInner[6])
        var left_corner = new THREE.Vector3(...ann.lipsUpperInner[0])
        var right_corner = new THREE.Vector3(...ann.lipsUpperInner[10])
        var right_cheek = new THREE.Vector3(...ann.rightCheek[0])
        var left_cheek = new THREE.Vector3(...ann.leftCheek[0])
        var between_eyes = new THREE.Vector3(...ann.midwayBetweenEyes[0])
        ///particleSystem.position.copy(left_cheek);

        var corner_to_corner_line = right_corner.clone().sub(left_corner).projectOnPlane(new THREE.Vector3(0, 0, 1))

        var corner_to_lower_line = lower_lip.clone().sub(left_corner).projectOnPlane(new THREE.Vector3(0, 0, 1))

        //var jaw_to_jaw_line = left_jaw.sub(right_jaw)
        var cheek_to_cheek_line = left_cheek.clone().sub(right_cheek);

        var side = cheek_to_cheek_line.clone().normalize();
        var forward = side.clone().cross(between_eyes.clone().sub(left_cheek)).normalize();
        var up = side.clone().cross(forward).normalize();
        var m = new THREE.Matrix4();
        m.set(
          side.x, side.y, side.z, 0,
          up.x, up.y, up.z, 0,
          forward.x, forward.y, forward.z, 0,
          0, 0, 0, 1
        )


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
    // TODO: Will, uncomment this.
    for (var i = predictions.length; i < N_FACES; i++) {
      faces[i].visible = false;
    }

    renderer.render(scene, camera);
    var output_canvas = document.getElementById('output');
    var output_context = output_canvas.getContext('2d');
    var window_ratio = window.innerWidth / window.innerHeight;
    if ((videoWidth / videoHeight) > window_ratio) {
      var y = videoHeight;
      var y_offset = 0;
      var x = y * window_ratio;
      var x_offset = (videoWidth - x) / 2;
    }
    else {
      var x = videoWidth
      var y = x / window_ratio;
      var x_offset = 0;
      var y_offset = (videoHeight - y) / 2;
    }
    output_context.drawImage(renderer_canvas, x_offset, y_offset, x, y,
      0, 0, window.innerWidth, window.innerHeight);

  }
  else {
    console.log("Video invalid");
  }

  requestAnimationFrame(renderPrediction);
};

async function main() {
  video_valid = false;
  await tf.setBackend(state.backend);


  document.body.addEventListener('click', takePic, true);

  myUVCoords = UV_COORDS.map(arr => { return arr.slice() })

  try {
    var locOrientation = screen.lockOrientation || screen.mozLockOrientation || screen.msLockOrientation || screen.orientation.lock;
    if (locOrientation) {
      locOrientation('landscape').then(
        (success) => console.log(success)
        , (failure) => console.log(failure)
      );
    }
  }
  catch (err) {
  }
  finally {

  }
  await setupCamera("user");
  video.play();




  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  output_canvas = document.getElementById('output');
  output_canvas.width = window.innerWidth;
  output_canvas.height = window.innerHeight;


  renderer_canvas = document.createElement("canvas");
  renderer_canvas.width = videoWidth
  renderer_canvas.height = videoHeight


  renderer = new THREE.WebGLRenderer({ canvas: renderer_canvas });


  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(65, videoWidth / videoHeight, 0.1, 1000);


  var texture = new THREE.VideoTexture(video);
  var imageObject = new THREE.Mesh(
    new THREE.PlaneGeometry(videoWidth, videoHeight),
    new THREE.MeshBasicMaterial({ map: texture }));

  scene.add(imageObject);
  scene.add(camera)
  camera.position.z = 350;



  initFaceMeshes()
  model = await facemesh.load({ maxFaces: state.maxFaces });
  renderPrediction();

};

main();
