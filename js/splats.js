
import { OrbitControls } from "./OrbitControls.js";
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest";
const bucket = "https://storage.googleapis.com/mf_gaussian_splats"

const SH_C0 = 0.28209479177387814;

function convertToFeatureBuffer(rawBuffer) {
    // Parse header to find vertex count, property count, and header end
    const headerText = new TextDecoder().decode(new Uint8Array(rawBuffer, 0, Math.min(4096, rawBuffer.byteLength)));
    const headerEnd = headerText.indexOf("end_header\n") + "end_header\n".length;
    const vertexMatch = headerText.match(/element vertex (\d+)/);
    const numVertices = parseInt(vertexMatch[1]);
    const propMatches = headerText.match(/property\s+\w+\s+\w+/g);
    const numProps = propMatches.length;

    const newBuffer = rawBuffer.slice(0);
    const dataView = new DataView(newBuffer, headerEnd);
    const bytesPerVertex = numProps * 4;

    // Collect semantic values (indices 62-64) for percentile normalization
    const semValues = [[], [], []];
    for (let i = 0; i < numVertices; i++) {
        for (let c = 0; c < 3; c++) {
            semValues[c].push(dataView.getFloat32(i * bytesPerVertex + (62 + c) * 4, true));
        }
    }

    const lo = [], hi = [];
    for (let c = 0; c < 3; c++) {
        const sorted = semValues[c].slice().sort((a, b) => a - b);
        const n = sorted.length;
        lo.push(sorted[Math.floor(n * 0.01)]);
        hi.push(sorted[Math.floor(n * 0.99)]);
    }

    for (let i = 0; i < numVertices; i++) {
        const off = i * bytesPerVertex;
        for (let c = 0; c < 3; c++) {
            const raw = dataView.getFloat32(off + (62 + c) * 4, true);
            const range = hi[c] - lo[c] || 1;
            const norm = Math.max(0, Math.min(1, (raw - lo[c]) / range));
            const fdc = (norm - 0.5) / SH_C0;
            dataView.setFloat32(off + (6 + c) * 4, fdc, true);
        }
        // Zero out f_rest (indices 9-53)
        for (let j = 9; j <= 53; j++) {
            dataView.setFloat32(off + j * 4, 0, true);
        }
    }

    return newBuffer;
}

function shuffleArray(array) {
    for (let i = array.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [array[i], array[j]] = [array[j], array[i]];
    }
}

async function listFolders() {
    // Interacting with the google cloud storage APIS requires full on node installatiosn and such so meh, just parse the returned XML :) 
    let xmlDoc = await fetch(bucket).then(response => response.text()).then(xmlString => {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xmlString, 'text/xml');
        return xmlDoc;
    }).catch(error => {
        console.error('Error:', error);
    });
    const contentsElements = xmlDoc.getElementsByTagName('Contents');
    let nodeArray = [].slice.call(contentsElements);

    function xmlToKey(folder) {
        let key = folder.getElementsByTagName("Key")[0].textContent;
        let parts = key.split("/");
        return parts[0] + "/" + parts[1];
    }

    const keys = [...new Set(nodeArray.map(f => xmlToKey(f)))];
    return keys.filter(f => !f.includes("."))
}

export function createSplatView(splatParent) {
    const parentDiv = document.getElementById(splatParent);
    const canvas = parentDiv.querySelector("canvas");
    const progressDialog = parentDiv.querySelector("#progress-dialog");
    const progressIndicator = progressDialog.querySelector("#progress-indicator");
    const view = new Object();
    view.canvas = canvas;
    view.progressDialog = progressDialog;
    view.progressIndicator = progressIndicator;
    view.runningAnimation = null;
    view.loading = false;
    view.lastClick = new Date()
    view.bufferCache = {};  // Cache raw PLY buffers by name
    view.featureBufferCache = {};  // Cache feature-converted buffers
    return view;
}

export async function setSplatScene(name, view, local = false, mode = "rgb") {
    view.loading = true;
    view.lastClick = new Date();

    const firstLoad = !view.camera;

    if (firstLoad) {
        const startRadius = 5.0;

        const cameraData = new SPLAT.CameraData();
        cameraData.fx = 0.5 * view.canvas.offsetWidth;
        cameraData.fy = 0.5 * view.canvas.offsetHeight;

        view.camera = new SPLAT.Camera(cameraData);
        view.renderer = new SPLAT.WebGLRenderer(view.canvas);
        view.controls = new OrbitControls(view.camera, view.canvas, /*alpha=*/0.0, /*beta=*/0.0, /*radius=*/startRadius, /*enableKeyboardControls=*/false);
        view.controls.minAngle = -90;
        view.controls.maxAngle = 90;
        view.controls.minZoom = 0.001;
        view.controls.maxZoom = 10000.0;
        view.controls.zoomSpeed = 0.5;
        view.controls.panSpeed = 1.0;
        view.controls.orbitSpeed = 1.75;
        view.controls.maxPanDistance = undefined;
    }

    const scene = new SPLAT.Scene();

    view.progressDialog.show();
    view.progressIndicator.value = 0.0;

    // Fetch and cache the raw buffer (only downloads once per model)
    if (!view.bufferCache[name]) {
        const url = local ? name : bucket + "/" + name;
        view.bufferCache[name] = await fetch(url).then(r => r.arrayBuffer());
    }
    view.progressIndicator.value = 50;

    // Use cached buffer, convert to features in-memory if needed (cached too)
    let buffer = view.bufferCache[name];
    if (mode === "features") {
        if (!view.featureBufferCache[name]) {
            view.featureBufferCache[name] = convertToFeatureBuffer(buffer);
        }
        buffer = view.featureBufferCache[name];
    }

    const blob = new Blob([buffer], { type: "application/octet-stream" });
    const blobUrl = URL.createObjectURL(blob);
    const splat = await SPLAT.PLYLoader.LoadAsync(blobUrl, scene, (progress) => (view.progressIndicator.value = 50 + progress * 50));
    URL.revokeObjectURL(blobUrl);

    // NerfStudio and gsplat.js don't agree on coordinate frames, just rotate after loading for now.
    const rotation = new SPLAT.Vector3(Math.PI - Math.PI / 20.0, Math.PI, Math.PI);
    splat.rotation = SPLAT.Quaternion.FromEuler(rotation);
    splat.applyRotation();

    view.progressDialog.close();

    // Swap the scene (camera/controls/renderer persist)
    view.currentScene = scene;

    if (firstLoad) {
        view.canvas.addEventListener("mousedown", function () {
            view.lastClick = new Date();
            view.interacting = true;
        });
        view.canvas.addEventListener("mouseup", function () {
            view.lastClick = new Date();
            view.interacting = false;
        });

        // Render loop
        let previousTimestamp = undefined;
        let previousDeltaTime = undefined;
        const animate = (timestamp) => {
            var deltaTime = 0.0;
            if (previousTimestamp !== undefined) {
                deltaTime = (timestamp - previousTimestamp) / 1000;
            }
            if (deltaTime > 0.1 && previousDeltaTime !== undefined) {
                deltaTime = previousDeltaTime;
            }
            previousTimestamp = timestamp;
            previousDeltaTime = deltaTime;

            if (!view.interacting) {
                const timeToSpin = 0.5;
                const accelTime = 4.0;
                const maxSpinSpeed = 0.2;

                const timeSinceClick = view.lastClick == undefined ? undefined : (new Date() - view.lastClick) / 1000.0;
                if (timeSinceClick > timeToSpin || timeSinceClick === undefined) {
                    const speed = timeSinceClick === undefined ? maxSpinSpeed : Math.min(Math.max(timeSinceClick / accelTime - timeToSpin, 0.0), 1.0) * maxSpinSpeed;
                    view.controls.rotateCameraAngle(speed * deltaTime, 0.0);
                }
            }

            view.controls.update();
            view.renderer.render(view.currentScene, view.camera);
            requestAnimationFrame(animate);
        };

        requestAnimationFrame(animate);
    }

    view.loading = false;
}

export async function setupCarousel(view, carousel) {
    let files = await listFolders();

    shuffleArray(files);

    const prototype = carousel.querySelector("#splat-carousel-prototype");
    const elements = Object.fromEntries(
        files.map(f => [f, prototype.firstElementChild.cloneNode(true)])
    );

    async function onClickSplatThumb(splatName) {
        if (view.loading) {
            // Only allow one splat at a time to load otherwise things get messy.
            return;
        }
        const elem = elements[splatName];
        if (elem.classList.contains("active")) {
            return;
        }

        const itemsParent = carousel.getElementsByClassName("splat-carousel-items")[0];
        const items = [...itemsParent.getElementsByClassName('splat-carousel-item')];
        currentIndex = items.indexOf(elem);

        elem.classList.add("loading");

        await setSplatScene(splatName + "/splat.ply", view)

        Object.values(elements).forEach(e => {
            e.classList.remove("active");
        });
        elem.classList.remove("loading");
        elem.classList.add("active");
    }

    // TODO: probably want to download a bunch of metadata-y thing here.
    for (var i = 0; i < files.length; ++i) {
        const file = files[i];

        // Wrap in function to capture CURRENT file.
        // Js is bad :)
        function setup(file) {
            const card = elements[file];
            var startScroll = undefined;

            card.addEventListener("mousedown", function () { startScroll = itemsParent.scrollLeft; });
            card.addEventListener("mouseup", function () {
                console.log(Math.abs(itemsParent.scrollLeft - startScroll));
                if (Math.abs(itemsParent.scrollLeft - startScroll) < 10) {
                    onClickSplatThumb(file);
                }
            });

            const img = card.querySelector("img");
            img.src = bucket + "/" + file + "/input.png";

            let isAnimating = false;
            let latestEvent = null;

            card.addEventListener('pointermove', (e) => {
                latestEvent = e;
                if (!isAnimating) {
                    isAnimating = true;
                    requestAnimationFrame(updateCardTransform);
                }
            });

            function updateCardTransform() {
                const e = latestEvent;
                if (e === null || e === undefined) {
                    isAnimating = false;
                    return;
                }
                const cardRect = card.getBoundingClientRect();
                const centerX = cardRect.left + cardRect.width / 2;
                const centerY = cardRect.top + cardRect.height / 2;

                const mouseX = e.clientX - centerX;
                const mouseY = e.clientY - centerY;

                const rotateY = (mouseX / cardRect.width) * 35;
                const rotateX = -(mouseY / cardRect.height) * 35;

                card.style.transform = `translateZ(15px) rotateY(${rotateY}deg) rotateX(${rotateX}deg)`;

                isAnimating = false;
                if (latestEvent !== e) { // Check for new event
                    requestAnimationFrame(updateCardTransform);
                }
            }

            card.addEventListener('mouseleave', () => {
                latestEvent = null; // Clear event to stop animation
                card.style.transform = ''; // Reset on leave
            });
            prototype.parentNode.appendChild(card);
        }

        setup(file);
    }

    prototype.remove();

    const itemsParent = carousel.getElementsByClassName("splat-carousel-items")[0];
    const items = [...itemsParent.getElementsByClassName('splat-carousel-item')];
    let currentIndex = 0;

    function scrollToTarget() {
        items[currentIndex].scrollIntoView({ behavior: "smooth", inline: "center", block: "nearest" });
    }

    carousel.querySelector('.splat-carousel-button.left').addEventListener('mousedown', () => {
        currentIndex = (currentIndex + items.length - 2) % items.length;
        scrollToTarget();
    });

    carousel.querySelector('.splat-carousel-button.right').addEventListener('mousedown', () => {
        currentIndex = (currentIndex + 2) % items.length;
        scrollToTarget();
    });

    let mouseDown = false;
    let startX, scrollLeft;

    const startDragging = (e) => {
        mouseDown = true;
        startX = e.pageX - itemsParent.offsetLeft;
        scrollLeft = itemsParent.scrollLeft;
    }

    const stopDragging = (e) => {
        e.preventDefault();
        mouseDown = false;
    }

    const move = (e) => {
        e.preventDefault();
        if (!mouseDown) { return; }
        const x = e.pageX - itemsParent.offsetLeft;
        const scroll = x - startX;
        itemsParent.scrollLeft = scrollLeft - scroll;
    }

    // Add the event listeners
    itemsParent.addEventListener('mousemove', move, false);
    itemsParent.addEventListener('mousedown', startDragging, false);
    itemsParent.addEventListener('mouseup', stopDragging, false);
    itemsParent.addEventListener('mouseleave', stopDragging, false);

    // Activate the first thumbnail.
    onClickSplatThumb(files[0]);
}
