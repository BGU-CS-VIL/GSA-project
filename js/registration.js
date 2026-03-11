import { OrbitControls } from "./OrbitControls.js";
import * as SPLAT from "https://cdn.jsdelivr.net/npm/gsplat@latest";

// PLY property layout: 65 floats per vertex
// 0-2: x,y,z  3-5: nx,ny,nz  6-8: f_dc_0/1/2  9-53: f_rest_0..44
// 54: opacity  55-57: scale_0/1/2  58-61: rot_0/1/2/3  62-64: semantic_0/1/2
const FLOATS_PER_VERTEX = 65;
const SH_C0 = 0.28209479177387814;

// ==================== PLY Parsing ====================

async function parsePLY(url, onProgress) {
    const response = await fetch(url);
    const buffer = await response.arrayBuffer();
    const bytes = new Uint8Array(buffer);

    // Find end_header
    let headerEnd = 0;
    const decoder = new TextDecoder();
    for (let i = 0; i < Math.min(bytes.length, 5000); i++) {
        if (bytes[i] === 0x0A) { // newline
            const line = decoder.decode(bytes.slice(headerEnd, i)).trim();
            if (line === "end_header") {
                headerEnd = i + 1;
                break;
            }
            headerEnd = i + 1;
        }
    }

    // Parse header to get vertex count
    const headerText = decoder.decode(bytes.slice(0, headerEnd));
    const vertexMatch = headerText.match(/element vertex (\d+)/);
    const numVertices = parseInt(vertexMatch[1]);

    // Parse properties to count floats per vertex
    const propLines = headerText.split('\n').filter(l => l.startsWith('property'));
    const numProps = propLines.length;

    const dataView = new DataView(buffer, headerEnd);
    const vertexData = new Float32Array(numVertices * numProps);

    for (let i = 0; i < numVertices * numProps; i++) {
        vertexData[i] = dataView.getFloat32(i * 4, true); // little-endian
    }

    if (onProgress) onProgress(1.0);

    return { vertexData, numVertices, numProps, headerText, headerEnd, rawBuffer: buffer };
}

function extractXYZ(vertexData, numVertices, numProps) {
    const xyz = new Float32Array(numVertices * 3);
    for (let i = 0; i < numVertices; i++) {
        const off = i * numProps;
        xyz[i * 3] = vertexData[off];
        xyz[i * 3 + 1] = vertexData[off + 1];
        xyz[i * 3 + 2] = vertexData[off + 2];
    }
    return xyz;
}

function extractSemantics(vertexData, numVertices, numProps) {
    const sem = new Float32Array(numVertices * 3);
    for (let i = 0; i < numVertices; i++) {
        const off = i * numProps;
        sem[i * 3] = vertexData[off + 62];
        sem[i * 3 + 1] = vertexData[off + 63];
        sem[i * 3 + 2] = vertexData[off + 64];
    }
    return sem;
}

// ==================== Linear Algebra Helpers ====================

function vecSub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function vecAdd(a, b) { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function vecScale(a, s) { return [a[0] * s, a[1] * s, a[2] * s]; }
function vecNorm(a) { return Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]); }

function matMul3x3(A, B) {
    // A, B are [9] row-major
    const C = new Float64Array(9);
    for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
            for (let k = 0; k < 3; k++)
                C[i * 3 + j] += A[i * 3 + k] * B[k * 3 + j];
    return C;
}

function matVec3(M, v) {
    return [
        M[0] * v[0] + M[1] * v[1] + M[2] * v[2],
        M[3] * v[0] + M[4] * v[1] + M[5] * v[2],
        M[6] * v[0] + M[7] * v[1] + M[8] * v[2],
    ];
}

function matTranspose3x3(M) {
    return new Float64Array([M[0], M[3], M[6], M[1], M[4], M[7], M[2], M[5], M[8]]);
}

function det3x3(M) {
    return M[0] * (M[4] * M[8] - M[5] * M[7])
         - M[1] * (M[3] * M[8] - M[5] * M[6])
         + M[2] * (M[3] * M[7] - M[4] * M[6]);
}

function identity3x3() {
    return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
}

// SVD for 3x3 via Jacobi iteration (sufficient for ICP)
function svd3x3(H) {
    // Compute H^T H
    const HtH = matMul3x3(matTranspose3x3(H), H);

    // Jacobi eigenvalue decomposition of H^T H
    let V = identity3x3();
    let D = new Float64Array(HtH);

    for (let iter = 0; iter < 50; iter++) {
        // Find largest off-diagonal
        let maxVal = 0, p = 0, q = 1;
        for (let i = 0; i < 3; i++) {
            for (let j = i + 1; j < 3; j++) {
                const val = Math.abs(D[i * 3 + j]);
                if (val > maxVal) { maxVal = val; p = i; q = j; }
            }
        }
        if (maxVal < 1e-12) break;

        const app = D[p * 3 + p], aqq = D[q * 3 + q], apq = D[p * 3 + q];
        const tau = (aqq - app) / (2 * apq);
        const t = Math.sign(tau) / (Math.abs(tau) + Math.sqrt(1 + tau * tau));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;

        // Rotate D
        const newD = new Float64Array(D);
        newD[p * 3 + p] = c * c * app - 2 * s * c * apq + s * s * aqq;
        newD[q * 3 + q] = s * s * app + 2 * s * c * apq + c * c * aqq;
        newD[p * 3 + q] = 0;
        newD[q * 3 + p] = 0;
        for (let i = 0; i < 3; i++) {
            if (i !== p && i !== q) {
                const dip = D[i * 3 + p], diq = D[i * 3 + q];
                newD[i * 3 + p] = c * dip - s * diq;
                newD[p * 3 + i] = newD[i * 3 + p];
                newD[i * 3 + q] = s * dip + c * diq;
                newD[q * 3 + i] = newD[i * 3 + q];
            }
        }
        D = newD;

        // Rotate V
        const newV = new Float64Array(V);
        for (let i = 0; i < 3; i++) {
            const vip = V[i * 3 + p], viq = V[i * 3 + q];
            newV[i * 3 + p] = c * vip - s * viq;
            newV[i * 3 + q] = s * vip + c * viq;
        }
        V = newV;
    }

    // Singular values = sqrt of eigenvalues of H^T H
    const singularValues = [Math.sqrt(Math.max(D[0], 0)), Math.sqrt(Math.max(D[4], 0)), Math.sqrt(Math.max(D[8], 0))];

    // U = H V S^-1
    const U_cols = [];
    for (let j = 0; j < 3; j++) {
        const vCol = [V[0 * 3 + j], V[1 * 3 + j], V[2 * 3 + j]];
        const hv = matVec3(H, vCol);
        const sv = singularValues[j];
        if (sv > 1e-10) {
            U_cols.push([hv[0] / sv, hv[1] / sv, hv[2] / sv]);
        } else {
            U_cols.push([j === 0 ? 1 : 0, j === 1 ? 1 : 0, j === 2 ? 1 : 0]);
        }
    }

    const U = new Float64Array([
        U_cols[0][0], U_cols[1][0], U_cols[2][0],
        U_cols[0][1], U_cols[1][1], U_cols[2][1],
        U_cols[0][2], U_cols[1][2], U_cols[2][2],
    ]);

    return { U, S: singularValues, Vt: matTranspose3x3(V) };
}

// ==================== ICP Algorithm ====================

function findSemanticCorrespondences(srcXYZ, srcSem, tgtXYZ, tgtSem, nSrc, nTgt, semanticThresh) {
    const correspondences = [];
    for (let i = 0; i < nSrc; i++) {
        const si = [srcSem[i * 3], srcSem[i * 3 + 1], srcSem[i * 3 + 2]];
        let bestSpatialDist = Infinity;
        let bestIdx = -1;

        for (let j = 0; j < nTgt; j++) {
            const sj = [tgtSem[j * 3], tgtSem[j * 3 + 1], tgtSem[j * 3 + 2]];
            const semDist = vecNorm(vecSub(si, sj));
            if (semDist < semanticThresh) {
                const pi = [srcXYZ[i * 3], srcXYZ[i * 3 + 1], srcXYZ[i * 3 + 2]];
                const pj = [tgtXYZ[j * 3], tgtXYZ[j * 3 + 1], tgtXYZ[j * 3 + 2]];
                const spatialDist = vecNorm(vecSub(pi, pj));
                if (spatialDist < bestSpatialDist) {
                    bestSpatialDist = spatialDist;
                    bestIdx = j;
                }
            }
        }
        if (bestIdx >= 0) {
            correspondences.push([i, bestIdx]);
        }
    }
    return correspondences;
}

function icpStep(srcXYZ, srcSem, tgtXYZ, tgtSem, nSrc, nTgt, semanticThresh) {
    const correspondences = findSemanticCorrespondences(srcXYZ, srcSem, tgtXYZ, tgtSem, nSrc, nTgt, semanticThresh);

    if (correspondences.length < 3) {
        return { R: identity3x3(), t: [0, 0, 0], scale: 1.0, numCorr: correspondences.length, converged: true };
    }

    // Compute centroids of correspondences
    let srcCentroid = [0, 0, 0], tgtCentroid = [0, 0, 0];
    for (const [i, j] of correspondences) {
        srcCentroid = vecAdd(srcCentroid, [srcXYZ[i * 3], srcXYZ[i * 3 + 1], srcXYZ[i * 3 + 2]]);
        tgtCentroid = vecAdd(tgtCentroid, [tgtXYZ[j * 3], tgtXYZ[j * 3 + 1], tgtXYZ[j * 3 + 2]]);
    }
    const n = correspondences.length;
    srcCentroid = vecScale(srcCentroid, 1 / n);
    tgtCentroid = vecScale(tgtCentroid, 1 / n);

    // Compute H matrix and scale
    const H = new Float64Array(9);
    let srcNormSq = 0, tgtNormSq = 0;
    for (const [i, j] of correspondences) {
        const sc = vecSub([srcXYZ[i * 3], srcXYZ[i * 3 + 1], srcXYZ[i * 3 + 2]], srcCentroid);
        const tc = vecSub([tgtXYZ[j * 3], tgtXYZ[j * 3 + 1], tgtXYZ[j * 3 + 2]], tgtCentroid);
        for (let r = 0; r < 3; r++)
            for (let c = 0; c < 3; c++)
                H[r * 3 + c] += sc[r] * tc[c];
        srcNormSq += sc[0] * sc[0] + sc[1] * sc[1] + sc[2] * sc[2];
        tgtNormSq += tc[0] * tc[0] + tc[1] * tc[1] + tc[2] * tc[2];
    }

    // Scale
    let scale = srcNormSq > 0 ? Math.sqrt(tgtNormSq / srcNormSq) : 1.0;
    scale = Math.max(0.01, Math.min(100.0, scale));

    // SVD
    const { U, S, Vt } = svd3x3(H);
    let R = matMul3x3(matTranspose3x3(Vt), matTranspose3x3(U));

    if (det3x3(R) < 0) {
        const VtFixed = new Float64Array(Vt);
        VtFixed[6] *= -1; VtFixed[7] *= -1; VtFixed[8] *= -1;
        R = matMul3x3(matTranspose3x3(VtFixed), matTranspose3x3(U));
    }

    const t = vecSub(tgtCentroid, matVec3(R, vecScale(srcCentroid, scale)));

    // Check convergence
    const delta = vecNorm([R[0] - 1, R[4] - 1, R[8] - 1, R[1], R[2], R[3], R[5], R[6], R[7]]);

    return { R, t, scale, numCorr: n, converged: delta < 1e-4, delta };
}

function computeCentroid(xyz, n) {
    let cx = 0, cy = 0, cz = 0;
    for (let i = 0; i < n; i++) {
        cx += xyz[i * 3];
        cy += xyz[i * 3 + 1];
        cz += xyz[i * 3 + 2];
    }
    return [cx / n, cy / n, cz / n];
}

function applyTransformToXYZ(xyz, n, R, t, scale) {
    const out = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        const p = [xyz[i * 3], xyz[i * 3 + 1], xyz[i * 3 + 2]];
        const rp = matVec3(R, vecScale(p, scale));
        out[i * 3] = rp[0] + t[0];
        out[i * 3 + 1] = rp[1] + t[1];
        out[i * 3 + 2] = rp[2] + t[2];
    }
    return out;
}

function sampleIndices(n, nSamples) {
    if (nSamples >= n) return Array.from({ length: n }, (_, i) => i);
    const indices = [];
    const used = new Set();
    while (indices.length < nSamples) {
        const idx = Math.floor(Math.random() * n);
        if (!used.has(idx)) { used.add(idx); indices.push(idx); }
    }
    return indices;
}

function subsampleArrays(xyz, sem, n, indices) {
    const m = indices.length;
    const subXYZ = new Float32Array(m * 3);
    const subSem = new Float32Array(m * 3);
    for (let i = 0; i < m; i++) {
        const idx = indices[i];
        subXYZ[i * 3] = xyz[idx * 3];
        subXYZ[i * 3 + 1] = xyz[idx * 3 + 1];
        subXYZ[i * 3 + 2] = xyz[idx * 3 + 2];
        subSem[i * 3] = sem[idx * 3];
        subSem[i * 3 + 1] = sem[idx * 3 + 1];
        subSem[i * 3 + 2] = sem[idx * 3 + 2];
    }
    return { subXYZ, subSem, m };
}

export function runICP(srcXYZ, srcSem, tgtXYZ, tgtSem, nSrc, nTgt, {
    maxIter = 10,
    nSamples = 5000,
    semanticThresh = 0.1,
    onProgress = null,
} = {}) {
    // Subsample for speed
    const srcIdx = sampleIndices(nSrc, nSamples);
    const tgtIdx = sampleIndices(nTgt, nSamples);
    const { subXYZ: srcSub, subSem: srcSubSem, m: mSrc } = subsampleArrays(srcXYZ, srcSem, nSrc, srcIdx);
    const { subXYZ: tgtSub, subSem: tgtSubSem, m: mTgt } = subsampleArrays(tgtXYZ, tgtSem, nTgt, tgtIdx);

    // Initial centroid alignment
    let srcCentroid = [0, 0, 0], tgtCentroid = [0, 0, 0];
    for (let i = 0; i < mSrc; i++) {
        srcCentroid[0] += srcSub[i * 3]; srcCentroid[1] += srcSub[i * 3 + 1]; srcCentroid[2] += srcSub[i * 3 + 2];
    }
    srcCentroid = vecScale(srcCentroid, 1 / mSrc);
    for (let i = 0; i < mTgt; i++) {
        tgtCentroid[0] += tgtSub[i * 3]; tgtCentroid[1] += tgtSub[i * 3 + 1]; tgtCentroid[2] += tgtSub[i * 3 + 2];
    }
    tgtCentroid = vecScale(tgtCentroid, 1 / mTgt);

    const initialT = vecSub(tgtCentroid, srcCentroid);

    // Apply initial translation to source sample
    let currentSrcXYZ = new Float32Array(srcSub);
    for (let i = 0; i < mSrc; i++) {
        currentSrcXYZ[i * 3] += initialT[0];
        currentSrcXYZ[i * 3 + 1] += initialT[1];
        currentSrcXYZ[i * 3 + 2] += initialT[2];
    }

    let R_final = identity3x3();
    let t_final = [...initialT];
    let s_final = 1.0;

    // Store iterations: each entry has the cumulative R, t, s to apply to the ORIGINAL source
    const iterations = [{ R: identity3x3(), t: [...initialT], s: 1.0, numCorr: 0, delta: Infinity }];

    for (let iter = 0; iter < maxIter; iter++) {
        const step = icpStep(currentSrcXYZ, srcSubSem, tgtSub, tgtSubSem, mSrc, mTgt, semanticThresh);

        if (step.numCorr < 3) {
            if (onProgress) onProgress(iter + 1, maxIter, 0, true);
            break;
        }

        // Accumulate: R_final = step.R @ R_final, etc.
        R_final = matMul3x3(step.R, R_final);
        t_final = vecAdd(matVec3(step.R, vecScale(t_final, step.scale)), step.t);
        s_final *= step.scale;

        // Apply step transform to current source sample
        currentSrcXYZ = applyTransformToXYZ(currentSrcXYZ, mSrc, step.R, step.t, step.scale);

        iterations.push({
            R: new Float64Array(R_final),
            t: [...t_final],
            s: s_final,
            numCorr: step.numCorr,
            delta: step.delta,
        });

        if (onProgress) onProgress(iter + 1, maxIter, step.numCorr, step.converged);
        if (step.converged) break;
    }

    return { R: R_final, t: t_final, s: s_final, iterations };
}

// ==================== PLY Reconstruction (apply transform, write new PLY buffer) ====================

function buildTransformedPLYBuffer(originalBuffer, headerEnd, numVertices, numProps, R, t, scale) {
    // Clone the entire buffer
    const newBuffer = originalBuffer.slice(0);
    const dataView = new DataView(newBuffer, headerEnd);

    const bytesPerVertex = numProps * 4;
    for (let i = 0; i < numVertices; i++) {
        const byteOff = i * bytesPerVertex;
        const x = dataView.getFloat32(byteOff, true);
        const y = dataView.getFloat32(byteOff + 4, true);
        const z = dataView.getFloat32(byteOff + 8, true);

        const p = vecScale([x, y, z], scale);
        const rp = matVec3(R, p);
        dataView.setFloat32(byteOff, rp[0] + t[0], true);
        dataView.setFloat32(byteOff + 4, rp[1] + t[1], true);
        dataView.setFloat32(byteOff + 8, rp[2] + t[2], true);

        // Also transform scale properties (indices 55-57)
        for (let s = 55; s <= 57; s++) {
            const sv = dataView.getFloat32(byteOff + s * 4, true);
            // log scale in 3DGS, so add log(scale)
            dataView.setFloat32(byteOff + s * 4, sv + Math.log(scale), true);
        }

        // Transform rotation (index 58-61: rot_0..3 as quaternion wxyz)
        // We need to compose the ICP rotation with the existing quaternion
        // Convert R to quaternion, then multiply
        const existingQ = [
            dataView.getFloat32(byteOff + 58 * 4, true), // w
            dataView.getFloat32(byteOff + 59 * 4, true), // x
            dataView.getFloat32(byteOff + 60 * 4, true), // y
            dataView.getFloat32(byteOff + 61 * 4, true), // z
        ];
        const rQ = rotMatToQuat(R);
        const newQ = quatMul(rQ, existingQ);
        dataView.setFloat32(byteOff + 58 * 4, newQ[0], true);
        dataView.setFloat32(byteOff + 59 * 4, newQ[1], true);
        dataView.setFloat32(byteOff + 60 * 4, newQ[2], true);
        dataView.setFloat32(byteOff + 61 * 4, newQ[3], true);
    }

    return newBuffer;
}

function rotMatToQuat(R) {
    // R is row-major [9]
    const trace = R[0] + R[4] + R[8];
    let w, x, y, z;
    if (trace > 0) {
        const s = 0.5 / Math.sqrt(trace + 1.0);
        w = 0.25 / s;
        x = (R[7] - R[5]) * s;
        y = (R[2] - R[6]) * s;
        z = (R[3] - R[1]) * s;
    } else if (R[0] > R[4] && R[0] > R[8]) {
        const s = 2.0 * Math.sqrt(1.0 + R[0] - R[4] - R[8]);
        w = (R[7] - R[5]) / s;
        x = 0.25 * s;
        y = (R[1] + R[3]) / s;
        z = (R[2] + R[6]) / s;
    } else if (R[4] > R[8]) {
        const s = 2.0 * Math.sqrt(1.0 + R[4] - R[0] - R[8]);
        w = (R[2] - R[6]) / s;
        x = (R[1] + R[3]) / s;
        y = 0.25 * s;
        z = (R[5] + R[7]) / s;
    } else {
        const s = 2.0 * Math.sqrt(1.0 + R[8] - R[0] - R[4]);
        w = (R[3] - R[1]) / s;
        x = (R[2] + R[6]) / s;
        y = (R[5] + R[7]) / s;
        z = 0.25 * s;
    }
    const len = Math.sqrt(w * w + x * x + y * y + z * z);
    return [w / len, x / len, y / len, z / len];
}

function quatMul(a, b) {
    // [w, x, y, z]
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ];
}

// ==================== Random Transformation ====================

export function randomTransform(centroid) {
    // Uniform random rotation
    const ax = Math.random() * Math.PI * 2 - Math.PI;
    const ay = Math.random() * Math.PI * 2 - Math.PI;
    const az = Math.random() * Math.PI * 2 - Math.PI;

    const cosx = Math.cos(ax), sinx = Math.sin(ax);
    const cosy = Math.cos(ay), siny = Math.sin(ay);
    const cosz = Math.cos(az), sinz = Math.sin(az);

    const R = new Float64Array([
        cosy * cosz, sinx * siny * cosz - cosx * sinz, cosx * siny * cosz + sinx * sinz,
        cosy * sinz, sinx * siny * sinz + cosx * cosz, cosx * siny * sinz - sinx * cosz,
        -siny, sinx * cosy, cosx * cosy,
    ]);

    // Scale ~0.2x to ~5x (log-uniform)
    const scale = Math.pow(10, Math.random() * 1.4 - 0.7);
    // Uniform translation across full range
    const tUser = [(Math.random() - 0.5) * 10.0, (Math.random() - 0.5) * 10.0, (Math.random() - 0.5) * 10.0];

    // Convert centroid-relative to world: R*s*p + t_world = R*s*(p-c) + c + t_user
    const Rsc = matVec3(R, vecScale(centroid, scale));
    const t = [
        centroid[0] + tUser[0] - Rsc[0],
        centroid[1] + tUser[1] - Rsc[1],
        centroid[2] + tUser[2] - Rsc[2],
    ];

    return { R, t, scale };
}

export function customTransform(axDeg, ayDeg, azDeg, scale, tx, ty, tz, centroid) {
    const ax = axDeg * Math.PI / 180;
    const ay = ayDeg * Math.PI / 180;
    const az = azDeg * Math.PI / 180;
    scale = Math.max(0.1, Math.min(10, scale));
    const clampT = v => Math.max(-10, Math.min(10, v));

    const cosx = Math.cos(ax), sinx = Math.sin(ax);
    const cosy = Math.cos(ay), siny = Math.sin(ay);
    const cosz = Math.cos(az), sinz = Math.sin(az);

    const R = new Float64Array([
        cosy * cosz, sinx * siny * cosz - cosx * sinz, cosx * siny * cosz + sinx * sinz,
        cosy * sinz, sinx * siny * sinz + cosx * cosz, cosx * siny * sinz - sinx * cosz,
        -siny, sinx * cosy, cosx * cosy,
    ]);

    // User wants: R * s * (p - c) + c + t_user
    // = R * s * p + (c + t_user - R * s * c)
    const tUser = [clampT(tx), clampT(ty), clampT(tz)];
    const Rsc = matVec3(R, vecScale(centroid, scale));
    const t = [
        centroid[0] + tUser[0] - Rsc[0],
        centroid[1] + tUser[1] - Rsc[1],
        centroid[2] + tUser[2] - Rsc[2],
    ];
    return { R, t, scale };
}

// ==================== Registration Viewer ====================

export class RegistrationViewer {
    constructor(parentId) {
        this.parentDiv = document.getElementById(parentId);
        this.canvas = this.parentDiv.querySelector("canvas");
        this.progressDialog = this.parentDiv.querySelector("#reg-progress-dialog");
        this.progressIndicator = this.progressDialog.querySelector("#reg-progress-indicator");
        this.statusEl = this.parentDiv.querySelector("#reg-status");

        this.camera = null;
        this.renderer = null;
        this.controls = null;
        this.currentScene = null;
        this.loading = false;
        this.lastClick = new Date();
        this.interacting = false;

        // Model data
        this.models = {}; // key -> { ply, xyz, sem, numVertices, numProps }
        this.sourceKey = null;
        this.targetKey = null;

        // ICP state
        this.icpResult = null;
        this.currentIteration = -1; // -1 = before ICP
        this.appliedTransform = null; // random transform applied to source

        // Scene cache for instant mode toggle
        this._sceneCache = {};
        this._cacheTransformKey = null;

        this._initRenderer();
    }

    _initRenderer() {
        const startRadius = 8.0;
        const cameraData = new SPLAT.CameraData();
        cameraData.fx = 0.5 * this.canvas.offsetWidth;
        cameraData.fy = 0.5 * this.canvas.offsetHeight;

        this.camera = new SPLAT.Camera(cameraData);
        this.renderer = new SPLAT.WebGLRenderer(this.canvas);
        this.controls = new OrbitControls(this.camera, this.canvas, 0.0, 0.0, startRadius, false);
        this.controls.minAngle = -90;
        this.controls.maxAngle = 90;
        this.controls.minZoom = 0.001;
        this.controls.maxZoom = 10000.0;
        this.controls.zoomSpeed = 0.5;
        this.controls.panSpeed = 1.0;
        this.controls.orbitSpeed = 1.75;
        this.controls.maxPanDistance = undefined;

        this.currentScene = new SPLAT.Scene();

        this.canvas.addEventListener("mousedown", () => { this.lastClick = new Date(); this.interacting = true; });
        this.canvas.addEventListener("mouseup", () => { this.lastClick = new Date(); this.interacting = false; });

        let previousTimestamp, previousDeltaTime;
        const animate = (timestamp) => {
            let deltaTime = 0;
            if (previousTimestamp !== undefined) deltaTime = (timestamp - previousTimestamp) / 1000;
            if (deltaTime > 0.1 && previousDeltaTime !== undefined) deltaTime = previousDeltaTime;
            previousTimestamp = timestamp;
            previousDeltaTime = deltaTime;

            if (!this.interacting) {
                const timeSinceClick = (new Date() - this.lastClick) / 1000.0;
                if (timeSinceClick > 0.5) {
                    const speed = Math.min(Math.max(timeSinceClick / 4.0 - 0.5, 0.0), 1.0) * 0.3;
                    this.controls.rotateCameraAngle(speed * deltaTime, 0.0);
                }
            }

            this.controls.update();
            this.renderer.render(this.currentScene, this.camera);
            requestAnimationFrame(animate);
        };
        requestAnimationFrame(animate);
    }

    setStatus(msg) {
        if (this.statusEl) this.statusEl.textContent = msg;
    }

    async loadModel(key, url) {
        this.loading = true;
        this.progressDialog.show();
        this.progressIndicator.value = 0;
        this.setStatus(`Loading ${key}...`);

        const ply = await parsePLY(url, (p) => { this.progressIndicator.value = p * 100; });
        const xyz = extractXYZ(ply.vertexData, ply.numVertices, ply.numProps);
        const sem = extractSemantics(ply.vertexData, ply.numVertices, ply.numProps);

        // Pre-compute feature buffer (done once per model)
        const featureBuffer = this._convertToFeatureBuffer(ply.rawBuffer, ply.headerEnd, ply.numVertices, ply.numProps);

        this.models[key] = { ply, xyz, sem, numVertices: ply.numVertices, numProps: ply.numProps, url, featureBuffer };

        this.progressDialog.close();
        this.loading = false;
        this.setStatus(`Loaded ${key} (${ply.numVertices} vertices)`);
    }

    _invalidateSceneCache() {
        this._sceneCache = {};
        this._cacheTransformKey = null;
    }

    _getTransformKey(sourceR, sourceT, sourceS) {
        if (!sourceR) return "none";
        // Use first/last elements + scale for a fast key
        return `${sourceR[0].toFixed(6)}_${sourceR[4].toFixed(6)}_${sourceR[8].toFixed(6)}_${sourceT[0].toFixed(4)}_${sourceT[1].toFixed(4)}_${sourceT[2].toFixed(4)}_${sourceS.toFixed(6)}`;
    }

    async _buildScene(mode, sourceR, sourceT, sourceS, showProgress = true) {
        const scene = new SPLAT.Scene();

        // Load target
        if (this.targetKey && this.models[this.targetKey]) {
            const tgtModel = this.models[this.targetKey];
            const tgtBuffer = mode === "features" ? tgtModel.featureBuffer : tgtModel.ply.rawBuffer;
            const tgtBlob = new Blob([tgtBuffer], { type: "application/octet-stream" });
            const tgtBlobUrl = URL.createObjectURL(tgtBlob);
            if (showProgress) this.setStatus(`Loading target into scene...`);
            const tgtSplat = await SPLAT.PLYLoader.LoadAsync(tgtBlobUrl, scene, showProgress ? (p) => { this.progressIndicator.value = p * 50; } : undefined);
            URL.revokeObjectURL(tgtBlobUrl);
            const rotation = new SPLAT.Vector3(Math.PI - Math.PI / 20.0, Math.PI, Math.PI);
            tgtSplat.rotation = SPLAT.Quaternion.FromEuler(rotation);
            tgtSplat.applyRotation();
        }

        // Load source (potentially transformed)
        if (this.sourceKey && this.models[this.sourceKey]) {
            const srcModel = this.models[this.sourceKey];

            let srcBuffer;
            if (sourceR && sourceT && sourceS !== null) {
                srcBuffer = buildTransformedPLYBuffer(
                    srcModel.ply.rawBuffer, srcModel.ply.headerEnd,
                    srcModel.numVertices, srcModel.numProps,
                    sourceR, sourceT, sourceS
                );
            } else {
                // No transform: shift source on X so objects appear side by side
                srcBuffer = buildTransformedPLYBuffer(
                    srcModel.ply.rawBuffer, srcModel.ply.headerEnd,
                    srcModel.numVertices, srcModel.numProps,
                    identity3x3(), [4, 0, 0], 1.0
                );
            }

            let loadBuffer = srcBuffer;
            if (mode === "features") {
                loadBuffer = this._convertToFeatureBuffer(srcBuffer, srcModel.ply.headerEnd, srcModel.numVertices, srcModel.numProps);
            }

            const blob = new Blob([loadBuffer], { type: "application/octet-stream" });
            const blobUrl = URL.createObjectURL(blob);
            if (showProgress) this.setStatus(`Loading source into scene...`);
            const srcSplat = await SPLAT.PLYLoader.LoadAsync(blobUrl, scene, showProgress ? (p) => { this.progressIndicator.value = 50 + p * 50; } : undefined);
            URL.revokeObjectURL(blobUrl);

            const rotation = new SPLAT.Vector3(Math.PI - Math.PI / 20.0, Math.PI, Math.PI);
            srcSplat.rotation = SPLAT.Quaternion.FromEuler(rotation);
            srcSplat.applyRotation();
        }

        return scene;
    }

    async renderCombined(mode = "rgb", sourceR = null, sourceT = null, sourceS = null) {
        const transformKey = this._getTransformKey(sourceR, sourceT, sourceS);

        // Check cache: if same transform and we have this mode cached, swap instantly
        if (transformKey === this._cacheTransformKey && this._sceneCache[mode]) {
            this.currentScene = this._sceneCache[mode];
            return;
        }

        // Transform changed - invalidate cache
        if (transformKey !== this._cacheTransformKey) {
            this._sceneCache = {};
            this._cacheTransformKey = transformKey;
        }

        this.loading = true;
        this.progressDialog.show();
        this.progressIndicator.value = 0;

        const scene = await this._buildScene(mode, sourceR, sourceT, sourceS, true);

        this._sceneCache[mode] = scene;
        this.currentScene = scene;
        this.progressDialog.close();
        this.loading = false;

        // Pre-build the other mode in background so toggle is instant
        const otherMode = mode === "rgb" ? "features" : "rgb";
        if (!this._sceneCache[otherMode]) {
            this._buildScene(otherMode, sourceR, sourceT, sourceS, false).then(otherScene => {
                // Only cache if transform hasn't changed since we started
                if (this._cacheTransformKey === transformKey) {
                    this._sceneCache[otherMode] = otherScene;
                }
            });
        }
    }

    _convertToFeatureBuffer(buffer, headerEnd, numVertices, numProps) {
        const newBuffer = buffer.slice(0);
        const dataView = new DataView(newBuffer, headerEnd);
        const bytesPerVertex = numProps * 4;

        // Collect semantic values for percentile normalization
        const semValues = [[], [], []];
        for (let i = 0; i < numVertices; i++) {
            for (let c = 0; c < 3; c++) {
                semValues[c].push(dataView.getFloat32(i * bytesPerVertex + (62 + c) * 4, true));
            }
        }

        // 1st/99th percentile per channel
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

    getSourceCentroid() {
        if (!this.sourceKey || !this.models[this.sourceKey]) return [0, 0, 0];
        const src = this.models[this.sourceKey];
        return computeCentroid(src.xyz, src.numVertices);
    }

    applyRandomTransform() {
        if (!this.sourceKey || !this.models[this.sourceKey]) return null;
        const tf = randomTransform(this.getSourceCentroid());
        return this._applyTransform(tf);
    }

    applyGivenTransform(tf) {
        if (!this.sourceKey || !this.models[this.sourceKey]) return null;
        return this._applyTransform(tf);
    }

    _applyTransform(tf) {
        this.appliedTransform = tf;
        const srcModel = this.models[this.sourceKey];
        srcModel.transformedXYZ = applyTransformToXYZ(srcModel.xyz, srcModel.numVertices, tf.R, tf.t, tf.scale);
        this.icpResult = null;
        this.currentIteration = -1;
        return tf;
    }

    async runRegistration({ maxIter = 10, nSamples = 5000, semanticThresh = 0.1, onProgress = null } = {}) {
        if (!this.sourceKey || !this.targetKey) return;
        const srcModel = this.models[this.sourceKey];
        const tgtModel = this.models[this.targetKey];

        // Use the transformed XYZ if we applied a random transform
        const srcXYZ = srcModel.transformedXYZ || srcModel.xyz;

        this.setStatus("Running registration...");
        this.icpResult = runICP(srcXYZ, srcModel.sem, tgtModel.xyz, tgtModel.sem,
            srcModel.numVertices, tgtModel.numVertices,
            { maxIter, nSamples, semanticThresh, onProgress });

        this.currentIteration = this.icpResult.iterations.length - 1;
        this.setStatus(`Done: ${this.icpResult.iterations.length - 1} iterations, final scale=${this.icpResult.s.toFixed(4)}`);
        return this.icpResult;
    }

    getIterationTransform(iterIdx) {
        // Compose: first the random transform, then the ICP iteration transform
        if (!this.icpResult || iterIdx < 0 || iterIdx >= this.icpResult.iterations.length) {
            // Just the random transform (or identity)
            if (this.appliedTransform) {
                return { R: this.appliedTransform.R, t: this.appliedTransform.t, s: this.appliedTransform.scale };
            }
            return { R: identity3x3(), t: [0, 0, 0], s: 1.0 };
        }

        const icpIter = this.icpResult.iterations[iterIdx];

        if (this.appliedTransform) {
            // Compose: ICP_transform(random_transform(x))
            // = icpR * (randomR * x * randomS + randomT) * icpS + icpT
            // = (icpR * randomR) * x * (randomS * icpS) + (icpR * randomT * icpS + icpT)
            const composedR = matMul3x3(icpIter.R, this.appliedTransform.R);
            const composedS = this.appliedTransform.scale * icpIter.s;
            const rt = matVec3(icpIter.R, vecScale(this.appliedTransform.t, icpIter.s));
            const composedT = vecAdd(rt, icpIter.t);
            return { R: composedR, t: composedT, s: composedS };
        }

        return { R: icpIter.R, t: icpIter.t, s: icpIter.s };
    }

    async showIteration(iterIdx, mode = "rgb") {
        this.currentIteration = iterIdx;
        const tf = this.getIterationTransform(iterIdx);
        await this.renderCombined(mode, tf.R, tf.t, tf.s);
    }
}
